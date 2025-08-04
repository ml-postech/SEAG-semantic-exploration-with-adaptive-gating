import pickle
from os import PathLike
import pickle
from os import PathLike
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC
from abc import ABC
from collections import defaultdict

import numpy as np
from tqdm import trange

from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example, Trace


class MCTSNode_SE(Generic[State, Action, Example]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[MCTSNode_SE]" = None,
                 fast_reward: float = 0., fast_reward_details=None,
                 is_terminal: bool = False, calc_q: Callable[[list[float]], float] = np.mean, sampling_prob: float = 1., grouped_actions: list = []):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode_SE.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[MCTSNode_SE]]' = None
        self.calc_q = calc_q
        self.sampling_prob = sampling_prob
        self.grouped_actions = grouped_actions
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)


class MCTSResult_SE(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode_SE]
    tree_state: MCTSNode_SE
    trace_in_each_iter: list[list[MCTSNode_SE]] = None
    tree_state_after_each_iter: list[MCTSNode_SE] = None
    aggregated_result: Optional[Hashable] = None
    path_for_all_iter: list[list[MCTSNode_SE]] = None


class MCTSAggregation_SE(Generic[State, Action, Example], ABC):
    def __init__(self, retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = 'edge'):
        assert weight_policy in ['edge', 'edge_inverse_depth', 'uniform', 'edge-weight-action-prob', 'edge-weight-group-length']
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(self, tree_state: MCTSNode_SE[State, Action,Example]) -> Optional[Hashable]:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode_SE[State, Action, Example]):
            if cur.state is None:
                return []
            if cur.is_terminal:
                print('cur.state: ', cur.state)
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                    print('1 answer_dict[answer]: ', answer, cur.reward, answer_dict[answer])
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / cur.depth
                elif self.weight_policy == 'uniform':
                    answer_dict[answer] += 1.0
                elif self.weight_policy == 'edge-weight-action-prob':
                    answer_dict[answer] += cur.reward * cur.sampling_prob
                    print('1 answer_dict[answer]: ', answer, cur.reward, cur.sampling_prob, answer_dict[answer])
                elif self.weight_policy == 'edge-weight-group-length':
                    answer_dict[answer] += cur.reward * len(cur.grouped_actions)
                    print('1 answer_dict[answer]: ', answer, cur.reward, cur.grouped_actions, cur.reward*len(cur.grouped_actions),answer_dict[answer])
                print('[(answer, cur.depth)]: ', [(answer, cur.depth)])
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                    print('2 answer_dict[answer]: ', answer, cur.reward, answer_dict[answer])
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / np.mean(depths)
                elif self.weight_policy == 'edge-weight-action-prob':
                    answer_dict[answer] += cur.reward * cur.sampling_prob
                    print('2 answer_dict[answer]: ', answer, cur.reward, cur.sampling_prob, answer_dict[answer])
                elif self.weight_policy == 'edge-weight-group-length':
                    answer_dict[answer] += cur.reward * len(cur.grouped_actions)
                    print('2 answer_dict[answer]: ', answer, cur.reward, cur.grouped_actions, cur.reward*len(cur.grouped_actions), answer_dict[answer])
            return cur_list

        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        print('answer_dict: ', answer_dict)
        print('max(answer_dict, key=lambda answer: answer_dict[answer]): ',max(answer_dict, key=lambda answer: answer_dict[answer]))
        answer = max(answer_dict, key=lambda answer: answer_dict[answer])
        return answer, answer_dict[answer]
        # return max(answer_dict, key=lambda answer: answer_dict[answer])

class MCTS_SE(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 early_term_threshold: float = np.inf,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 uct_with_fast_reward: bool = True,
                 aggregator: Optional[MCTSAggregation_SE] = None,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode_SE], dict] = lambda x: x.__dict__):
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.early_term_threshold = early_term_threshold
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        assert output_strategy in ['max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter',
                                   'last_terminal_iter']
        self.output_strategy = output_strategy
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[MCTSNode_SE] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode_SE]] = None
        self.root: Optional[MCTSNode_SE] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator

    def iterate(self, node: MCTSNode_SE) -> list[MCTSNode_SE]:
        path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_reward = self._back_propagate(path)
        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode_SE):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode_SE) -> list[MCTSNode_SE]:
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            node = self._uct_select(node)

    def _puct(self, node: MCTSNode_SE) -> float:
        # return node.Q + self.w_exp * node.prior_action_value * np.sqrt(len(node.parent.cum_rewards)) / (len(node.cum_rewards)+1)
        print('node.action: ', node.action, 'node.Q: ', node.Q, 'node.sampling_prob: ', node.sampling_prob, 'np.sqrt(len(node.parent.cum_rewards)): ', np.sqrt(len(node.parent.cum_rewards)), '(len(node.cum_rewards)+1): ', (len(node.cum_rewards)+1))
        print('node.Q + self.w_exp * node.sampling_prob * np.sqrt(len(node.parent.cum_rewards)) / (len(node.cum_rewards)+1): ', node.Q + self.w_exp * node.sampling_prob * np.sqrt(len(node.parent.cum_rewards)) / (len(node.cum_rewards)+1))
        # self.w_exp = 0.5
        # print('self.w_exp: ', self.w_exp)
        return node.Q + self.w_exp * node.sampling_prob * np.sqrt(len(node.parent.cum_rewards)) / (len(node.cum_rewards)+1)
        # return node.Q + self.w_exp * node.sampling_prob * np.sqrt(len(node.parent.cum_rewards)+1) / (len(node.cum_rewards)+1e-6)

    def _uct_select(self, node: MCTSNode_SE) -> MCTSNode_SE:
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            # print()
            # print('pass1')
            
            return max(node.children, key=self._puct)
        else:                       
            unvisited_children = filter(lambda x: x.state is None, node.children)
            # print('pass2')
            return max(unvisited_children, key=lambda x: x.fast_reward)
    def _expand(self, node: MCTSNode_SE):
        if node.state is None:
            node.state, aux, num_inf_for_answer_, input_tokens_count_for_answer_, output_tokens_count_for_answer_ = self.world_model.step(node.parent.state, node.action, node.grouped_actions)
            self.num_inf_for_answer += num_inf_for_answer_
            self.input_tokens_count_for_answer += input_tokens_count_for_answer_
            self.output_tokens_count_for_answer += output_tokens_count_for_answer_
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation
            node.reward, node.reward_details = self.search_config. \
                reward(node.parent.state, node.action, **node.fast_reward_details, **aux)
            node.is_terminal = self.world_model.is_terminal(node.state)

        if node.is_terminal:
            return

        children = []
        actions, num_inf_for_get_actions_, input_tokens_count_for_get_actions_, output_tokens_count_for_get_actions_ = self.search_config.get_actions(node.state)
        self.num_inf_for_get_actions += num_inf_for_get_actions_
        self.input_tokens_count_for_get_actions += input_tokens_count_for_get_actions_
        self.output_tokens_count_for_get_actions += output_tokens_count_for_get_actions_
        
        for action, group_and_prob in actions.items():
            fast_reward, fast_reward_details, input_tokens_count_for_fast_reward_ = self.search_config.fast_reward(node.state, action)
            child = MCTSNode_SE(state=None, action=action, parent=node,
                             fast_reward=fast_reward, fast_reward_details=fast_reward_details, calc_q=self.calc_q, grouped_actions=group_and_prob['grouped_actions'], sampling_prob=group_and_prob['action_prob'])
            children.append(child)
            self.num_inf_for_fast_reward += 1
            self.input_tokens_count_for_fast_reward += input_tokens_count_for_fast_reward_
            self.output_tokens_count_for_fast_reward += 2

        node.children = children

    def _simulate(self, path: list[MCTSNode_SE]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode_SE]):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
        return cum_reward

    def _dfs_max_reward(self, path: list[MCTSNode_SE]) -> tuple[float, list[MCTSNode_SE]]:
        cur = path[-1]
        if cur.is_terminal:
            # print('self.cum_reward([node.reward for node in path[1:]]), path:', self.cum_reward([node.reward for node in path[1:]]), path)
            for node_ in path:
                print(node_.action, node_.reward)
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])

    def search(self):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.root = MCTSNode_SE(state=self.world_model.init_state(), action=None, parent=None, calc_q=self.calc_q)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        self.num_inf_for_get_actions = 0
        self.num_inf_for_answer = 0
        self.num_inf_for_fast_reward = 0
        self.input_tokens_count_for_answer, self.output_tokens_count_for_answer = 0, 0
        self.input_tokens_count_for_get_actions, self.output_tokens_count_for_get_actions = 0, 0
        self.input_tokens_count_for_fast_reward, self.output_tokens_count_for_fast_reward = 0, 0
        final_answer_list = []
        self.path_for_all_iter = []
        for _ in trange(self.n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            path = self.iterate(self.root)
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

            print()
            print('iter: ', _)
            all_actions = []
            all_states = []
            print('path: ', path)
            path_for_each_iter = []
            for node in path:
                # print('node: ', node.__dict__)
                # print node.__dict without embeddings
                node_dict = deepcopy(node.__dict__)
                # del node_dict['embeddings']
                print('node_dict: ', node_dict)
                # print('action: ', node.action)
                all_actions.append(node.action)
                all_states.append(node.state)
                path_for_each_iter.append(node)
            self.path_for_all_iter.append(path_for_each_iter)
            print('all_actions: ', all_actions)
            print('all_states: ', all_states)  
            print()

            print('self.root: ', self.root.__dict__)
            print('self.early_term_threshold: ', self.early_term_threshold)
            try:
                ans_, ans_val_ = self.aggregator(self.root)
            except:
                ans_, ans_val_ = None, -100
            if ans_val_ > self.early_term_threshold:
                break

        if self.output_strategy == 'follow_max':
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                cur = max(visited_children, key=lambda x: x.reward)
            self._output_cum_reward = self.cum_reward([node.reward for node in self._output_iter[1::-1]])
        if self.output_strategy == 'max_reward':
            self._output_cum_reward, self._output_iter = self._dfs_max_reward([self.root])
            if self._output_cum_reward == -math.inf:
                self._output_iter = None

    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 log_file: Optional[str] = None,
                 **kwargs) -> MCTSResult_SE:
        MCTSNode_SE.reset_id()
        self.world_model = world_model
        self.search_config = search_config

        self.search()

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [node.action for node in self._output_iter[1:]]
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        result = MCTSResult_SE(terminal_state=terminal_state,
                            cum_reward=self._output_cum_reward,
                            trace=trace,
                            trace_of_nodes=self._output_iter,
                            tree_state=self.root,
                            trace_in_each_iter=trace_in_each_iter,
                            tree_state_after_each_iter=tree_state_after_each_iter)
        print('result 1: ', result)
        if self.aggregator is not None:
            result = MCTSResult_SE(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),
            )
        print('result 2: ', result)
        # print('self.path_for_all_iter: ', self.path_for_all_iter)
        # for path in self.path_for_all_iter:
        #     for node in path:
        #         print('node: ', node.__dict__)
        print('num_inf_for_get_actions: ', self.num_inf_for_get_actions)
        print('num_inf_for_answer: ', self.num_inf_for_answer)
        print('num_inf_for_fast_reward: ', self.num_inf_for_fast_reward)
        print('input_tokens_count_for_answer: ', self.input_tokens_count_for_answer)
        print('output_tokens_count_for_answer: ', self.output_tokens_count_for_answer)
        print('input_tokens_count_for_get_actions: ', self.input_tokens_count_for_get_actions)
        print('output_tokens_count_for_get_actions: ', self.output_tokens_count_for_get_actions)
        print('input_tokens_count_for_fast_reward: ', self.input_tokens_count_for_fast_reward)
        print('output_tokens_count_for_fast_reward: ', self.output_tokens_count_for_fast_reward)

        llm_usage_stats = {}
        llm_usage_stats['num_inf_for_get_actions'] = self.num_inf_for_get_actions
        llm_usage_stats['num_inf_for_answer'] = self.num_inf_for_answer
        llm_usage_stats['num_inf_for_fast_reward'] = self.num_inf_for_fast_reward
        llm_usage_stats['input_tokens_count_for_answer'] = self.input_tokens_count_for_answer
        llm_usage_stats['output_tokens_count_for_answer'] = self.output_tokens_count_for_answer
        llm_usage_stats['input_tokens_count_for_get_actions'] = self.input_tokens_count_for_get_actions
        llm_usage_stats['output_tokens_count_for_get_actions'] = self.output_tokens_count_for_get_actions
        llm_usage_stats['input_tokens_count_for_fast_reward'] = self.input_tokens_count_for_fast_reward
        llm_usage_stats['output_tokens_count_for_fast_reward'] = self.output_tokens_count_for_fast_reward
        print('llm_usage_stats: ', llm_usage_stats)

        return result, llm_usage_stats