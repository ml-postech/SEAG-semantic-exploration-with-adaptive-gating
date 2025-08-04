import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils
import numpy as np
from world_model import check_entailment

# Bidirectional entailment clustering with probabilities
def bidirectional_entailment_clustering(sentences, probabilities, x_prompt_wo_shots):
    # Initialize clusters with the first sentence and its probability
    clusters = [[(sentences[0], probabilities[0])]]
    cluster_probs = [probabilities[0]]  # Initialize the probability sum for the first cluster

    # Loop over sentences starting from the second one
    for i in range(1, len(sentences)):
        added_to_cluster = False
        for cluster_idx, cluster in enumerate(clusters):
            # Take the first sentence from the current cluster as representative
            representative, _ = cluster[0]
            # Perform bidirectional entailment check
            if check_entailment(representative, sentences[i], x_prompt_wo_shots) == 'Entailment' and check_entailment(sentences[i], representative, x_prompt_wo_shots) == 'Entailment':
                cluster.append((sentences[i], probabilities[i]))
                cluster_probs[cluster_idx] += probabilities[i]  # Add the probability to the cluster's total
                added_to_cluster = True
                break

        # If sentence is semantically distinct, create a new cluster
        if not added_to_cluster:
            clusters.append([(sentences[i], probabilities[i])])
            cluster_probs.append(probabilities[i])

    return clusters, cluster_probs


import io
import re
from typing import TypedDict, Optional
import numpy as np

from world_model import GSM8kState, GSM8kAction, GSM8kPromptDict
from reasoners import SearchConfig, LanguageModel

class GSM8kUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str

class GSM8kConfig(SearchConfig):
    def __init__(self,
                base_model: LanguageModel,
                useful_prompt: GSM8kUsefulPrompt,
                n_actions=4,
                batch_size=1,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                reward_alpha=0.5,
                reward_confidence_default=0.8,
                depth_limit=5,
                force_terminating_on_depth_limit=True,
                force_overall_prompt_on_overall_question=True,
                force_overall_question_on_overall_prompt=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.useful_prompt = useful_prompt
        self.example = ''
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None
        self.prompt_examples = ""
        self.n_shots = 0
        self.eos_token = 'Answer' if 'llama3' in str(self.base_model).lower() else '\n'

    def update_example(self, example: str, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example, prompt=prompt)

        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()

        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            print(self.example)
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example, flags=re.DOTALL)[1]
            self.overall_question = re.match('.*((([A-Z].* (calculate|how|what|find|true or false))|((Calculate|How|What|Find|True or false))).*)$', self.example, flags=re.DOTALL)[1]

    def get_actions(self, state: GSM8kState, ) -> list[GSM8kAction]:
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])
            model_input = f.getvalue()
        print("Action generation")
        print(model_input)
        
        # input(">")

        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0 if at_depth_limit else self.temperature
        outputs = []
        outputs_probs = []
        num_inf_for_get_actions = 0
        input_tokens_count_for_get_actions, output_tokens_count_for_get_actions = 0, 0
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            output_, input_tokens_count, output_tokens_count = self.base_model.generate([model_input] * n_samples,
                                                                                        hide_input=True,
                                                                                        do_sample=True,
                                                                                        top_k=self.top_k,
                                                                                        top_p=self.top_p,
                                                                                        temperature=temperature,
                                                                                        eos_token_id=self.eos_token,
                                                                                        output_log_probs=True)
            outputs += output_.text
            # outputs_probs += output_.log_prob
            if 'llama2' in str(self.base_model).lower():
                output_log_prob = output_.log_prob
                # print('output_log_prob:', output_log_prob)
                output_log_prob = output_log_prob.cpu().numpy().squeeze()
                outputs_probs.append(output_log_prob)
            else:
                outputs_probs += output_.log_prob
            num_inf_for_get_actions += n_samples
            input_tokens_count_for_get_actions += input_tokens_count
            output_tokens_count_for_get_actions += output_tokens_count
            
        all_outputs_prob = []
        for log_prob in outputs_probs:
            overall_prob = np.exp(np.sum(log_prob) / len(log_prob))
            all_outputs_prob.append(overall_prob)

        outputs = [output.strip() for output in outputs]
        
        if at_depth_limit:
            outputs = [self.prompt["overall_question_prefix"] + ' ' + output for output in outputs]
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.prompt["overall_question_prefix"] in output:
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if self.overall_question.lower() == output.lower():
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question

                
        with io.StringIO() as f:
            f.write("Question: ")
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(self.prompt["overall_question_prefix"] + " ")
            prompt_for_cluster = f.getvalue()

        clusters, cluster_probs = bidirectional_entailment_clustering(outputs, all_outputs_prob, prompt_for_cluster)
        
        # continuous_prob (using log_prob) 
        # normalized_probs = np.array(cluster_probs) / np.sum(cluster_probs)

        # discretized prob (when cannot using log_prob) 
        total_sentences = sum(len(cluster) for cluster in clusters)
        cluster_sizes = np.array([len(cluster) for cluster in clusters])
        normalized_probs = cluster_sizes / total_sentences

        outputs_dict = {}
        for idx, (cluster, prob) in enumerate(zip(clusters, normalized_probs)):
            print(f"Cluster {idx+1} (Normalized Probability: {prob:.4f}):")
            # select the sentence with the highest probability in the cluster
            selected_sentence, _ = max(cluster, key=lambda x: x[1])
            groued_sentences = [sentence for sentence, _ in cluster]
            outputs_dict[selected_sentence] = {}
            outputs_dict[selected_sentence]['action_prob'] = prob
            outputs_dict[selected_sentence]['grouped_actions'] = groued_sentences

        for key in outputs_dict:
            print(key, outputs_dict[key]['grouped_actions'], outputs_dict[key]['action_prob'])    

        return outputs_dict, num_inf_for_get_actions, input_tokens_count_for_get_actions, output_tokens_count_for_get_actions

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue()
        print('useful reward')
        print(model_input)
        logits, input_tokens_count = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])
        logits = logits[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {'r_useful': useful_prob}, input_tokens_count

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful, 'r_conf': r_conf}

    def reward(self, state: GSM8kState, action: GSM8kAction,
            r_useful: float = None,
            confidence: float = None) -> tuple[float, dict]:
        # return confidence, {'r_conf': confidence}
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)