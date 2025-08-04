import sys
sys.path.append('.')
from typing import Type, Callable, Optional, Literal

import numpy as np
from datetime import datetime

from reasoners.benchmark import GSM8KEvaluator

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS_SE, MCTSNode_SE, MCTSAggregation_SE

from world_model import GSM8kWorldModel, GSM8kPromptDict
from search_config import GSM8kConfig, GSM8kUsefulPrompt
import utils

from reasoners.lm import ExLlamaModel
import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
from reasoners.lm import  Llama2Model, Llama3Model
import utils
from typing import Literal
import fire
import transformers

import torch
from tqdm import tqdm
import pickle
import copy

from collections import defaultdict

class CoTReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
        
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        inputs = f"Please solve the following question step by step and conclude by saying \"The answer is\".\n\n" + inputs
        print(example)
        do_sample = True
        
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
            
        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, BardCompletionModel) or isinstance(self.base_model, ClaudeModel):
            eos_token_id = []
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [108]
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama2Model):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama3Model):
            eos_token_id = ["\n\n", ".\n", "\n", ".\n\n", "\nQ"]
        elif self.base_model.model.config.architectures[0] == 'InternLM2ForCausalLM':
            eos_token_id = [364, 402, 512, 756]
        elif self.base_model.model.config.architectures[0] == 'Qwen2ForCausalLM':
            eos_token_id = [198, 271, 382, 624, 151645]
        else:
            assert isinstance(self.base_model.model, transformers.LlamaForCausalLM)###need to be modified for other model
            eos_token_id = [13]
            
        outputs = list()
        num_inf_for_get_actions, num_inf_for_answer, num_inf_for_fast_reward = 0, 0, 0
        
        for i in range((self.n_sc - 1) // self.bs + 1):
            local_bs = min(self.bs, self.n_sc - i * self.bs)
            
            outputs += self.base_model.generate([inputs] * local_bs,
                                            hide_input=True,
                                            do_sample=do_sample,
                                            temperature=self.temperature,
                                            eos_token_id=eos_token_id)[0].text
            num_inf_for_answer += local_bs
            
        outputs= [o.strip() if o.strip().endswith(".") else o.strip() + "." for o in outputs]
        print(outputs)
        return outputs, num_inf_for_get_actions, num_inf_for_answer, num_inf_for_fast_reward

def node_visualizer(x: MCTSNode_SE):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}

def rap_gsm8k(base_model: LanguageModel,
            prompt: GSM8kPromptDict,
            useful_prompt: GSM8kUsefulPrompt,
            search_algo: Type[SearchAlgorithm] = MCTS_SE,
            resume: int = 0,
            n_action: int = 4,
            n_confidence: int = 8,
            depth_limit: int = 5,
            force_terminating_on_depth_limit: bool = True,
            batch_size: int = 2,
            temperature: float = 0.8,
            early_stop_base: int = 2,
            early_stop_threshold: float = 0.5,
            reward_alpha: float = 0.5,
            reward_confidence_default: float = 0.8,
            cum_reward: Callable[[list[float]], float] = np.mean,
            calc_q: Callable[[list[float]], float] = max,
            log_dir: Optional[str] = None,
            disable_log: bool = False,
            disable_tqdm: bool = False,
            output_trace_in_each_iter: bool = True,
            early_term_threshold: float = np.inf,
            aggregate: bool = True,
            adaptive_gating: bool = True,
            entropy_thres: float = 1.5,
            n_sc: int = 10,
            **search_algo_params):

    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/gsm8k_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume >= 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
            
    if aggregate:
        aggregator = MCTSAggregation_SE(utils.retrieve_answer, weight_policy='edge-weight-group-length')
    else:
        aggregator = None

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm,
                        'output_trace_in_each_iter': output_trace_in_each_iter,
                        'node_visualizer': node_visualizer, 'aggregator': aggregator, 'early_term_threshold': early_term_threshold}
    world_model = GSM8kWorldModel(base_model=base_model,
                                n_confidence=n_confidence, batch_size=batch_size, temperature=temperature,
                                early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = GSM8kConfig(base_model=base_model, useful_prompt=useful_prompt,
                        n_actions=n_action, batch_size=batch_size, temperature=temperature,
                        reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                        force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    cot_reasoner = CoTReasoner(base_model, temperature=temperature, n_sc=n_sc, bs=batch_size)
    

    disable_tqdm = disable_tqdm or \
        (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
    
    ## GSM8K dataset
    import datasets
    dataset = datasets.load_dataset('gsm8k', 'main', split='test')
    dataset = np.array(dataset)
    
    correct_count = 0
    error_cases = list()
    
    input_processor = lambda x: x["question"]
    
    # if sample_prompt_type == "cot":
    init_cot_prompt = "example_gsm8k/prompts/cot.json"
    with open(init_cot_prompt) as f:
        init_cot_prompt = json.load(f)
    
    prompt_dict = {}
    num_shot = 1
    examples = init_cot_prompt["cot_pool"][:num_shot]
    prompt_dict["cot"] = "".join(examples) + init_cot_prompt["prefix"]

    # elif sample_prompt_type == "rap":
    ret = copy.deepcopy(prompt)
    ret['interactive_examples'], ret['useful_examples'] = zip(*random.sample(list(zip(ret['interactive_examples'],
                                                                                    ret['useful_examples'])),
                                                                                    k=num_shot))

    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
                                    desc='gsm8k', disable=disable_tqdm)):
        total_num_inf = 0
        if adaptive_gating:
            algo_output, num_inf_for_get_actions, num_inf_for_answer, num_inf_for_fast_reward = cot_reasoner(input_processor(example),
                                            prompt=prompt_dict)
            total_num_inf = num_inf_for_get_actions + num_inf_for_answer + num_inf_for_fast_reward
            answers_set, answers, output = utils.cot_sc_extractor(algo_output) # answers_set including "None"

            counts = {value: answers_set.count(value) for value in set(answers_set)}
            total_count = sum(counts.values())
            probs = [count / total_count for count in counts.values()]
            H = utils.compute_entropy(probs, base=2)
            if H < entropy_thres:
                answer = utils.retrieve_answer_from_dataset(example)
                correct = utils.eval_output(answer, output)
                correct_count += correct
                accuracy = correct_count / (i + 1)
                log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=}; '\
                            f'{accuracy=:.3f} ({correct_count}/{i + 1}); '\
                            f'{num_inf_for_fast_reward=}, {num_inf_for_answer=}, {num_inf_for_get_actions=}, {total_num_inf=}'
                tqdm.write(log_str)

                if (not disable_log) and \
                    (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                    with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                        print(log_str, file=f)
                
                    with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
                        pickle.dump(algo_output, f)
            else:
                try:
                    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
                    seed_num = 12306
                    random.seed(seed_num)
                    np.random.seed(seed_num)
                    torch.manual_seed(seed_num)
                    torch.cuda.manual_seed(seed_num)
                    torch.backends.cudnn.deterministic = True
                    algo_output, num_inf_for_get_actions_2, num_inf_for_answer_2, num_inf_for_fast_reward_2 = reasoner(input_processor(example),
                                                prompt=ret)
                    num_inf_for_get_actions += num_inf_for_get_actions_2
                    num_inf_for_answer += num_inf_for_answer_2
                    num_inf_for_fast_reward += num_inf_for_fast_reward_2
                    total_num_inf = num_inf_for_get_actions + num_inf_for_answer + num_inf_for_fast_reward
                    output = utils.retrieve_answer(algo_output)
                    answer = utils.retrieve_answer_from_dataset(example)
                    correct = utils.eval_output_tup(answer, output)
                    correct_count += correct
                    accuracy = correct_count / (i + 1)
                    log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=}; '\
                                f'{accuracy=:.3f} ({correct_count}/{i + 1}); '\
                                f'{num_inf_for_fast_reward=}, {num_inf_for_answer=}, {num_inf_for_get_actions=}, {total_num_inf=}'
                    tqdm.write(log_str)

                    if (not disable_log) and \
                        (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                        with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                            print(log_str, file=f)
                    
                        with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
                            pickle.dump(algo_output, f)
                except Exception as e:
                    print(f"Error in case {resume + i + 1}: {e}")
                    error_cases.append([resume + i + 1, e])
                    print(error_cases)
    
            # save list of error cases as a single pickle file
            with open(os.path.join(log_dir, 'error_cases.pkl'), 'wb') as f:
                pickle.dump(error_cases, f)
        else:
            try:
                reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
                seed_num = 12306
                random.seed(seed_num)
                np.random.seed(seed_num)
                torch.manual_seed(seed_num)
                torch.cuda.manual_seed(seed_num)
                torch.backends.cudnn.deterministic = True
                algo_output, num_inf_for_get_actions, num_inf_for_answer, num_inf_for_fast_reward = reasoner(input_processor(example),
                                            prompt=ret)
                total_num_inf = num_inf_for_get_actions + num_inf_for_answer + num_inf_for_fast_reward
                output = utils.retrieve_answer(algo_output)
                answer = utils.retrieve_answer_from_dataset(example)

                correct = utils.eval_output_tup(answer, output)
                correct_count += correct
                accuracy = correct_count / (i + 1)
                log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=}; '\
                            f'{accuracy=:.3f} ({correct_count}/{i + 1}); '\
                            f'{num_inf_for_fast_reward=}, {num_inf_for_answer=}, {num_inf_for_get_actions=}, {total_num_inf=}'
                tqdm.write(log_str)

                if (not disable_log) and \
                    (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                    with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                        print(log_str, file=f)
                
                    with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
                        pickle.dump(algo_output, f)
            except Exception as e:
                print(f"Error in case {resume + i + 1}: {e}")
                error_cases.append([resume + i + 1, e])
                print(error_cases)

        # save list of error cases as a single pickle file
        with open(os.path.join(log_dir, 'error_cases.pkl'), 'wb') as f:
            pickle.dump(error_cases, f)

    # print(accuracy)

if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA2_CKPTS", None)
    llama_3_ckpts = os.environ.get("LLAMA3_CKPTS", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama', 'llama-3'] = 'llama-3',
            llama_ckpts: str = llama_ckpts,
            llama_2_ckpts: str = llama_2_ckpts,
            llama_3_ckpts: str = llama_3_ckpts,
            llama_size: str = '13B',
            llama_cpp_path: str = None,
            llama_cpp_n_batch: int = 512,
            hf_path: str = 'meta-llama/Llama-2-13b-hf',
            hf_peft_path: Optional[str] = None,
            hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
            hf_load_awq_path: Optional[str] = None,
            exllama_model_dir: str = 'WizardMath-13B-V1.0-GPTQ',
            exllama_lora_dir: Optional[str] = None,
            exllama_mem_map: Optional[str] = None,
            batch_size: int = 1,
            useful_prompt: str = 'example_gsm8k/prompts/useful_examples.json',
            prompt: str = 'example_gsm8k/prompts/prompt_pool.json',
            disable_log: bool = False,
            disable_tqdm: bool = False,
            **kwargs):

        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        with open(prompt) as f:
            prompt = json.load(f)
        if base_lm in ['llama', 'llama-2', 'llama-3']:    
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        if base_lm == 'llama':
            from reasoners.lm import LlamaModel
            base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            from reasoners.lm import LlamaCppModel
            base_model = LlamaCppModel(llama_cpp_path, n_batch=llama_cpp_n_batch)
        elif base_lm == 'llama-2':
            from reasoners.lm import Llama2Model
            base_model = Llama2Model(llama_2_ckpts, llama_size, max_batch_size=batch_size,max_seq_len=4096)
        elif base_lm == 'llama-3':
            from reasoners.lm import Llama3Model
            base_model = Llama3Model(llama_3_ckpts, llama_size, max_batch_size=batch_size,max_seq_len=4096)
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                    max_batch_size=batch_size, max_new_tokens=200, max_seq_length=3072)
        else:
            assert False, f'cannot resolve {base_lm=}'
        
        rap_gsm8k(base_model=base_model,
                useful_prompt=useful_prompt,
                prompt=prompt,
                batch_size=batch_size,
                disable_log=disable_log or local_rank != 0,
                disable_tqdm=disable_tqdm or local_rank != 0,
                **kwargs)

    fire.Fire(main)
