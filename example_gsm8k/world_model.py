import os
import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils
from reasoners.base import Example
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize the tokenizer and model
tokenizer_for_cluster = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model_for_cluster = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")

# model_for_cluster.to('cuda:0')
local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = torch.device(f'cuda:{local_rank}')
model_for_cluster.to(device)

# Function to perform bidirectional entailment check using NLI
def check_entailment(premise, hypothesis, x_prompt_wo_shots):
    premise = x_prompt_wo_shots + premise
    # print('premise:', premise)
    hypothesis = x_prompt_wo_shots + hypothesis
    inputs = tokenizer_for_cluster(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
    # print('inputs for check_entailment:', inputs)
    with torch.no_grad():
        outputs = model_for_cluster(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    labels = ['Contradiction', 'Neutral', 'Entailment']
    return labels[predicted_class]

class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float


GSM8kState = list[SubResult]
GSM8kAction = str
GSM8kExample = str


class GSM8kPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


class GSM8kWorldModel(WorldModel[GSM8kState, GSM8kAction, GSM8kExample]):
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                base_model: LanguageModel,
                n_confidence=8,
                batch_size=2,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                early_stop_base=None,
                early_stop_threshold=1.) -> None:
        super().__init__()
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold
        self.prompt_examples = ""
        self.n_shots = 0
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token = 'Question' if 'llama3' in str(self.base_model).lower() else '\n'

    def update_example(self, example: Example, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example, prompt)
        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()

    def init_state(self) -> list:
        return []

    def step(self, state: GSM8kState, action: GSM8kAction, grouped_actions) -> tuple[GSM8kState, dict]:
        state = state.copy()
        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        
        num_inf_for_answers = 0
        input_tokens_count_for_answers, output_tokens_count_for_answers = 0, 0
        
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):

                action_random_select = random.choice(grouped_actions)
                print('action_random_select: ', action_random_select)
                with io.StringIO() as f:
                    f.write(self.prompt_examples)
                    f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")
                    for idx, (q, a, _) in enumerate(state):
                        f.write(
                            self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")
                        f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
                    f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1,
                                                                    sub_idx=len(state) + 1) + " " + action_random_select + "\n") # action_random_selectaction_random_selectaction_random_selectaction_random_selectaction_random_selectaction_random_select
                    f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))
                    model_input = f.getvalue()

                print('Answer generation')
                print(model_input)
        
                stop = min(start + self.batch_size, stop1)
                num = stop - start
                output_, input_tokens_count, output_tokens_count = self.base_model.generate([model_input] * num,
                                                                                            hide_input=True,
                                                                                            do_sample=True,
                                                                                            temperature=self.temperature,
                                                                                            top_k=self.top_k,
                                                                                            top_p=self.top_p,
                                                                                            eos_token_id=self.eos_token)
                outputs = output_.text
                num_inf_for_answers += num
                input_tokens_count_for_answers += input_tokens_count
                output_tokens_count_for_answers += output_tokens_count
                print('input_tokens_count_for_answers, output_tokens_count_for_answers in step')
                print(input_tokens_count_for_answers, output_tokens_count_for_answers)
                for output in outputs:
                    result = output.strip()
                    if len(result) > 500:
                        continue
                    
                    answer = utils.retrieve_answer(result)               
                    if answer is not None: # entailment check        
                        answer_dict[answer].append(result)

            # Early stop if confidence is high enough
            if len(answer_dict) == 0: # no answer yet
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 2 and max_len == len(sorted_answer_dict[1][1]):
                    pass  # Tie with the second best answer
                else:
                    break

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        aux = {'confidence': confidence}
        return state, aux, num_inf_for_answers, input_tokens_count_for_answers, output_tokens_count_for_answers

    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False