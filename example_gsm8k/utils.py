import re
from typing import Optional, Union
from reasoners.algorithm import BeamSearchResult

from reasoners.base import AlgorithmOutput
from collections import Counter
import numpy as np

def retrieve_answer(output: Union[list, str, AlgorithmOutput]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, AlgorithmOutput):
        if (result := getattr(output, 'aggregated_result', None)) is not None:
            return result
        output = output.terminal_state
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*The answer is .*?([ $.0-9,\-=]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer

def retrieve_answer_bs(output: Union[list, str, BeamSearchResult]) -> Optional[str]:

    if isinstance(output, BeamSearchResult):
        output = output.terminal_node.state
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*The answer is .*?([ $.0-9,\-=]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer

def retrieve_answer_from_dataset(answer: Union[str, dict]) -> str:
    if isinstance(answer, dict):
        answer = answer['answer']
    return re.match(r'[\S\s]*#### (.*)$', answer)[1].replace(',', '').replace(' ', '')


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer


def cot_sc_extractor(algo_output, sc=True):
    # aggregate the results from multiple reasoning chains with majority vote
    answers_set = [retrieve_answer(x) for x in algo_output]
    answers = [x for x in answers_set if x is not None]
    counter = Counter(answers)
    if counter == {}:
        return None
    return answers_set, answers, counter.most_common(1)[0][0]

def eval_output(answer, output):
        if output is None:
            return False
        try:
            output = int(output)
            answer = int(answer)
            return output == answer
        except ValueError:
            pass
        try:
            output = float(output)
            answer = float(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer

def eval_output_tup(answer, output):
        if output is None:
            return False
        try:
            if isinstance(output, tuple):
                output = int(output[0])
            else:
                output = int(output)
            answer = int(answer)
            return output == answer
        except ValueError:
            pass
        try:
            if isinstance(output, tuple):
                output = float(output[0])
            else:
                output = float(output)
            answer = float(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer
    
def compute_entropy(counts, base=2):
    """
    Compute entropy manually from a dictionary of counts.

    Parameters:
    counts (dict): A dictionary where keys are outcomes and values are counts of occurrences.
    base (int or float): The base of the logarithm for entropy (default is 2 for bits).

    Returns:
    float: The entropy value.
    """
    total_count = sum(counts)
    
    # If total_count is zero, return 0 entropy (empty distribution)
    if total_count == 0:
        return 0.0

    # Normalize counts to probabilities
    probs = [count / total_count for count in counts]

    # Compute entropy using the formula: H = -sum(p * log(p))
    entropy_value = -sum(p * np.log(p) / np.log(base) for p in probs if p > 0)

    return entropy_value