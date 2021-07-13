import json
import argparse
import re
import string
import numpy as np
from collections import namedtuple, Counter
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from scipy.optimize import linear_sum_assignment
from word2number.w2n import word_to_num
from common_utils import *

# From here through _match_numbers_if_present was originally copied from the evaluation code of DROP dataset:
# https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py

def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_word_number(text: str) -> bool:
    try:
        word_to_num(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    #TODO: this is not included in the original drop evaluation script, we need to have our own in the end anyways.
    elif _is_word_number(text):
        return str(float(word_to_num(text)))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False



def list_em(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        return 1.0
    else:
        return 0.0


def list_f1(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, gold_answers):
    scores_for_ground_truths = []
    for gold_answer in gold_answers:
        score = metric_fn(prediction, gold_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_predictions(predictions, gold_answers, example_types=None):
    """To support multiple gold annotations, `gold_answers` should be a list,
    with each item (either a string or a list) corresponding to one valid reference answer."""
    instance_eval_results = {}
    instance_eval_results_by_types = {}
    eval_funcs = {
        "list_em": list_em,
        "list_f1": list_f1
    }
    for qas_id in gold_answers:
        ref_answers = gold_answers[qas_id]
        if qas_id not in predictions:
            print(f"Missing prediction for question {qas_id}, and all scores for this question are set to zero")
            instance_eval_results[qas_id] = {
                metric: 0.0 for metric in eval_funcs.keys()
            }
        else:
            pred_answer = predictions[qas_id]
            instance_eval_results[qas_id] = {
                metric: metric_max_over_ground_truths(
                    func, pred_answer, ref_answers
                ) for metric, func in eval_funcs.items()
            }
        if example_types is not None:
            example_type = example_types[qas_id]
            if example_type not in instance_eval_results_by_types:
                instance_eval_results_by_types[example_type] = {}
            instance_eval_results_by_types[example_type][qas_id] = instance_eval_results[qas_id]

    eval_scores = {metric: np.mean([result[metric] for result in instance_eval_results.values()]) * 100
                   for metric in eval_funcs.keys()}

    if example_types is not None:
        eval_scores_by_types = {}
        for example_type, type_instance_eval_results in instance_eval_results_by_types.items():
            eval_scores_by_types[example_type] = {
                metric: np.mean([result[metric] for result in type_instance_eval_results.values()]) * 100 for metric in eval_funcs.keys()
            }
        return eval_scores, instance_eval_results, eval_scores_by_types
    else:
        return eval_scores, instance_eval_results


def evaluate_prediction_file(prediction_path, gold_path, save_instance_eval_result_path=None):
    predicted_answers = json.load(open(prediction_path, encoding="utf-8"))
    Example = namedtuple("Example", ["qas_id", "question_type", "question", "ref_answers", "answer_modality"])
    eval_examples = []
    with open(gold_path, encoding="utf8") as fin:
        for line in fin:
            line_data = json.loads(line)
            for qa in line_data["qas"]:
                qid = qa["qid"]
                question_type = qa["metadata"]["type"]
                if len(qa["answers"]) == 0:
                    print("No annotation for question {}".format(qa["qid"]))
                    continue
                ground_truth_answer = [str(item["answer"]) for item in qa["answers"]]
                # currently we only have one ground truth answer,
                # but our evaluation script would support multiple annotated answers in the future.
                ref_answers = [ground_truth_answer]
                answer_modality = set([item["modality"] for item in qa["answers"]])
                assert len(answer_modality) == 1
                answer_modality = answer_modality.pop()
                eval_examples.append(Example(qid, question_type, qa["question"], ref_answers, answer_modality))
    gold_answers = {e.qas_id: e.ref_answers for e in eval_examples}

    answer_modalities = {e.qas_id: e.answer_modality for e in eval_examples}
    modality_counts = Counter(answer_modalities.values())
    eval_scores, instance_eval_results, eval_scores_by_modalities = \
        evaluate_predictions(predicted_answers, gold_answers, answer_modalities)
    print("\n\nOverall result with different metrics: ")
    for metric, value in eval_scores.items():
        print(f"{metric}: {value}")
    print("\n\nEval results for different modalities:")
    for answer_modality in sorted(eval_scores_by_modalities.keys()):
        result = eval_scores_by_modalities[answer_modality]
        print(f"{answer_modality}")
        print(f"# of examples: {modality_counts[answer_modality]}")
        for metric, value in result.items():
            print(f"{metric}: {value}")

    single_modal_qtypes = \
        TEXT_SINGLE_HOP_QUESTION_TYPES + TABLE_SINGLE_HOP_QUESTION_TYPES + IMAGE_SINGLE_HOP_QUESTION_TYPES
    hop_types = {
        e.qas_id: "Single-modal" if e.question_type in single_modal_qtypes else "Multi-modal" for e in eval_examples
    }
    hop_type_counts = Counter(hop_types.values())
    eval_scores, instance_eval_results, eval_scores_by_hop_types = \
        evaluate_predictions(predicted_answers, gold_answers, hop_types)
    print("\n\nType\tCount\tEM\tF1")
    for hop_type in sorted(eval_scores_by_hop_types.keys()):
        result = eval_scores_by_hop_types[hop_type]
        print(f"{hop_type}\t{hop_type_counts[hop_type]}\t{result['list_em']}\t{result['list_f1']}")

    question_types = {e.qas_id: e.question_type for e in eval_examples}
    question_type_counts = Counter(question_types.values())
    eval_scores, instance_eval_results, eval_scores_by_qtypes = \
        evaluate_predictions(predicted_answers, gold_answers, question_types)
    print("\n\nType\tCount\tEM\tF1")
    for question_type in sorted(eval_scores_by_qtypes.keys()):
        result = eval_scores_by_qtypes[question_type]
        print(f"{question_type}\t{question_type_counts[question_type]}\t{result['list_em']}\t{result['list_f1']}")
    if save_instance_eval_result_path:
        with open(args.save_instance_eval_result_path, "w") as fout:
            metrics = list(instance_eval_results[eval_examples[0].qas_id].keys())
            fout.write(
                f"qid\tquestion\tquestion_type\tanswer modality\tgold_answers\tprediction\t" + "\t".join(metrics) + "\n"
            )
            for example in eval_examples:
                info_to_write = [
                    example.qas_id,
                    example.question,
                    example.question_type,
                    example.answer_modality,
                    json.dumps(example.ref_answers),
                    json.dumps(predicted_answers[example.qas_id])
                ]
                info_to_write += [str(float(instance_eval_results[example.qas_id][metric])) for metric in metrics]
                fout.write("\t".join(info_to_write) + "\n")
    return eval_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate on drop dataset")
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="predictions.json",
        help="location of the prediction file",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        default="dev.json",
        help="location of the gold file",
    )
    parser.add_argument(
        "--save_instance_eval_result_path",
        type=str,
        required=False,
        help="location to save the evaluation scores of each instance"
    )
    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path, args.gold_path, args.save_instance_eval_result_path)