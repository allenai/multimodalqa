from transformers.data.metrics.squad_metrics import *
from transformers.data.metrics.squad_metrics import _get_best_indexes, _compute_softmax
from data_reader import ans_type2id_map
from common_utils import extract_numbers_from_str


ans_id2type_map = {v: k for k, v in ans_type2id_map.items()}


class PredictionResultForCells(object):
    def __init__(self, unique_id, cell_scores, answer_type_logits):
        self.unique_id = unique_id
        self.cell_scores = cell_scores
        self.answer_type_logits = answer_type_logits


def compute_predictions_with_cell_logits(
    all_examples,
    all_features,
    all_results,
    output_prediction_file=None,
    cell_selection_thresh=0.5
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file is not None:
        logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_prediction_info = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prediction_info = {}
        ans_type_logits = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            for idx, logit in enumerate(result.answer_type_logits):
                ans_type_logits.append((ans_id2type_map[idx], logit))
        ans_type_logits.sort(key=lambda x: x[1], reverse=True)
        pred_ans_type = ans_type_logits[0][0]

        prediction_info["answer_type_preds"] = ans_type_logits

        if pred_ans_type in ["yes", "no"]:
            final_answer = pred_ans_type
        else:
            cell_string_and_probs = []
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                for cell_idx, cell_prob in enumerate(result.cell_scores):
                    orig_cell_pos = feature.cell_to_orig_pos_map[cell_idx]
                    if orig_cell_pos == (-1, -1):
                        continue
                    else:
                        cell_string = example.table_cells[orig_cell_pos[0]][orig_cell_pos[1]]
                    cell_string_and_probs.append((cell_string, cell_prob))

            prediction_info["cell_probs"] = sorted(cell_string_and_probs, key=lambda x: x[1], reverse=True)

            selected_cell_strings = \
                [cell_string for cell_string, prob in cell_string_and_probs if prob >= cell_selection_thresh]
            if not selected_cell_strings:
                cell_string_and_probs.sort(key=lambda x: x[1], reverse=True)
                selected_cell_strings = [cell_string_and_probs[0][0]]

            if pred_ans_type == "count":
                final_answer = str(len(selected_cell_strings))
            elif pred_ans_type in ["sum", "mean"]:
                answer_numbers = []
                for cell_string in selected_cell_strings:
                    cell_numbers = extract_numbers_from_str(cell_string)
                    if len(cell_numbers) == 1:
                        answer_numbers.append(cell_numbers[0])
                if answer_numbers:
                    sum_answer = sum(answer_numbers)
                    final_answer = str(sum_answer) if pred_ans_type == "sum" else str(sum_answer / len(answer_numbers))
                else:
                    final_answer = selected_cell_strings
            else:
                final_answer = selected_cell_strings

        prediction_info["final_answer"] = final_answer

        all_predictions[example.qas_id] = final_answer
        all_prediction_info[example.qas_id] = prediction_info

    if output_prediction_file is not None:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        output_prediction_info_file = output_prediction_file + ".info.json"
        with open(output_prediction_info_file, "w") as writer:
            writer.write(json.dumps(all_prediction_info, indent=4) + "\n")

    return all_predictions