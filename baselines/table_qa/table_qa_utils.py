from transformers.data.metrics.squad_metrics import *
from transformers.data.metrics.squad_metrics import _get_best_indexes, _compute_softmax
from data_reader import ans_type2id_map
from common_utils import extract_numbers_from_str


ans_id2type_map = {v: k for k, v in ans_type2id_map.items()}


class PredictionResult(object):
    def __init__(self, unique_id, start_logits, end_logits, answer_type_logits,
                 start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.answer_type_logits = answer_type_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def compute_predictions_with_yesno_logits(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    tokenizer,
    do_lower_case,
    output_prediction_file=None,
    output_nbest_file=None,
    output_null_log_odds_file=None,
    verbose_logging=False,
    version_2_with_negative=False,
    null_score_diff_threshold=None,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file is not None:
        logger.info("Writing predictions to: %s" % (output_prediction_file))
    if output_nbest_file is not None:
        logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []

        # if the highest prob the model predicts for yes / no answer is larger than the highest prob for span answer,
        # add yes / no as the top predictions. Otherwise, we will not consider yes / no.
        answer_type_predictions = []
        for feature in features:
            result = unique_id_to_result[feature.unique_id]
            for idx, logit in enumerate(result.answer_type_logits):
                if idx == 0:
                    answer_type_predictions.append(("span", logit))
                elif idx == 1:
                    answer_type_predictions.append(("yes", logit))
                elif idx == 2:
                    answer_type_predictions.append(("no", logit))
        answer_type_predictions.sort(key=lambda x: x[1], reverse=True)

        for answer_type, logit in answer_type_predictions:
            if answer_type in ["yes", "no"] and answer_type not in seen_predictions:
                final_text = answer_type
                nbest.append(_NbestPrediction(text=final_text, start_logit=logit / 2, end_logit=logit / 2))
                seen_predictions[final_text] = True
            else:
                break

        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file is not None:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file is not None:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative and output_null_log_odds_file is not None:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


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