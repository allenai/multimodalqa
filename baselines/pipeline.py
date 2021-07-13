import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'table_qa/')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'image_qa/')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'text_qa/')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'qtype_clf/')))

import json
import logging
import argparse
import torch
from tqdm import tqdm
from qtype_clf.inference import QuestionClassificationInferenceModel as QuestionTypePredictor
from text_qa.inferece import TextInferenceModel as TextQAPredictor
from table_qa.inference import TableInferenceModel as TableQAPredictor
from image_qa.inference import InferenceModel as ImageQAPredictor
from table_qa.data_reader import extract_answer
from common_utils import (TEXT_SINGLE_HOP_QUESTION_TYPES, TEXT_AS_FIRST_HOP_QUESTION_TYPES,
                          TEXT_AS_SECOND_HOP_QUESTION_TYPES, TABLE_SINGLE_HOP_QUESTION_TYPES,
                          TABLE_AS_FIRST_HOP_QUESTION_TYPES, TABLE_AS_SECOND_HOP_QUESTION_TYPES,
                          IMAGE_SINGLE_HOP_QUESTION_TYPES, IMAGE_AS_FIRST_HOP_QUESTION_TYPES,
                          IMAGE_AS_SECOND_HOP_QUESTION_TYPES)
from evaluate import list_em, list_f1, evaluate_prediction_file

logger = logging.getLogger(__name__)


def add_qtype_clf_args(parser):
    parser.add_argument("--qtype_model_dir",
                        help="model checkpoint dir for initializing the question type predictor.")


def add_text_qa_args(parser):
    parser.add_argument("--text_qa_model_dir",
                        help="model checkpiont dir for initializing the predictor for text qa.")


def add_table_qa_args(parser):
    parser.add_argument("--table_qa_model_dir",
                        help="model checkpiont dir for initializing the predictor for table qa.")


def add_image_qa_args(parser):
    parser.add_argument("--image_qa_model_path",
                        help="model checkpint path for initializing the predictor for image qa.")
    parser.add_argument("--image_qa_vocab",
                        help="vocab file for generating answers in image qa.")
    parser.add_argument("--image_feature_dump",
                        help="pickle dump file with precomputed feature for all image files")


def add_inference_args(parser):
    parser.add_argument("--input_file", required=True,
                        help="path to the input file in .jsonl format.")
    parser.add_argument("--output_dir", required=True,
                        help="Dir to save predicted answers and intermediate prediction results.")
    parser.add_argument("--method", default="implicit_decomp",
                        choices=["text_only", "table_only", "image_only", "auto_routing", "implicit_decomp"],
                        help="inference method.")
    parser.add_argument("--use_oracle_qtype", action="store_true",
                        help="use gold question type for sanity check")
    parser.add_argument("--context_only", action="store_true",
                        help="set question to be a null string")
    parser.add_argument("--predict_only", action="store_true",
                        help="only predict the answer, without evaluating the metrics - mainly used for the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=100, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")


def predict(args, question, context, qtype_predictor,
            text_qa_predictor, table_qa_predictor, image_qa_predictor, oracle_qa_info=None):
    result = {}
    if oracle_qa_info:
        docs = {doc["id"]: doc for doc in context["documents"]}
        text_contexts = [docs[id]["text"] for id in oracle_qa_info["metadata"]["text_doc_ids"]]
        image_contexts = [docs[id]["image"]["title"] for id in oracle_qa_info["metadata"]["image_doc_ids"]]
        table_contexts = [it["table"] for it in context["documents"] if "table" in it]
    else:
        text_contexts = [it["text"] for it in context["documents"] if "text" in it]
        table_contexts = [it["table"] for it in context["documents"] if "table" in it]
        image_contexts = [it["image"]["title"] for it in context["documents"] if "image" in it]
    if args.method == "text_only":
        result["pred_answer"] = text_qa_predictor.predict(args, question, text_contexts)
    elif args.method == "table_only":
        result["pred_answer"] = table_qa_predictor.predict(args, question, table_contexts)
    elif args.method == "image_only":
        result["pred_answer"] = image_qa_predictor.predict(question, image_contexts)
    elif args.method == "auto_routing":
        if args.use_oracle_qtype:
            question_type = oracle_qa_info["metadata"]["type"]
        else:
            if not question and oracle_qa_info is not None:
                question_type = qtype_predictor.predict(guid=0, question=oracle_qa_info["question"])
            else:
                question_type = qtype_predictor.predict(guid=0, question=question)
        if question_type in TEXT_SINGLE_HOP_QUESTION_TYPES + TEXT_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = text_qa_predictor.predict(args, question, text_contexts)
        elif question_type in TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = table_qa_predictor.predict(args, question, table_contexts)
        elif question_type in IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = image_qa_predictor.predict(question, image_contexts, question_type=question_type)
        else:
            raise ValueError("The predicted question type is not in predefined list: {}".format(question_type))
        result["pred_answer"] = pred_ans
        result["pred_question_type"] = question_type
    elif args.method == "implicit_decomp":
        if args.use_oracle_qtype:
            question_type = oracle_qa_info["metadata"]["type"]
        else:
            if not question and oracle_qa_info is not None:
                question_type = qtype_predictor.predict(guid=0, question=oracle_qa_info["question"])
            else:
                question_type = qtype_predictor.predict(guid=0, question=question)

        # the first hop for all types:
        if question_type in TEXT_SINGLE_HOP_QUESTION_TYPES + TEXT_AS_FIRST_HOP_QUESTION_TYPES:
            pred_ans = text_qa_predictor.predict(args, question, text_contexts,
                                                 question_type=question_type, hop=0, bridge_entity="")
        elif question_type in TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_FIRST_HOP_QUESTION_TYPES:
            pred_ans = table_qa_predictor.predict(args, question, table_contexts,
                                                  qtype=question_type, hop=0, bridge_entity="")
        elif question_type in IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_FIRST_HOP_QUESTION_TYPES:
            pred_ans = image_qa_predictor.predict(
                question, image_contexts, question_type=question_type)
        else:
            raise ValueError("The predicted question type is not in predefined list: {}".format(question_type))

        # the text model might predict string answer, but second hop models expect to see a list.
        if isinstance(pred_ans, str):
            pred_ans = [pred_ans]

        if question_type in TEXT_AS_FIRST_HOP_QUESTION_TYPES \
                + TABLE_AS_FIRST_HOP_QUESTION_TYPES + IMAGE_AS_FIRST_HOP_QUESTION_TYPES:
            result["first_hop_pred_answer"] = pred_ans
        else:
            result["first_hop_pred_answer"] = None

        # the second hop if necessary:
        if question_type in TEXT_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = text_qa_predictor.predict(args, question, text_contexts,
                                                 question_type=question_type, hop=1, bridge_entity=pred_ans)
            result["second_hop_pred_answer"] = pred_ans
        elif question_type in TABLE_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = table_qa_predictor.predict(args, question, table_contexts,
                                                  qtype=question_type, hop=1, bridge_entity=pred_ans)
            result["second_hop_pred_answer"] = pred_ans
        elif question_type in IMAGE_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = image_qa_predictor.predict(
                question, image_contexts, question_type=question_type, bridge_entities=pred_ans)
            result["second_hop_pred_answer"] = pred_ans
        else:
            result["second_hop_pred_answer"] = None
        result["pred_answer"] = pred_ans
        result["pred_question_type"] = question_type
    return result


def main():
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    add_qtype_clf_args(parser)
    add_text_qa_args(parser)
    add_table_qa_args(parser)
    add_image_qa_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    qtype_predictor = QuestionTypePredictor(args.qtype_model_dir, device)
    text_qa_predictor = TextQAPredictor(
        args.text_qa_model_dir, args.do_lower_case, device,
        mode="implicit_decomp" if args.method == "implicit_decomp" else "base"
    )
    table_qa_predictor = TableQAPredictor(
        args.table_qa_model_dir, device,
        mode="implicit_decomp" if args.method == "implicit_decomp" else "base"
    )
    image_qa_predictor = ImageQAPredictor(
        args.image_qa_model_path, args.image_qa_vocab, device,
        mode="context_only" if args.context_only else args.method,
        precomputed_features_info_filename=args.image_feature_dump,
        use_distractors=True
    )
    all_results = {}
    with open(args.input_file) as fin:
        for line in tqdm(fin):
            sample = json.loads(line)
            for qa in sample["qas"]:
                qid = qa["qid"]
                question = qa["question"]
                try:
                    pred_result = predict(
                        args,
                        question if not args.context_only else "",
                        sample["context"],
                        qtype_predictor, text_qa_predictor,
                        table_qa_predictor, image_qa_predictor,
                        oracle_qa_info=qa
                    )
                except Exception as e:
                    print("==" * 10)
                    print("Error message:")
                    print(e)
                    print(f"Question id: {qid}")
                    pred_result = {
                        "pred_answer": "Error in prediction",
                        "pred_question_type": None
                    }

                if not args.predict_only:
                    annotated_answers = extract_answer(qa)[3]
                    gold_qtype = qa["metadata"]["type"]
                    gold_first_hop_modality = None
                    if gold_qtype in TEXT_AS_FIRST_HOP_QUESTION_TYPES:
                        gold_first_hop_modality = "text"
                    elif gold_qtype in TABLE_AS_FIRST_HOP_QUESTION_TYPES:
                        gold_first_hop_modality = "table"
                    elif gold_qtype in IMAGE_AS_FIRST_HOP_QUESTION_TYPES:
                        gold_first_hop_modality = "image"

                    if gold_first_hop_modality:
                        if qa["metadata"]["arg1_modality"][0] == gold_first_hop_modality:
                            annotated_first_hop_answers = [it["answer"] for it in qa["metadata"]["arg1_answers"]]
                        elif qa["metadata"]["arg2_modality"][0] == gold_first_hop_modality:
                            annotated_first_hop_answers = [it["answer"] for it in qa["metadata"]["arg2_answers"]]
                        else:
                            raise ValueError
                    else:
                        annotated_first_hop_answers = None

                    if "first_hop_pred_answer" in pred_result and pred_result["first_hop_pred_answer"] is not None \
                            and annotated_first_hop_answers is not None:
                        first_hop_f1_score = \
                            list_f1(pred_result["first_hop_pred_answer"], annotated_first_hop_answers)
                    else:
                        first_hop_f1_score = 0

                    all_results[qid] = {
                        "qid": qid,
                        "question": question,
                        "gold_question_type": gold_qtype,
                        "gold_first_hop_modality": gold_first_hop_modality,
                        "annotated_first_hop_answer": annotated_first_hop_answers,
                        "annotated_answers": annotated_answers,
                        "first_hop_f1_score": first_hop_f1_score,
                        "em_score": list_em(pred_result["pred_answer"], annotated_answers[0]),
                        "f1_score": list_f1(pred_result["pred_answer"], annotated_answers[0])
                    }
                    assert len(annotated_answers) == 1
                    all_results[qid].update(pred_result)
                else:
                    all_results[qid] = {
                        "qid": qid,
                        "question": question,
                    }
                    all_results[qid].update(pred_result)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "all_prediction_info.json"), "w") as fout:
        json.dump(all_results, fout, indent=2)
    pred_answers = {qid: result["pred_answer"] for qid, result in all_results.items()}
    with open(os.path.join(args.output_dir, "predictions_.json"), "w") as fout:
        json.dump(pred_answers, fout, indent=2)

    if not args.predict_only:
        evaluate_prediction_file(os.path.join(args.output_dir, "predictions_.json"), args.input_file)


if __name__ == '__main__':
    main()
