import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'table_qa/')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'image_qa/')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'text_qa/')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'qtype_clf/')))

import json
import argparse
import torch
from tqdm import tqdm
from qtype_clf.inference import QuestionClassificationInferenceModel as QuestionTypePredictor
from text_qa.inferece import TextInferenceModel as TextQAPredictor
from table_qa.inference import TableInferenceModel as TableQAPredictor
from image_qa.inference import InferenceModel as ImageQAPredictor
from common_utils import (TEXT_SINGLE_HOP_QUESTION_TYPES, TEXT_AS_FIRST_HOP_QUESTION_TYPES,
                          TEXT_AS_SECOND_HOP_QUESTION_TYPES, TABLE_SINGLE_HOP_QUESTION_TYPES,
                          TABLE_AS_FIRST_HOP_QUESTION_TYPES, TABLE_AS_SECOND_HOP_QUESTION_TYPES,
                          IMAGE_SINGLE_HOP_QUESTION_TYPES, IMAGE_AS_FIRST_HOP_QUESTION_TYPES,
                          IMAGE_AS_SECOND_HOP_QUESTION_TYPES)
from common_utils import read_jsonl
from evaluate import list_em, list_f1, evaluate_prediction_file


def add_qtype_clf_args(parser):
    parser.add_argument(
        "--qtype_model_dir",
        help="Model checkpoint dir for initializing the question type predictor."
    )


def add_text_qa_args(parser):
    parser.add_argument(
        "--text_qa_model_dir",
        help="Model checkpoint dir for initializing the predictor for text qa."
    )


def add_table_qa_args(parser):
    parser.add_argument(
        "--table_qa_model_dir",
        help="Model checkpoint dir for initializing the predictor for table qa."
    )


def add_image_qa_args(parser):
    parser.add_argument(
        "--image_qa_model_path",
        help="Model checkpoint path for initializing the predictor for image qa."
    )
    parser.add_argument(
        "--image_qa_vocab",
        help="Vocab file for generating answers in image qa."
    )
    parser.add_argument(
        "--image_features_dir",
        help="Dir that stores the precomputed feature for all image files"
    )
    parser.add_argument(
        "--raw_image_dir",
        help="If `image_features_dir` is not provided, "
             "we will find the images in this dir and compute its features online."
    )
    parser.add_argument(
        "--vilbert_dir",
        default="downloads/vilbert/",
        help="Dir that stores vilbert model checkpoint and config file."
    )


def add_inference_args(parser):
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the data dir, which should contains the example file (dev/test), "
             "along with all the context files: MMQA_texts.jsonl, MMQA_tables.jsonl, MMQA_images.jsonl."
             "Running `scripts/download.sh` should download all the necessary files."
    )
    parser.add_argument(
        "--test_file",
        default="MMQA_dev.jsonl",
        help="Which file to use: MMQA_dev.jsonl or MMQA_test.jsonl."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Dir to save predicted answers and intermediate prediction results."
    )
    parser.add_argument(
        "--method",
        default="implicit_decomp",
        choices=["auto_routing", "implicit_decomp"],
        help="Inference method: "
             "auto_routing will predict which modality will the final answer come from, "
             "and then call the corresponding model to answer the question;"
             "implicit_decomp will predict the modalities of the first & second hop (if necessary), "
             "and then call the corresponding models in sequence to answer the question."
    )
    parser.add_argument(
        "--use_oracle_qtype",
        action="store_true",
        help="Use gold question type for sanity check"
    )
    parser.add_argument(
        "--context_only",
        action="store_true",
        help="Set question to be a null string, used only for the context_only baselien in our paper."
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="only predict the answer, without evaluating the metrics - mainly used for the test set."
    )
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--max_query_length",
        default=100,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length."
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to consider in the text qa model."
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another."
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Whether not to use CUDA when available"
    )


def predict(args, input, qtype_predictor,
            text_qa_predictor, table_qa_predictor, image_qa_predictor):
    result = {}
    if args.use_oracle_qtype or args.context_only:
        question_type = input["gold_question_type"]
    else:
        question_type = qtype_predictor.predict(guid=0, question=input["question"])

    if args.method == "auto_routing":
        if question_type in TEXT_SINGLE_HOP_QUESTION_TYPES + TEXT_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = text_qa_predictor.predict(args, input["question"], input["text_context"])
        elif question_type in TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = table_qa_predictor.predict(args, input["question"], input["table_context"])
        elif question_type in IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_SECOND_HOP_QUESTION_TYPES:
            pred_ans = image_qa_predictor.predict(input["question"], input["image_context"],
                                                  question_type=question_type)
        else:
            raise ValueError("The predicted question type is not in predefined list: {}".format(question_type))
        result["pred_answer"] = pred_ans
        result["pred_question_type"] = question_type
    elif args.method == "implicit_decomp":
        # the first hop for all types:
        if question_type in TEXT_SINGLE_HOP_QUESTION_TYPES + TEXT_AS_FIRST_HOP_QUESTION_TYPES:
            first_hop_ans = text_qa_predictor.predict(args, input["question"], input["text_context"],
                                                      question_type=question_type, hop=0, bridge_entity="")
        elif question_type in TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_FIRST_HOP_QUESTION_TYPES:
            first_hop_ans = table_qa_predictor.predict(args, input["question"], input["table_context"],
                                                       question_type=question_type, hop=0, bridge_entity="")
        elif question_type in IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_FIRST_HOP_QUESTION_TYPES:
            first_hop_ans = image_qa_predictor.predict(
                input["question"], input["image_context"], question_type=question_type)
        else:
            raise ValueError("The predicted question type is not in predefined list: {}".format(question_type))

        # the text_qa model might predict string answer, but second hop models expect to see a list.
        if isinstance(first_hop_ans, str):
            first_hop_ans = [first_hop_ans]

        # the second hop if necessary:
        if question_type in TEXT_AS_SECOND_HOP_QUESTION_TYPES:
            second_hop_ans = text_qa_predictor.predict(args, input["question"], input["text_context"],
                                                       question_type=question_type, hop=1, bridge_entity=first_hop_ans)
        elif question_type in TABLE_AS_SECOND_HOP_QUESTION_TYPES:
            second_hop_ans = table_qa_predictor.predict(args, input["question"], input["table_context"],
                                                        question_type=question_type, hop=1, bridge_entity=first_hop_ans)
        elif question_type in IMAGE_AS_SECOND_HOP_QUESTION_TYPES:
            second_hop_ans = image_qa_predictor.predict(
                input["question"], input["image_context"], question_type=question_type, bridge_entities=first_hop_ans)
        else:
            second_hop_ans = None

        result["pred_first_hop_answer"] = first_hop_ans
        result["pred_second_hop_answer"] = second_hop_ans
        result["pred_answer"] = second_hop_ans if second_hop_ans is not None else first_hop_ans
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
    print('Running pipeline with the following arguments:')
    for arg, val in vars(args).items():
        print(f"  {arg}: {val}")

    print("Loading checkpoints ...")
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
        checkpoint_path=args.image_qa_model_path,
        vilbert_dir=args.vilbert_dir,
        img_info_path=os.path.join(args.data_dir, "MMQA_images.jsonl"),
        vocab_path=args.image_qa_vocab,
        device=device,
        mode="context_only" if args.context_only else args.method,
        img_features_dir=args.image_features_dir,
        use_distractors=True
    )

    print("Loading the text, table and image contexts ...")
    text_docs = read_jsonl(os.path.join(args.data_dir, "MMQA_texts.jsonl"))
    text_docs = {doc["id"]: doc for doc in text_docs}
    table_docs = read_jsonl(os.path.join(args.data_dir, "MMQA_tables.jsonl"))
    table_docs = {doc["id"]: doc for doc in table_docs}

    print(f"Loading test examples from {args.test_file} ...")
    all_examples = read_jsonl(os.path.join(args.data_dir, args.test_file))
    all_results = {}

    print("Running inference for the test examples ...")
    for example in tqdm(all_examples):
        qid = example["qid"]
        input = {
            "qid": example["qid"],
            "question": example["question"] if not args.context_only else "",
            "text_context": [text_docs[doc_id]["text"] for doc_id in example["metadata"]["text_doc_ids"]],
            "table_context": table_docs[example["metadata"]["table_id"]]["table"],
            "image_context": example["metadata"]["image_doc_ids"],
            "gold_question_type": example["metadata"]["type"] if args.use_oracle_qtype or args.context_only else None
        }
        try:
            pred_result = predict(
                args,
                input,
                qtype_predictor,
                text_qa_predictor,
                table_qa_predictor,
                image_qa_predictor
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

        all_results[qid] = {
            "qid": qid,
            "question": example["question"],
        }
        all_results[qid].update(pred_result)

        if not args.predict_only:
            # Currently we only have one ground truth answer.
            # Even if there are multiple entries in example["answers"],
            # the whole list should be regarded as one ref answer.
            gold_answer = [str(it["answer"]) for it in example["answers"]]
            gold_question_type = example["metadata"]["type"]

            if gold_question_type in TEXT_AS_FIRST_HOP_QUESTION_TYPES:
                gold_first_hop_modality = "text"
            elif gold_question_type in TABLE_AS_FIRST_HOP_QUESTION_TYPES:
                gold_first_hop_modality = "table"
            elif gold_question_type in IMAGE_AS_FIRST_HOP_QUESTION_TYPES:
                gold_first_hop_modality = "image"
            else:
                gold_first_hop_modality = None

            if example["metadata"]["intermediate_answers"]:
                gold_first_hop_answer = [str(it["answer"]) for it in example["metadata"]["intermediate_answers"][0]]
            else:
                gold_first_hop_answer = []
            if "pred_first_hop_answer" in pred_result and pred_result[
                "pred_first_hop_answer"] and gold_first_hop_answer:
                first_hop_em_score = \
                    list_em(pred_result["pred_first_hop_answer"], gold_first_hop_answer)
                first_hop_f1_score = \
                    list_f1(pred_result["pred_first_hop_answer"], gold_first_hop_answer)
            else:
                first_hop_em_score = 0
                first_hop_f1_score = 0

            all_results[qid].update({
                "gold_question_type": gold_question_type,
                "gold_first_hop_modality": gold_first_hop_modality,
                "gold_first_hop_answer": gold_first_hop_answer,
                "gold_answer": gold_answer,
                "first_hop_em_score": first_hop_em_score,
                "first_hop_f1_score": first_hop_f1_score,
                "em_score": list_em(pred_result["pred_answer"], gold_answer),
                "f1_score": list_f1(pred_result["pred_answer"], gold_answer)
            })

    print("Writing predictions to files ...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "all_prediction_info.json"), "w") as fout:
        json.dump(all_results, fout, indent=2)
    pred_answers = {qid: result["pred_answer"] for qid, result in all_results.items()}
    with open(os.path.join(args.output_dir, "predictions_.json"), "w") as fout:
        json.dump(pred_answers, fout, indent=2)

    if not args.predict_only:
        print("Evaluating the metrics ...")
        evaluate_prediction_file(
            os.path.join(args.output_dir, "predictions_.json"),
            os.path.join(args.data_dir, args.test_file)
        )


if __name__ == '__main__':
    main()
