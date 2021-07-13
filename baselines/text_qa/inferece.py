from common_utils import TEXT_SINGLE_HOP_QUESTION_TYPES, TEXT_AS_FIRST_HOP_QUESTION_TYPES, TEXT_AS_SECOND_HOP_QUESTION_TYPES, process_question_for_implicit_decomp
import torch

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling_roberta import RobertaForQuestionAnsweringYesNo
from modeling_bert import BertForQuestionAnsweringYesNo
import collections
import argparse

from tqdm import tqdm
import json
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from squad_preprocess import SquadFeatures, SquadProcessor, squad_convert_examples_to_features, SquadResult
from squad_metrics import compute_predictions_logits_inference
import jsonlines

import sys
sys.path.append('..')


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits"])

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class TextInferenceModel:
    def __init__(self,
                 model_dir,
                 do_lower_case,
                 device, mode):
        print('initializing Reader...', flush=True)
        self.config = AutoConfig.from_pretrained(model_dir)
        if self.config.model_type == "bert":
            self.model = BertForQuestionAnsweringYesNo.from_pretrained(
                model_dir)
        elif self.config.model_type == "roberta":
            self.model = RobertaForQuestionAnsweringYesNo.from_pretrained(
                model_dir)
        else:
            raise ValueError(
                f"Unsupported model type: {self.config.model_type}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, do_lower_case=do_lower_case)
        self.device = device
        self.mode = mode

        self.model.to(device)
        self.model.eval()

    def convert_question_paragraph_input(self, example_idx, question, paragraphs):
        squad_style_data = {'data': [], 'version': '2.0'}

        for paragraph in paragraphs:
            example_id = example_idx
            title = "dummy_title"
            question_text = question
            squad_example = {'context': paragraph, 'qas': [
                {'question': question_text, 'id': example_id}]}
            squad_style_data["data"].append(
                {'title': title, 'paragraphs': [squad_example]})
        return squad_style_data

    def predict(self, args, question, paragraphs, example_idx=0, question_type='Simple(Image)', hop=0, bridge_entity='', n_best_num=1):
        if not paragraphs:
            return "Empty answer: no paragraphs are provided."

        if self.mode == 'implicit_decomp':
            question = process_question_for_implicit_decomp(
                question, question_type, hop, bridge_entity)
        elif self.mode == 'context_only':
            question = ''

        squad_style_data = self.convert_question_paragraph_input(
            example_idx, question, paragraphs)
        processor = SquadProcessor()
        examples = processor._create_examples(
            squad_style_data["data"], "eval", tqdm_enabled=False)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            return_dataset="pt",
            tqdm_enabled=False
        )

        all_results = []
        self.model.eval()

        batch = tuple(t.to(self.device) for t in dataset.tensors)
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if self.config.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]
            outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    switch_logits = output[3]
                    end_top_index = output[4]
                    cls_logits = output[5]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        switch_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits, switch_logits = output
                    result = SquadResult(
                        unique_id, start_logits, end_logits, switch_logits)

                all_results.append(result)

        output_prediction_file = "inference_predictions.json"
        predictions = compute_predictions_logits_inference(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file=output_prediction_file,
            verbose_logging=False,
            version_2_with_negative=False,
            null_score_diff_threshold=0.0,
            tokenizer=self.tokenizer,
            n_best_num=1
        )

        final_result = predictions[example_idx]

        return final_result


def load_input_examples(input_file_name):
    exampels = []
    # {"question": question, "paragraphs": text_context, "hop": number_of_hop, "bridge_entity": gold_bridge_entity}
    input_data = read_jsonlines(input_file_name)
    for example in tqdm(input_data):
        raw_context = example["context"]["documents"]
        qas = example["qas"]
        for qa in qas:
            docs = {doc["id"]: doc for doc in raw_context}
            paragraphs = [docs[id]["text"]
                          for id in qa["metadata"]["text_doc_ids"]]
            question = qa["question"]
            q_id = qa["qid"]
            q_type = qa["metadata"]["type"]
            answers_original = qa["answers"]
            bridge_entity = qa["metadata"]["bridge_entity"] if "bridge_entity" in qa["metadata"] else ""

            if q_type in TEXT_SINGLE_HOP_QUESTION_TYPES:
                exampels.append({"question": question, "paragraphs": paragraphs, "q_id": q_id,
                                 "answers": answers_original, "bridge_entity": "", "hop": 1, "q_type": q_type})

            elif q_type in TEXT_AS_FIRST_HOP_QUESTION_TYPES:
                exampels.append({"question": question, "paragraphs": paragraphs, "q_id": q_id,  "answers": bridge_entity if len(
                    bridge_entity) > 0 else answers_original, "bridge_entity": "", "hop": 1, "q_type": q_type})

            elif q_type in TEXT_AS_SECOND_HOP_QUESTION_TYPES:
                exampels.append({"question": question, "paragraphs": paragraphs, "q_id": q_id,
                                 "answers": answers_original, "bridge_entity": bridge_entity, "hop": 2, "q_type": q_type})
            else:
                print("skip example:{0} (q_type: {1})".format(q_id, q_type))
    return exampels


# TODO: updata the inference codes for sanity check.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="path to model")
    parser.add_argument("--reader_path", default=None, type=str, required=True,
                        help="path to model")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--eval_batch_size", default=8,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--mode", default="implicit_decomp",
                        type=str, help="Total batch size for predictions.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="model type")

    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    # load input data
    input_examples = load_input_examples(args.input_file)
    # Initialize the text QA model
    reader = TextInferenceModel(
        args.reader_path, args.do_lower_case, device, mode=args.mode)
    
    # run predictions
    final_results = {}
    for example in tqdm(input_examples):
        example_idx = example["q_id"]
        paragraphs = example["paragraphs"]
        question = example["question"]
        if args.mode == "implicit_decomp":
            hop = example["hop"]
            bridge_entity = example["bridge_entity"]
            q_type = example["q_type"]
            results = reader.predict(
                args, question, paragraphs, example_idx=example_idx, question_type=q_type, hop=hop, bridge_entity=bridge_entity)
            final_results[example_idx] = results
        else:
            results = reader.predict(
                args, question, paragraphs, example_idx=example_idx)
            final_results[example_idx] = results

    # save predictions
    with open('reader_predicted_results.json', 'w') as outfile:
        json.dump(final_results, outfile)


if __name__ == "__main__":
    main()
