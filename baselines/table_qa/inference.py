import json
import torch
import argparse

from run_table_qa import to_list
from transformers import AutoConfig, AutoTokenizer
from models import BertQaForTable as BertForQuestionAnswering
from models import RobertaQaForTable as RobertaForQuestionAnswering
from data_reader import convert_examples_to_features, extract_answer, TableQaExample
from table_qa_utils import compute_predictions_with_cell_logits, PredictionResultForCells
from common_utils import process_question_for_implicit_decomp


class TableInferenceModel:
    def __init__(self, model_dir, device=None, mode="base"):
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.config.model_type == "bert":
            self.model = BertForQuestionAnswering.from_pretrained(model_dir)
        elif self.config.model_type == "roberta":
            self.model = RobertaForQuestionAnswering.from_pretrained(model_dir)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
        self.model.to(device)
        self.model.eval()
        assert mode in {'base', 'implicit_decomp', 'context_only'}
        self.mode = mode

    def predict(self, args, question, table, qtype=None, hop=None, bridge_entity=None):
        if not table:
            return "Empty answer: no table context is provided."
        # if not question:
        #     return "Empty answer: no question context is provided."
        if isinstance(table, list):
            table = table[0]

        # support json str input
        if isinstance(table, str):
            try:
                table = json.loads(table)
            except:
                return "Cannot parse the table string into json object."

        if self.mode == "implicit_decomp":
            question = process_question_for_implicit_decomp(
                question, qtype, hop, bridge_entity, sep_token=self.tokenizer.sep_token
            )
        elif self.mode == "context_only":
            question = ""

        qas_id = "ID: " + question
        example = TableQaExample(
            qas_id=qas_id,
            question_text=question,
            table_data=table
        )
        features, dataset = convert_examples_to_features([example], self.tokenizer,
                                                         max_seq_length=args.max_seq_length,
                                                         doc_stride=args.doc_stride,
                                                         max_query_length=args.max_query_length,
                                                         is_training=False, verbose=False)

        batch = tuple(t.to(self.device) for t in dataset.tensors)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "cell_token_masks": batch[3],
            }

            if self.config.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            example_indices = batch[4]

            # XLNet and XLM use more arguments for their predictions
            if self.config.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

            outputs = self.model(**inputs)

        all_results = []
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                raise ValueError("Currently some models are not supported.")
            else:
                cell_scores, answer_type_logits = output
                cell_scores = cell_scores[: len(eval_feature.cell_spans)]
                result = PredictionResultForCells(unique_id, cell_scores, answer_type_logits)
            all_results.append(result)

        predictions = compute_predictions_with_cell_logits(
            [example],
            features,
            all_results
        )

        answer = predictions[qas_id]
        return answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="dir path with model checkpoints.")
    parser.add_argument("--input", required=True,
                        help="input file")
    parser.add_argument("--do_lower_case", default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = TableInferenceModel(args.model_dir, device)

    with open(args.input) as fin:
        for line in fin:
            obj = json.loads(line)
            table = obj["context"]["documents"][0]["table"]
            for qa in obj["qas"]:
                question = qa["question"]
                answer_text = extract_answer(qa)[2]
                pred_answer = model.predict(args, question, table)
                print("=" * 10)
                print(pred_answer)
                print(answer_text)


if __name__ == '__main__':
    main()