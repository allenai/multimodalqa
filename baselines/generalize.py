
import os
import glob
from pipeline import *
from evaluate import evaluate_predictions


def read_squad(file):
    examples = []
    with open(file) as fin:
        data = json.load(fin)["data"]
        for doc in data:
            for para in doc["paragraphs"]:
                for qa in para["qas"]:
                    examples.append(
                        {"id": qa["id"],
                         "question": qa["question"],
                         "paragraphs": [para["context"]],
                         "answers": [[ans["text"]] for ans in qa["answers"]]}
                    )
    return examples


def read_hotpot(file):
    examples = []
    with open(file) as fin:
        data = json.load(fin)
        for example in data:
            paragraphs = [" ".join(it[1]) for it in example["context"]]
            examples.append(
                {"id": example["_id"],
                 "question": example["question"],
                 "paragraphs": paragraphs,
                 "answers": [[example["answer"]]]}
            )
    return examples


def read_wikitable(file):
    table_files = glob.glob(os.path.join(os.path.dirname(file), "../csv/**/*.tsv"))
    tables = {}
    for table_file in table_files:
        name = table_file.split("../")[-1][:-3] + "csv"
        print(name)
        with open(table_file) as fin:
            header_line = fin.readline()
            header = [{"column_name": name} for name in header_line.rstrip("\n").split("\t")]
            rows = []
            for line in fin:
                row = [{"text": value} for value in line.rstrip("\n").split("\t")]
                rows.append(row)
            table = {"header": header, "table_rows": rows}
            tables[name] = table
    examples = []
    with open(file) as fin:
        fin.readline()
        for line in fin:
            line_info = line.strip().split("\t")
            examples.append(
                {"id": line_info[0],
                 "question": line_info[1],
                 "table": tables[line_info[2]],
                 "answers": [[line_info[3]]]}
            )
    return examples


def read_wikisql(file):
    table_file = os.path.join(os.path.dirname(file), "dev.tables.jsonl")
    tables = {}
    with open(table_file) as fin:
        for line in fin:
            data = json.loads(line)
            header = [{"column_name": str(it)} for it in data["header"]]
            rows = [[{"text": str(val)} for val in row] for row in data["rows"]]
            tables[data["id"]] = {"header": header, "table_rows": rows}
    examples = []
    with open(file) as fin:
        for line in fin:
            data = json.loads(line)
            examples.append(
                {"id": str(len(examples)),
                 "question": data["question"],
                 "table": tables[data["table_id"]],
                 "answers": [[str(it) for it in data["answer"]]]}
            )
    return examples


def read_vqa(file):
    examples = []
    with open(file) as fin:
        data = json.load(fin)
        for question in data["questions"]:
            examples.append(
                {"id": question["question_id"], "question": question["question"]}
            )
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    add_qtype_clf_args(parser)
    add_text_qa_args(parser)
    add_table_qa_args(parser)
    add_image_qa_args(parser)

    parser.add_argument("--eval_qtype", action="store_true")
    parser.add_argument("--eval_qa", action="store_true")
    parser.add_argument("--dataset", choices=["squad", "hotpot", "wikitable", "wikisql", "vqa", "hybridqa"], required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    qtype_predictor = QuestionTypePredictor(args.qtype_model_dir, device)
    if args.eval_qtype:
        if args.dataset == "squad":
            examples = read_squad(args.input_file)
            gold_qtype = "text"
        elif args.dataset == "hotpot":
            examples = read_hotpot(args.input_file)
            gold_qtype = "text"
        elif args.dataset == "wikitable":
            examples = read_wikitable(args.input_file)
            gold_qtype = "table"
        elif args.dataset == "wikisql":
            examples = read_wikisql(args.input_file)
            gold_qtype = "table"
        elif args.dataset == "vqa":
            examples = read_vqa(args.input_file)
            gold_qtype = "image"
        # elif args.dataset == "hybridqa":
        #     examples = read_hybridqa(args.input_file)
        else:
            raise ValueError
        correct, total = 0, 0
        for example in examples:
            pred_qtype = qtype_predictor.predict(guid=0, question=example["question"])
            print("==" * 10)
            print(example["question"])
            print(f"Orig pred: {pred_qtype}")
            if pred_qtype in TEXT_SINGLE_HOP_QUESTION_TYPES + TEXT_AS_SECOND_HOP_QUESTION_TYPES:
                pred_qtype = "text"
            elif pred_qtype in TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_SECOND_HOP_QUESTION_TYPES:
                pred_qtype = "table"
            elif pred_qtype in IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_SECOND_HOP_QUESTION_TYPES:
                pred_qtype = "image"
            else:
                raise ValueError
            if gold_qtype == pred_qtype:
                correct += 1
            total += 1
            print(f"Pred: {pred_qtype}")
            print(f"Gold: {gold_qtype}")
        print(f"{correct}\t{total}\t{correct / total}")

    if args.eval_qa:
        predictions, gold_answers = {}, {}
        if args.dataset in ["squad", "hotpot"]:
            text_qa_predictor = TextQAPredictor(
                args.text_qa_model_dir, args.do_lower_case, device,
                mode="implicit_decomp" if args.method == "implicit_decomp" else "base"
            )
            if args.dataset == "squad":
                examples = read_squad(args.input_file)
                for example in tqdm(examples):
                    pred_qtype = qtype_predictor.predict(guid=0, question=example["question"])
                    pred_ans = text_qa_predictor.predict(
                        args, example["question"], example["paragraphs"], question_type=pred_qtype, hop=0, bridge_entity="")
                    predictions[example["id"]] = pred_ans
                    gold_answers[example["id"]] = example["answers"]
            else:
                examples = read_hotpot(args.input_file)
                for example in tqdm(examples):
                    pred_qtype = qtype_predictor.predict(guid=0, question=example["question"])
                    pred_ans = text_qa_predictor.predict(
                        args, example["question"], example["paragraphs"], question_type=pred_qtype, hop=0, bridge_entity="")
                    predictions[example["id"]] = pred_ans
                    gold_answers[example["id"]] = example["answers"]
        elif args.dataset in ["wikitable", "wikisql"]:
            table_qa_predictor = TableQAPredictor(
                args.table_qa_model_dir, device,
                mode="implicit_decomp" if args.method == "implicit_decomp" else "base"
            )
            examples = read_wikitable(args.input_file) if args.dataset == "wikitable" else read_wikisql(args.input_file)
            for example in tqdm(examples):
                pred_qtype = qtype_predictor.predict(guid=0, question=example["question"])
                pred_ans = table_qa_predictor.predict(args, example["question"], example["table"], qtype=pred_qtype, hop=0, bridge_entity="")
                predictions[example["id"]] = pred_ans
                gold_answers[example["id"]] = example["answers"]
        # elif args.dataset in ["hybridqa"]:
        #     text_qa_predictor = TextQAPredictor(
        #         args.text_qa_model_dir, args.do_lower_case, device,
        #         mode="implicit_decomp" if args.method == "implicit_decomp" else "base"
        #     )
        #     table_qa_predictor = TableQAPredictor(
        #         args.table_qa_model_dir, device,
        #         mode="implicit_decomp" if args.method == "implicit_decomp" else "base"
        #     )
        else:
            raise ValueError
        eval_scores, _ = evaluate_predictions(predictions, gold_answers)
        print(eval_scores)