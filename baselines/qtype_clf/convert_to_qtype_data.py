import jsonlines
import numpy as np
import json
import os
import json
from tqdm import tqdm
import csv
import argparse

import sys
sys.path.append('..')
from common_utils import ALL_QUESTION_TYPES
q_types = ALL_QUESTION_TYPES

def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def convert_qa_data_into_qtype_data(input_data, split="train"):
    question_type_dic_list = []
    type_counter = {}
    type_counter_original = {}
    for qa in tqdm(input_data):
        question = qa["question"]
        q_id = qa["qid"]
        q_type = qa["metadata"]["type"]            
        type_counter_original.setdefault(q_type, 0)
        type_counter_original[q_type] += 1
        if q_type not in q_types:
            print("the question type does not match.")
            print(q_type)
            continue
        type_counter.setdefault(q_type, 0)
        type_counter[q_type] += 1
        answers = [answer["answer"] for answer in qa["answers"]]
        question_type_dic_list.append(
            {"q_id": q_id, "question": question, "q_type": q_type, "answers": answers})
    assert type_counter_original == type_counter
    return question_type_dic_list


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_fp",
                        default=None, type=str, required=True)
    parser.add_argument("--dev_data_fp",
                        default=None, type=str, required=True)
    parser.add_argument("--test_data_fp",
                        default=None, type=str)
    parser.add_argument("--output_data_dir",
                        default=None, type=str, required=True)

    args = parser.parse_args()
    train_data = read_jsonlines(args.train_data_fp)
    eval_data = read_jsonlines(args.dev_data_fp)
    if args.test_data_fp is not None:
        test_data = read_jsonlines(args.test_data_fp)

    train_qtype_data = convert_qa_data_into_qtype_data(train_data, "train")
    dev_qtype_data = convert_qa_data_into_qtype_data(eval_data, "dev")
    if args.test_data_fp is not None:
        test_qtype_data = convert_qa_data_into_qtype_data(test_data, "test")

    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    with open(os.path.join(args.output_data_dir, 'qtype_clf_train_data.tsv'), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for item in train_qtype_data:
            tsv_writer.writerow(
                [item["question"], item["q_type"], item["q_id"]])

    with open(os.path.join(args.output_data_dir, 'qtype_clf_dev_data.tsv'), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for item in dev_qtype_data:
            tsv_writer.writerow(
                [item["question"], item["q_type"], item["q_id"]])

    if args.test_data_fp is not None:
        with open(os.path.join(args.output_data_dir, 'qtype_clf_test_data.tsv'), 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for item in test_qtype_data:
                tsv_writer.writerow(
                    [item["question"], item["q_type"], item["q_id"]])


if __name__ == "__main__":
    main()