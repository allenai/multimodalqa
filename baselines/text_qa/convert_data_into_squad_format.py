import argparse
import jsonlines
import random
import os
import json
from tqdm import tqdm
import sys
sys.path.append('..')
from common_utils import ALL_QUESTION_TYPES
from common_utils import TEXT_SINGLE_HOP_QUESTION_TYPES, TEXT_AS_FIRST_HOP_QUESTION_TYPES, TEXT_AS_SECOND_HOP_QUESTION_TYPES, process_question_for_implicit_decomp

def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def convert_qa_data_into_squad_format_implicit_decomp_final(qa_data, doc_data, split="train", mode="implicit_decomp", model_type="bert"):
    squad_style_data = {'data': [], 'version': 'v2.0'}
    count_distractor = 0
    count_gold = 0
    skipped_questions = []
    q_types = {}
    hp_bridge = 0
    q_count = 0
    missing_qtype = {}
    # create doc_data dict
    doc_data_dict = {}
    for doc in doc_data:
        doc_data_dict[doc["id"]] = {"text": doc["text"], "title": doc["title"]}

    for qa in tqdm(qa_data):
        context = {}
        q_count += 1
        supporting_context = qa["supporting_context"]
        question = qa["question"]
        q_id = qa["qid"]
        text_doc_ids = qa["metadata"]["text_doc_ids"]

        for text_doc_id in text_doc_ids:
            context[text_doc_id] = doc_data_dict[text_doc_id]

        q_type = qa["metadata"]["type"]
        q_types.setdefault(q_type, 0)
        q_types[q_type] += 1
        answers_original = qa["answers"]
        text_answers = []
        for answer_orig in answers_original:
            if answer_orig["type"] == "string" and answer_orig["modality"] == "text":
                text_answers.append(
                    {"answer_start": answer_orig["text_instances"][0]["start_byte"] if len(answer_orig["text_instances"]) > 0 else -1,
                        "doc_id": answer_orig["text_instances"][0]["doc_id"] if len(answer_orig["text_instances"]) > 0 else -1,
                        "text": answer_orig["answer"], "type": answer_orig["type"]})
            if answer_orig["type"] == "yesno" and answer_orig["modality"] == "text":
                text_answers.append(
                        {"doc_id": answer_orig["text_instances"][0]["doc_id"] if len(answer_orig["text_instances"]) > 0 else -1,
                        "text": answer_orig["answer"], "type": answer_orig["type"]})

        # collect supporting facts in text modality.
        sp_paragraph_id = [sc["doc_id"]
                            for sc in supporting_context if sc["doc_part"] == "text"]

        # Skip train questions that are only related to Table or Image Question.
        if split == "train" and len(sp_paragraph_id) == 0:
            skipped_questions.append(q_id)
            continue

        # Add examples for second hop or one-hop questions.
        if q_type in TEXT_AS_FIRST_HOP_QUESTION_TYPES:
            hop = 0
            updated_question = process_question_for_implicit_decomp(
                question, q_type, hop=hop, bridge_entity="", sep_token="</s>" if model_type == "roberta" else "[SEP]")
            
            # collect intermediate answers and their supporting doc
            intermediate_answers = qa["metadata"]["intermediate_answers"]
            sp_to_intermediate_answers = {}
            for answer_data in intermediate_answers:
                for answer in answer_data:
                    text_instances = answer["text_instances"]
                    for sp in text_instances:
                        sp_to_intermediate_answers[sp["doc_id"]] = {"start_byte": sp["start_byte"], "text": sp["text"]}

            for paragraph_id, paragraph in context.items():
                if paragraph_id not in sp_paragraph_id:
                    squad_example = {'context': paragraph["text"],
                                        'qas': [{'question': updated_question, 'is_impossible': True,
                                                'answers': [] if split == "train" else text_answers,
                                                'id': "{}_distractor".format(q_id)}]}
                    squad_style_data["data"].append(
                        {"title": "{0}_distractor".format(paragraph["title"]), 'paragraphs': [squad_example]})
                    count_distractor += 1
                elif paragraph_id in sp_paragraph_id and paragraph_id in sp_to_intermediate_answers:
                    squad_example = {'context': paragraph["text"],
                                        'qas': [{'question': updated_question, 'is_impossible': False,
                                                'answers': sp_to_intermediate_answers[paragraph_id],
                                                'id': q_id}]}
                    squad_style_data["data"].append(
                        {"title": paragraph["title"], 'paragraphs': [squad_example]})

        elif q_type in TEXT_SINGLE_HOP_QUESTION_TYPES:
            if mode == "implicit_decomp":
                hop = 0
                updated_question = process_question_for_implicit_decomp(
                    question, q_type, hop=hop, bridge_entity="", sep_token="</s>" if model_type == "roberta" else "[SEP]")
            elif mode == "context_only":
                updated_question = "NULL"
            elif mode == "auto":
                updated_question = question
            # Add each train / dev example (Q and P pairs)
            for paragraph_id, paragraph in context.items():
                if paragraph_id not in sp_paragraph_id:
                    squad_example = {'context': paragraph["text"],
                                        'qas': [{'question': updated_question, 'is_impossible': True,
                                                'answers': [] if split == "train" else text_answers,
                                                'id': "{}_distractor".format(q_id)}]}
                    squad_style_data["data"].append(
                        {"title": "{0}_distractor".format(paragraph["title"]), 'paragraphs': [squad_example]})
                    count_distractor += 1

                # add supporting facts
                else:
                    final_answers = []
                    for ans in text_answers:
                        if ans["type"] == "yesno":
                            final_answers.append(
                                {'text': ans["text"], "answer_start": -1})
                        elif paragraph["text"][ans["answer_start"]:ans["answer_start"]+len(ans["text"])] == ans["text"]:
                            final_answers.append(
                                {'text': ans["text"], "answer_start": ans["answer_start"]})
                        elif paragraph["text"][ans["answer_start"]:ans["answer_start"]+len(ans["text"])] != ans["text"] and paragraph["text"].find(ans["text"]) > -1:
                            final_answers.append(
                                {'text': ans["text"], "answer_start": paragraph["text"].find(ans["text"])})
                        else:
                            continue
                    if len(final_answers) > 0:
                        squad_example = {'context': paragraph["text"],
                                            'qas': [{'question': updated_question, 'is_impossible': True if final_answers[0]["text"] in ["yes", "no"] else False,
                                                    'answers': final_answers,
                                                    'id': q_id}]}
                        squad_style_data["data"].append(
                            {"title": paragraph["title"], 'paragraphs': [squad_example]})
                        count_gold += 1
                    elif len(final_answers) == 0 and split == "train":
                        continue
                    elif len(final_answers) == 0 and split == "dev":
                        squad_example = {'context': paragraph["text"],
                                            'qas': [{'question': updated_question, 'is_impossible': True,
                                                    'answers': [{"text": answer["text"], "answer_start": answer["answer_start"]} for answer in text_answers],
                                                    'id': q_id}]}
                        squad_style_data["data"].append(
                            {"title": paragraph["title"], 'paragraphs': [squad_example]})
                        count_distractor += 1

        elif q_type in TEXT_AS_SECOND_HOP_QUESTION_TYPES:
            if mode == "implicit_decomp":
                hop = 1
                intermediate_answers = qa["metadata"]["intermediate_answers"]
                intermediate_answers_text = []
                for answer_data in intermediate_answers:
                    for answer in answer_data:
                        intermediate_answers_text.append(answer["answer"])
                        bridge_entity = ";".join(intermediate_answers_text)
                        if split == "train" and random.random() < 0.3:
                            bridge_entity = ""
                    updated_question = process_question_for_implicit_decomp(
                        question, q_type, hop=hop, bridge_entity=bridge_entity, sep_token="</s>" if model_type == "roberta" else "[SEP]")
            elif mode == "context_only":
                updated_question = "NULL"
            elif mode == "auto":
                updated_question = question
            # Add each train / dev example (Q and P pairs)
            for paragraph_id, paragraph in context.items():
                if paragraph_id not in sp_paragraph_id:
                    squad_example = {'context': paragraph["text"],
                                        'qas': [{'question': updated_question, 'is_impossible': True,
                                                'answers': [] if split == "train" else text_answers,
                                                'id': "{}_distractor".format(q_id)}]}
                    squad_style_data["data"].append(
                        {"title": "{0}_distractor".format(paragraph["title"]), 'paragraphs': [squad_example]})
                    count_distractor += 1
                # add supporting facts
                else:
                    # update final answer start index
                    final_answers = []
                    for ans in text_answers:
                        if ans["type"] == "yesno":
                            final_answers.append(
                                {'text': ans["text"], "answer_start": -1})
                        elif paragraph["text"][ans["answer_start"]:ans["answer_start"]+len(ans["text"])] == ans["text"]:
                            final_answers.append(
                                {'text': ans["text"], "answer_start": ans["answer_start"]})
                        elif paragraph["text"][ans["answer_start"]:ans["answer_start"]+len(ans["text"])] != ans["text"] and paragraph["text"].find(ans["text"]) > -1:
                            final_answers.append(
                                {'text': ans["text"], "answer_start": paragraph["text"].find(ans["text"])})
                        else:
                            continue
                    if len(final_answers) > 0:
                        squad_example = {'context': paragraph["text"],
                                            'qas': [{'question': updated_question, 'is_impossible': True if final_answers[0]["text"] in ["yes", "no"] else False,
                                                    'answers': final_answers,
                                                    'id': q_id}]}

                        squad_style_data["data"].append(
                            {"title": paragraph["title"], 'paragraphs': [squad_example]})
                        count_gold += 1
                    elif len(final_answers) == 0 and split == "dev":
                        squad_example = {'context': paragraph["text"],
                                            'qas': [{'question': updated_question, 'is_impossible': True,
                                                    'answers': [{"text": answer["text"], "answer_start": answer["answer_start"]} for answer in text_answers],
                                                    'id': q_id}]}
                        squad_style_data["data"].append(
                            {"title": paragraph["title"], 'paragraphs': [squad_example]})
                        count_distractor += 1
                    else:
                        continue
        else:
            print(q_type)
            missing_qtype.setdefault(q_type, 0)
            missing_qtype[q_type] += 1
    print("# of original questions: {}".format(q_count))
    print("{} data converted into squad format".format(
        len(squad_style_data["data"])))
    print("{0} gold, {1} distractor".format(count_gold, count_distractor))
    print("hotpot qa count" + str(hp_bridge))
    print(q_types)
    print(squad_style_data["data"][-1])
    print("missing q")
    print(missing_qtype)
    return squad_style_data

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_fp",
                        default=None, type=str, required=True)
    parser.add_argument("--dev_data_fp",
                        default=None, type=str, required=True)
    parser.add_argument("--doc_data_fp",
                        default=None, type=str, required=True)
    parser.add_argument("--output_data_dir",
                        default=None, type=str, required=True)
    parser.add_argument('--mode',
                        default="implicit_decomp", type=str)
    parser.add_argument('--model_type',
                        default="bert", type=str)
    args = parser.parse_args()
    # save the data into the target directory.
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    train_data = read_jsonlines(args.train_data_fp)
    eval_data = read_jsonlines(args.dev_data_fp)
    doc_data = read_jsonlines(args.doc_data_fp)

    converted_train_data = convert_qa_data_into_squad_format_implicit_decomp_final(
        train_data, doc_data, split="train", mode=args.mode, model_type=args.model_type)
    converted_eval_data = convert_qa_data_into_squad_format_implicit_decomp_final(
        eval_data, doc_data, split="dev", mode=args.mode, model_type=args.model_type)

    with open(os.path.join(args.output_data_dir, "MultiModalQA_train_converted_to_squad.mode={0}_model={1}.json".format(args.mode, args.model_type)), 'w') as outfile:
        json.dump(converted_train_data, outfile)

    with open(os.path.join(args.output_data_dir, "MultiModalQA_dev_converted_to_squad.mode={0}_model={1}.json".format(args.mode, args.model_type)), 'w') as outfile:
        json.dump(converted_eval_data, outfile)


if __name__ == "__main__":
    main()
