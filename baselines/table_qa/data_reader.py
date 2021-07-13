import os
import random
import torch
import logging
import json
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from torch.utils.data import TensorDataset
from transformers.data.processors.squad import SquadProcessor, is_torch_available
from common_utils import TABLE_SINGLE_HOP_QUESTION_TYPES, TABLE_AS_FIRST_HOP_QUESTION_TYPES, \
    TABLE_AS_SECOND_HOP_QUESTION_TYPES, process_question_for_implicit_decomp


logger = logging.getLogger(__name__)
ans_type2id_map = {"extractive": 0, "yes": 1, "no": 2, "mean": 3, "sum": 4, "count": 5, "image": 6}


def load_and_cache_table_examples(args, tokenizer, evaluate=False, output_examples=False):

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."

    logger.info("Creating features from dataset file at %s", input_dir)

    processor = TableQaProcessor(
        tokenizer=tokenizer,
        question_format=args.question_format,
        include_first_hop_in_training=args.include_first_hop_in_training
    )
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
    else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

    logger.info("Loaded {} examples for {}.".format(len(examples), "evaluation" if evaluate else "training"))
    if not evaluate:
        valid_examples = [e for e in examples if not e.is_impossible]
        logger.info(
            "{} out of them are valid for training the table model.".format(len(valid_examples)))
        logger.info("We only keep the valid questions for training.")
        examples = valid_examples

    features, dataset = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        threads=args.threads,
    )

    if output_examples:
        return dataset, examples, features
    return dataset


class TableQaExample(object):
    def __init__(
            self,
            qas_id,
            question_text,
            table_data,
            answer_type=None,
            answer_modality=None,
            answer_cells=[],
            answers=[],
            is_impossible=False,
            question_type=None
    ):
        self.qas_id = qas_id
        self.question_text = question_text if question_text.strip() else "NULL"
        self.raw_table_data = table_data
        self.answer_type = answer_type
        self.answer_modality = answer_modality
        self.answer_cells = answer_cells
        self.is_impossible = is_impossible
        self.answers = answers
        self.qa_type = question_type

        self.table_headers = [it["column_name"] for it in table_data["header"]]

        table_cells = []
        for row_data in table_data["table_rows"]:
            row_cells = [cell["text"] for cell in row_data]
            table_cells.append(row_cells)

        assert all(len(row_cells) == len(self.table_headers) for row_cells in table_cells)
        self.table_cells = table_cells


class TableQaProcessor(SquadProcessor):

    def __init__(self,
                 tokenizer,
                 train_file="MultiModalQA_train.jsonl",
                 dev_file="MultiModalQA_dev.jsonl",
                 question_format="plain",
                 include_first_hop_in_training=False):
        self.tokenizer = tokenizer
        self.train_file = train_file
        self.dev_file = dev_file
        self.question_format = question_format
        self.include_first_hop_in_training=include_first_hop_in_training

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.
        """
        if data_dir is None:
            data_dir = ""

        examples = []
        with open(
                os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as fin:
            for line in fin:
                new_examples = self.create_examples(json.loads(line), "train")
                examples += new_examples
                # if len(examples) > 1000:
                #     break
        return examples

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.
        """
        if data_dir is None:
            data_dir = ""

        examples = []
        with open(
                os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as fin:
            for line in fin:
                new_examples = self.create_examples(json.loads(line), "dev")
                examples += new_examples
                # if len(examples) > 100:
                #     break
        return examples

    def create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        tables = [doc for doc in input_data["context"]["documents"] if "table" in doc]
        assert len(tables) == 1, "Only support one table context for now."
        table_data = tables[0]["table"]
        for qa in input_data["qas"]:
            qtype = qa["metadata"]["type"]
            assert len(qa["answers"]) > 0, "No annotation for question {}".format(qa["qid"])

            # Note: we only use table questions for both training, and validating here.
            if qtype in TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_SECOND_HOP_QUESTION_TYPES:
                answer_type, answer_modality, answer_text, answer_list, answer_cells = extract_answer(qa)

                if self.question_format == "implicit_decomp":
                    if qtype in TABLE_AS_SECOND_HOP_QUESTION_TYPES:
                        assert "arg1_answers" in qa["metadata"] and "arg2_answers" in qa["metadata"]
                        assert "arg1_modality" in qa["metadata"] and "arg2_modality" in qa["metadata"]
                        # use the first hop answer that is not from table as the bridge answer
                        if qa["metadata"]["arg1_modality"][0] == "table":
                            bridge_answer = "; ".join([it["answer"] for it in qa["metadata"]["arg2_answers"]])
                        elif qa["metadata"]["arg2_modality"][0] == "table":
                            bridge_answer = "; ".join([it["answer"] for it in qa["metadata"]["arg1_answers"]])
                        else:
                            raise ValueError("No final answer is in table modality.")
                        if is_training and random.random() < 0.3:
                            bridge_answer = ""
                        question_text = process_question_for_implicit_decomp(
                            qa["question"], qtype, hop=1, bridge_entity=bridge_answer, sep_token=self.tokenizer.sep_token
                        )
                    else:
                        question_text = process_question_for_implicit_decomp(
                            qa["question"], qtype, hop=0, bridge_entity="", sep_token=self.tokenizer.sep_token
                        )
                elif self.question_format == "empty":
                    question_text = ""
                elif self.question_format == "plain":
                    question_text = qa["question"]
                else:
                    raise ValueError(f"Unsupported question format: {self.question_format}")
                example = TableQaExample(
                    qas_id=qa["qid"],
                    question_text=question_text,
                    table_data=table_data,
                    answer_modality=answer_modality,
                    answer_type=answer_type,
                    answer_cells=answer_cells,
                    answers=answer_list,
                    is_impossible=False,  # Since we only load table questions here, we regard all examples as possible.
                    question_type=qtype
                )
                examples.append(example)

            # During training, we also include first hop answer if specified
            if is_training and self.include_first_hop_in_training and qtype in TABLE_AS_FIRST_HOP_QUESTION_TYPES:
                answer_type, answer_modality, answer_text, answer_list, answer_cells = extract_first_hop_table_answer(
                    qa["metadata"]
                )

                # because we only consider specific question types that should have table as the first hop answers
                assert answer_modality == "table"

                if self.question_format == "implicit_decomp":
                    question_text = process_question_for_implicit_decomp(
                        qa["question"], qtype, hop=0, bridge_entity="", sep_token=self.tokenizer.sep_token
                    )
                elif self.question_format == "empty":
                    question_text = ""
                elif self.question_format == "plain":
                    question_text = qa["question"]
                else:
                    raise ValueError(f"Unsupported question format: {self.question_format}")

                example = TableQaExample(
                    qas_id=qa["qid"] + "_first_hop",
                    question_text=question_text,
                    table_data=table_data,
                    answer_type=answer_type,
                    answer_modality=answer_modality,
                    answer_cells=answer_cells,
                    answers=answer_list,
                    is_impossible=False,
                    question_type=qtype + "-firsthop"
                )
                examples.append(example)
        return examples


def extract_answer(qa):
    answer_type, answer_modality, answer_text, annotated_answers, answer_cells = None, None, None, [], []
    if qa["answer_properties"]["sequence_format"] == "single_answer":
        annotated_answers.append([str(ans["answer"]) for ans in qa["answers"]])
        answer_info = qa["answers"][0]
        answer_text = str(answer_info["answer"])
        if answer_info["is_extractive"]:
            answer_type = "extractive"
        elif answer_info["type"] == "yesno":
            if answer_info["answer"].lower() == "yes":
                answer_type = "yes"
            elif answer_info["answer"].lower() == "no":
                answer_type = "no"
            else:
                raise ValueError("Incorrect yesno answer: {}".format(answer_info["answer"]))
        elif answer_info["modality"] == "image":
            answer_type = "image"
        else:
            if not (answer_info["type"] == "number"
                    and qa["metadata"]["arithmetic_opp"] in ["mean", "count", "sum"]):
                logger.error("Unexpected!")
                logger.error(qa["qid"])
                logger.error(answer_info["type"])
                logger.error(qa["metadata"]["arithmetic_opp"])
            answer_type = qa["metadata"]["arithmetic_opp"]
        answer_modality = answer_info["modality"]
        if "table_indices" in answer_info:
            answer_cells = answer_info["table_indices"]
    elif qa["answer_properties"]["sequence_format"] == "list":
        answer_type = "extractive"
        annotated_answers.append([str(ans["answer"]) for ans in qa["answers"]])
        answer_text = "; ".join([ans["answer"] for ans in qa["answers"]])
        answer_modalities = set([ans["modality"] for ans in qa["answers"]])
        assert len(answer_modalities) == 1, f"Question {qa['qid']} has multiple answer modalities: {answer_modalities}."
        answer_modality = answer_modalities.pop()
        for ans in qa["answers"]:
            if "table_indices" in ans:
                answer_cells += ans["table_indices"]
    else:
        raise ValueError("Question {} has undefined sequence format: {}".format(
            qa["qid"],
            qa["answer_properties"]["sequence_format"])
        )
    return answer_type, answer_modality, answer_text, annotated_answers, answer_cells


def extract_first_hop_table_answer(metadata):
    answer_type, answer_modality, answer_text, annotated_answers, answer_cells = None, None, None, [], []
    assert "arg1_answers" in metadata and "arg2_answers" in metadata
    assert "arg1_modality" in metadata and "arg2_modality" in metadata
    if metadata["arg1_meta"]["modalities"][0] == "table":
        first_hop_answer = metadata["arg1_answers"]
    elif metadata["arg2_meta"]["modalities"][0] == "table":
        first_hop_answer = metadata["arg2_answers"]
    else:
        raise ValueError("No first-hop answer is in table modality.")
    answer_text = "; ".join([ans["answer"] for ans in first_hop_answer])
    annotated_answers.append([ans["answer"] for ans in first_hop_answer])
    answer_modalities = set([ans["modality"] for ans in first_hop_answer])
    assert len(answer_modalities) == 1, f"Multiple answer modalities for the first hop: {answer_modalities}."
    answer_modality = answer_modalities.pop()
    for ans in first_hop_answer:
        if "table_indices" in ans:
            answer_cells += ans["table_indices"]
    answer_type = "extractive"
    return answer_type, answer_modality, answer_text, annotated_answers, answer_cells


def convert_examples_to_features(
        examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, threads=1, verbose=True
):
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_tableqa_example_to_features,
            max_seq_length=max_seq_length,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert table qa examples to features",
                disable=not verbose
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id",
                                 disable=not verbose):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if not is_torch_available():
        raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_cell_token_masks = torch.tensor([f.cell_token_masks for f in features], dtype=torch.float)

    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
    if not is_training:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_cell_token_masks,
            all_example_index,
            all_cls_index,
            all_p_mask,
        )
    else:
        all_cell_labels = torch.tensor([f.cell_labels for f in features], dtype=torch.long)
        all_answer_types = torch.tensor([f.answer_type_id for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_cell_token_masks,
            all_cell_labels,
            all_answer_types,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
        )
    return features, dataset


def linearize_table_row_with_tokenization(
        table_cells, table_headers, row_idx, tokenizer, answer_cells=[], COL_DELIM=" ; ", ROW_DELIM=" . "):
    row_cells = table_cells[row_idx]
    row_tokens, cell_spans, cell_labels, cell_to_orig_pos_map = [], [], [], []
    delim_tokens = tokenizer.tokenize(COL_DELIM)
    row_indicator_str = f"Row {row_idx} is: "
    row_tokens += tokenizer.tokenize(row_indicator_str)
    for col_idx, (header, cell) in enumerate(zip(table_headers, row_cells)):
        cell = cell.strip()
        if not cell:
            continue
        cell_indicator_str = f"{header} is "
        row_tokens += tokenizer.tokenize(cell_indicator_str)
        cell_tokens = tokenizer.tokenize(cell)
        cell_spans.append((len(row_tokens), len(row_tokens) + len(cell_tokens)))
        cell_to_orig_pos_map.append((row_idx, col_idx))
        row_tokens += cell_tokens
        row_tokens += delim_tokens
        if [row_idx, col_idx] in answer_cells:
            cell_labels.append(1)
        else:
            cell_labels.append(0)
    row_tokens = row_tokens[: -len(delim_tokens)] + tokenizer.tokenize(ROW_DELIM)
    # in very rare cases, all the cells in the row are empty and we just skip such rows.
    if not cell_spans:
        row_tokens = []
    return row_tokens, cell_spans, cell_labels, cell_to_orig_pos_map


def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_tableqa_example_to_features(example, max_seq_length, max_query_length, is_training):
    table_headers = example.table_headers
    table_cells = example.table_cells

    encoded_examples = []
    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    num_of_special_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    table_tokens, cell_spans, cell_labels, cell_to_orig_pos_map = [], [], [], []
    row_idx = 0

    # Note: currently we don't support stride when splitting the table into several chunks.
    # Each row will be contained in only one example.
    while row_idx < len(table_cells):
        row_tokens, row_cell_spans, row_cell_labels, row_cell_to_orig_pos_map \
            = linearize_table_row_with_tokenization(table_cells, table_headers, row_idx, tokenizer,
                                                    example.answer_cells)
        if max_seq_length - num_of_special_tokens - len(truncated_query) - len(table_tokens) >= len(row_tokens) \
                or len(table_tokens) == 0:
            # add the new row if the sequence doesn't exceeded the max length or there are no tokens in the cache
            cell_spans += [(span[0] + len(table_tokens), span[1] + len(table_tokens)) for span in row_cell_spans]
            cell_to_orig_pos_map += row_cell_to_orig_pos_map
            table_tokens += row_tokens
            cell_labels += row_cell_labels
            row_idx += 1

        if (max_seq_length - num_of_special_tokens - len(truncated_query) - len(table_tokens) < len(row_tokens)
            or row_idx == len(table_cells)) and len(table_tokens) > 0:

            assert tokenizer.padding_side == "right"  # only support right padding now
            encoded_dict = tokenizer.encode_plus(
                truncated_query,
                table_tokens,
                max_length=max_seq_length,
                pad_to_max_length=True,
                truncation_strategy="only_second",
                return_token_type_ids=True,
                return_special_tokens_mask=True,
                return_overflowing_tokens=True,
            )

            # if one row is too long, we will have overflowing tokens.
            # And we just ignore these tokens and the corresponding cells.
            if "overflowing_tokens" in encoded_dict:
                table_tokens = table_tokens[:len(table_tokens) - len(encoded_dict["overflowing_tokens"])]
                cell_spans = [span for span in cell_spans if span[1] <= len(table_tokens)]
                cell_to_orig_pos_map = cell_to_orig_pos_map[: len(cell_spans)]
                cell_labels = cell_labels[: len(cell_spans)]

            # hack for getting the start position of the table tokens
            table_offset = None
            for i in range(len(truncated_query), len(encoded_dict["special_tokens_mask"])):
                if encoded_dict["special_tokens_mask"][i] == 0 and encoded_dict["special_tokens_mask"][i - 1] == 1:
                    table_offset = i
            cell_spans = [(span[0] + table_offset, span[1] + table_offset) for span in cell_spans]

            paragraph_len = min(
                len(table_tokens),
                max_seq_length - len(truncated_query) - num_of_special_tokens
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                non_padded_ids = encoded_dict["input_ids"][
                                 : encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            encoded_dict["tokens"] = tokens
            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["length"] = paragraph_len
            encoded_dict["cell_spans"] = cell_spans
            encoded_dict["cell_labels"] = cell_labels
            encoded_dict["cell_to_orig_pos_map"] = cell_to_orig_pos_map

            encoded_examples.append(encoded_dict)
            table_tokens, cell_spans, cell_labels, cell_to_orig_pos_map = [], [], [], []

    features = []
    for encoded_dict in encoded_examples:
        # Identify the position of the CLS token
        cls_index = encoded_dict["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(encoded_dict["token_type_ids"])
        p_mask = np.minimum(p_mask, 1)
        p_mask = 1 - p_mask
        p_mask[np.where(np.array(encoded_dict["input_ids"]) == tokenizer.sep_token_id)[0]] = 1
        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        # cell_token_masks: for each cell, mask with 1 for token in that cell.
        # Since the number of cells vary in different examples, we create redundant masks for ``num_of_tokens'' cells.
        num_of_tokens = len(encoded_dict["input_ids"])
        cell_token_masks = np.zeros((num_of_tokens, num_of_tokens))
        cell_labels = np.zeros(num_of_tokens)
        for idx, (cell_span, cell_label) in enumerate(zip(encoded_dict["cell_spans"], encoded_dict["cell_labels"])):
            cell_token_masks[idx][cell_span[0]: cell_span[1]] = 1
            cell_labels[idx] = cell_label

        features.append(
            TableQaFeatures(
                encoded_dict["input_ids"],
                encoded_dict["attention_mask"],
                encoded_dict["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=encoded_dict["paragraph_len"],
                tokens=encoded_dict["tokens"],
                cell_spans=encoded_dict["cell_spans"],
                cell_to_orig_pos_map=encoded_dict["cell_to_orig_pos_map"],
                cell_token_masks=cell_token_masks,
                cell_labels=cell_labels,
                answer_type_id=ans_type2id_map[example.answer_type] if is_training or example.answer_type else None,
                is_impossible=sum(cell_labels) == 0
            )
        )
    return features


class TableQaFeatures(object):
    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            cell_token_masks,
            cell_spans,
            cell_to_orig_pos_map,
            example_index,
            unique_id,
            paragraph_len,
            tokens,
            cell_labels=None,
            answer_type_id=None,
            is_impossible=False
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.cell_token_masks = cell_token_masks
        self.cell_spans = cell_spans
        self.cell_to_orig_pos_map = cell_to_orig_pos_map

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.tokens = tokens

        self.answer_type_id = answer_type_id
        self.cell_labels = cell_labels
        self.is_impossible = is_impossible
