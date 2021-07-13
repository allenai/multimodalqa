import os
import json
import torch
import pickle
import numpy as np
import argparse
from tqdm import tqdm

from transformers import AutoModelForQuestionAnswering, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from utils import InputExample
from preprocess import glue_convert_examples_to_features

import sys
sys.path.append('..')
from common_utils import ALL_QUESTION_TYPES

class QuestionClassificationInferenceModel:
    def __init__(self, model_dir, device):
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir)
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.labels = ALL_QUESTION_TYPES

        if hasattr(self.tokenizer, "do_lower_case"):
            self.do_lower_case = self.tokenizer.do_lower_case
        elif hasattr(self.tokenizer, "basic_tokenizer"):
            self.do_lower_case = self.tokenizer.basic_tokenizer.do_lower_case
        else:
            self.do_lower_case = True

    def predict(self, guid, question):
        example = InputExample(guid=guid, text_a=question,
                               label=self.labels[0])  # Hack: psudo label only for data preparation.
        features = glue_convert_examples_to_features(
            [example],
            tokenizer=self.tokenizer,
            label_list=self.labels,
            output_mode="classification",
            # pad on the left for xlnet
            pad_on_left=bool(self.config.model_type in ["xlnet"]),
            pad_token=self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.pad_token])[0],
            pad_token_segment_id=4 if self.config.model_type in [
                "xlnet"] else 0,
        )
        input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long)
        token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long)

        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.config.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    token_type_ids if self.config.model_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            for k, v in inputs.items():
                if v is not None:
                    inputs[k] = v.to(self.device)
            outputs = self.model(**inputs)
            logits = outputs[0]
            pred_label_id = torch.argmax(logits[0]).detach().cpu().item()
            pred_label = self.labels[pred_label_id]
        return pred_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="path to model")
    parser.add_argument("--classification_model_path", default=None, type=str, required=True,
                        help="path to model")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    input_examples = json.load(open(args.input_file))
    final_results = {}
    question_classifier = QuestionClassificationInferenceModel(
        args.classification_model_path, device)

    # predict
    for example_idx, example in tqdm(input_examples.items()):
        final_results[example_idx] = question_classifier.predict(
            example_idx, example["question"])

    print(final_results)
    with open('classifier_predicted_results.json', 'w') as outfile:
        json.dump(final_results, outfile)


if __name__ == "__main__":
    main()
