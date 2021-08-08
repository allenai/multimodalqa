import pickle
import torch

import os

from pytorch_transformers import BertTokenizer

from model import VilbertForMQA
from image_features import FeatureExtractor
from dataset import process_raw_inputs, collate, read_jsonl
from dataset import QUESTION_TYPES_WITH_SINGLE_CORRECT_ANS, QUESTIONS_WITH_VOCAB_ANSWERS
from dataset import DISTRACTOR_TOK, IS_ANSWER_TOK, IS_NOT_ANSWER_TOK
from train_and_eval import get_config

from baselines.common_utils import IMAGE_AS_SECOND_HOP_QUESTION_TYPES
import numpy as np

basedir = os.path.dirname(os.path.abspath(__file__))

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_closest(s1, s2_list):
    return min(s2_list, key=lambda s2: levenshtein_distance(s1, s2))

def get_answers(
        model, raw_data, question_type, device, tokenizer,
        answer2idx, idx2answer, bridge_entities, mode, use_distractors):
    if bridge_entities is not None:
        clean_bridge_entities = set()
        for bridge_entity in bridge_entities:
            clean_bridge_entities.add(get_closest(bridge_entity, raw_data['img_names']))
        bridge_entities = list(clean_bridge_entities)

    has_answers = 'answers' in raw_data
    bad_indexes = []
    if not use_distractors:
        # Filter data so it contains only the actual image.
        if has_answers:
            bad_indexes = [i for i, answer in enumerate(raw_data['answers']) if answer == DISTRACTOR_TOK]
            if question_type in QUESTIONS_WITH_VOCAB_ANSWERS:
                assert len(bad_indexes) == len(raw_data['img_names']) - 1
        elif question_type in QUESTIONS_WITH_VOCAB_ANSWERS:
            bad_indexes = [i for i, img_id in enumerate(raw_data['img_names']) if img_id not in raw_data['question']]
            if len(bad_indexes) != len(raw_data['img_names']) - 1:
                # something went wrong, fall back to using distractors
                bad_indexes = []
    elif question_type in IMAGE_AS_SECOND_HOP_QUESTION_TYPES and mode not in ['auto_routing', 'context_only']:
        assert bridge_entities is not None
        assert isinstance(bridge_entities, list)
        bad_indexes = [idx for idx, img_name in enumerate(raw_data['img_names']) if img_name not in bridge_entities]

    num_samples = len(raw_data['img_names'])
    if has_answers: assert len(raw_data['answers']) == num_samples
    if 'img_features_files' in raw_data:
        assert len(raw_data['img_features_files']) == num_samples
    elif 'img_features' in raw_data:
        assert len(raw_data['img_features']) == num_samples
    else:
        raise ValueError('Expected either `img_features_files` or `img_features` to be present.')

    raw_inputs = []
    for idx in range(num_samples):
        if idx in bad_indexes:
            continue
        question_type = raw_data['question_type']
        hop = 1 if question_type in IMAGE_AS_SECOND_HOP_QUESTION_TYPES else 0
        raw_inputs.append({
            'question_id': raw_data['question_id'],
            'question_type': question_type,
            'question': raw_data['question'],
            'img_id': raw_data['img_ids'][idx],
            'img_name': raw_data['img_names'][idx],
            'hop': hop,
        })
        if has_answers:
            raw_inputs[-1]['answer'] = raw_data['answers'][idx]
        if 'img_features_files' in raw_data:
            raw_inputs[-1]['img_features_file'] = raw_data['img_features_files'][idx]
        elif 'img_features' in raw_data:
            raw_inputs[-1]['img_features'] = raw_data['img_features'][idx]
        
    inputs = [process_raw_inputs(raw_input, tokenizer, answer2idx, mode) for raw_input in raw_inputs]
    raw_questions = [d['raw_question'] for d in inputs]

    if has_answers:
        if question_type in QUESTIONS_WITH_VOCAB_ANSWERS:
            ground_truth_answers = [
                answer for answer in raw_data['answers'] if
                answer not in [IS_NOT_ANSWER_TOK, DISTRACTOR_TOK]]
            assert len(ground_truth_answers) == 1
        elif question_type in ['Intersect(ImageListQ,TextQ)']:
            ground_truth_answers = [
                img_id for (img_id, answer) in zip(raw_data['img_names'], raw_data['answers'])
                if answer == IS_ANSWER_TOK and img_id in raw_data['golden_bridge_entities']]
        else:
            ground_truth_answers = [
                img_id for (img_id, answer) in zip(raw_data['img_names'], raw_data['answers'])
                if answer == IS_ANSWER_TOK]
            if question_type in QUESTION_TYPES_WITH_SINGLE_CORRECT_ANS:
                assert len(ground_truth_answers) == 1

    if len(inputs) == 0:
        predicted_answers = ['empty inputs']
    else:
        batched_data = collate(inputs, tokenizer.pad_token_id)

        feats = batched_data['img_features'].to(device)
        boxes = batched_data['img_bboxes'].to(device)
        questions = batched_data['questions'].to(device)
        mask = batched_data['masks'].to(device)
        model.to(device)

        logits = model(feats, boxes, questions, mask).detach().cpu().numpy()

        if question_type in QUESTIONS_WITH_VOCAB_ANSWERS:
            # If the question has distractors, we compute predictions for all
            # distractors and ground truth. We first choose the answer the model is most confident
            # is *not* DISTRACTOR_TOK, then choose the other answer it assigns the highest probability to.
            irrelevant_img_probs = logits[:, answer2idx[DISTRACTOR_TOK]]
            best_guess_idx = np.argmin(irrelevant_img_probs)
            best_guess_logits = logits[best_guess_idx]
            best_guess_logits[answer2idx[DISTRACTOR_TOK]] -= 1.e9
            best_guess_logits[answer2idx[IS_ANSWER_TOK]] -= 1.e9
            best_guess_logits[answer2idx[IS_NOT_ANSWER_TOK]] -= 1.e9
            best_guess = np.argmax(best_guess_logits)
            predicted_answers = [idx2answer[best_guess]]
        elif question_type in QUESTION_TYPES_WITH_SINGLE_CORRECT_ANS:
            # These images have a single correct answer. So we return the one the model is most confident
            # on the answer IS_ANSWER_TOK.
            relevant_img_probs = logits[:, answer2idx[IS_ANSWER_TOK]]
            best_guess_idx = np.argmax(relevant_img_probs)
            predicted_answers = [inputs[best_guess_idx]['img_name']]
        else:
            # For all others, answers are entities, and we select the ones where the model assigns
            # a higher score to IS_ANSWER_TOK than IS_NOT_ANSWER_TOK.
            yes_idx = answer2idx[IS_ANSWER_TOK]
            no_idx = answer2idx[IS_NOT_ANSWER_TOK]
            positive_occurrances = (logits[:, yes_idx] > logits[:, no_idx])
            predicted_answers = []
            for i in range(len(logits)):
                if positive_occurrances[i]:
                    predicted_answers.append(inputs[i]['img_name'])

    if has_answers:
        return predicted_answers, ground_truth_answers, raw_questions
    return predicted_answers, raw_questions


class InferenceModel:
    def __init__(self, checkpoint_path, vilbert_dir, img_info_path,
                 vocab_path, device, mode, img_features_dir=None, use_distractors=True):
        self.device = device
        assert mode in {'auto_routing', 'implicit_decomp', 'context_only'}
        self.use_distractors = use_distractors
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

        self.img_docs = read_jsonl(img_info_path)
        self.img_docs = {doc['id']: doc for doc in self.img_docs}

        self.idx2answer, self.answer2idx = pickle.load(open(vocab_path, 'rb'))
        self.num_labels = len(self.idx2answer)
        self.model = VilbertForMQA(
            os.path.join(vilbert_dir, "multi_task_model.bin"),
            get_config(vilbert_dir),
            self.num_labels,
            mask_vis=False,
            mask_lang=False,
            dropout_prob=0)

        if checkpoint_path is not None:
            # print("Loading model weights from checkpoint: %s" % checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path, device))

        self.model.eval()
        self.model.to(device)

        if img_features_dir is None:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = None
            self.img_features_dir = img_features_dir

    def predict(self, question, img_ids_or_paths, question_type, bridge_entities=None):
        if question_type in IMAGE_AS_SECOND_HOP_QUESTION_TYPES and self.mode not in ['auto_routing', 'context_only']:
            assert bridge_entities is not None
            assert isinstance(bridge_entities, list)
            assert self.feature_extractor is None, 'Prediction from image paths not supported yet for multihop'
        raw_inputs = {
            'question_id': '',  # not relevant.
            'question_type': question_type,
            'question': question,
            'img_ids': [],
            'img_names': [],
            'img_features': [],
        }
        for img_id_or_path in img_ids_or_paths:
            if self.feature_extractor is not None:
                img_path = img_id_or_path
                assert isinstance(img_path, str)
                img_features = self.feature_extractor.extract_features(img_path)
                img_id = os.path.splitext(os.path.basename(img_path))[0]
                img_name = img_id
            else:
                img_id = img_id_or_path
                img_features_filename = os.path.join(self.img_features_dir, img_id + '.pickle')
                if not os.path.exists(img_features_filename):
                    print(f"Skip image: {img_id} since no precomputed feature is found at {img_features_filename}.")
                    continue
                img_features = pickle.load(open(img_features_filename, 'rb'))
                img_name = self.img_docs[img_id]['title']

            raw_inputs['img_ids'].append(img_id)
            raw_inputs['img_names'].append(img_name)
            raw_inputs['img_features'].append(img_features)
        return get_answers(
            self.model, raw_inputs, question_type,
            self.device, self.tokenizer, self.answer2idx, self.idx2answer,
            bridge_entities, self.mode, use_distractors=self.use_distractors)[0]


if __name__ == '__main__':
    mode = 'implicit_decomp'
    checkpoint_filename = os.path.join(basedir, 'checkpoints', 'my_model.pt')
    vocab_filename = os.path.join(basedir, '../../dataset/cache/vocab.pickle')
    vilbert_dir = os.path.join(basedir, "../../deps/vilbert-multi-task")
    img_info_path = os.path.join(basedir, '../../dataset/MMQA_images.jsonl')
    img_features_dir = os.path.join(basedir, '../../dataset/img_features')
    device = 'cuda:1'
    question = "What shape is in the center of Viduthalal Chiruthaigal Katchi?"
    question_type = "ImageQ"
    img_ids_or_paths = ['e7a17bf9eb43fbe95843f682a03b0eb3']

    inference_model = InferenceModel(
        checkpoint_filename, vilbert_dir, img_info_path, vocab_filename, device, mode, img_features_dir)
    print('Running inference on question:', question)
    pred = inference_model.predict(question, img_ids_or_paths, question_type)
    print('='*50)
    print('Model prediction: %s' % pred)
