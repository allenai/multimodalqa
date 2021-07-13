import sys
import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from image_features import FeatureExtractor

from tqdm import tqdm
import urllib

import pickle

import subprocess

import traceback

from zipfile import ZipFile

dir_name = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.join(dir_name, '../../')
sys.path.append(parent_dir)

from baselines.common_utils import *


DISTRACTOR_TOK = '<DISTRACTOR>'
IS_ANSWER_TOK = '<IS_ANSWER>'
IS_NOT_ANSWER_TOK = '<IS_NOT_ANSWER>'

QUESTION_TYPES_WITH_SINGLE_CORRECT_ANS = [
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compare(Compose(TableQ,ImageQ),TableQ)'
]
QUESTIONS_WITH_VOCAB_ANSWERS =  [
    'Compose(ImageQ,TextQ)',
    'ImageQ',
    'Compose(ImageQ,TableQ)'
]
AUTO_ROUTING_QTYPES = IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_SECOND_HOP_QUESTION_TYPES

basedir = os.path.dirname(os.path.abspath(__file__))


def zip_images(data_dir, splits):
    print('Zipping data for splits %s.' % splits)
    name = '_'.join(splits)
    img_local_paths = download_images(data_dir, splits)
    img_ft_filenames = get_img_features(data_dir, img_local_paths)    
    info_file = os.path.join(data_dir, name + '_img_info.pickle')
    pickle.dump(img_ft_filenames, open(info_file, 'wb'))
    print('Total files being zipped: %d files' % len(img_ft_filenames))
    with ZipFile(os.path.join(data_dir, name + '_img_features.zip'), 'w') as z:
        z.write(info_file)
        for filename in img_ft_filenames.values():
            z.write(filename)


def download_images(data_dir, splits):
    img_base_url = 'https://multimodalqa-images.s3-us-west-2.amazonaws.com/'
    os.makedirs(os.path.join(data_dir, 'img'), exist_ok=True)
    img_urls = {}
    for split in splits:
        print('Processing split %s' % split)
        filename = os.path.join(data_dir, 'MultiModalQA_%s.jsonl' % split)
        with open(filename, 'r') as f:
            data = [json.loads(l.strip()) for l in f.readlines()]
        for d in data:
            context = d['context']
            qas = d['qas']
            _, local_img_urls = process_context_and_qas(context, qas)
            for img_id, img_url in local_img_urls.items():
                img_urls[img_id] = img_url
    
    print('Downloading %d images' % len(img_urls))
    img_paths = {}
    img_dir = os.path.join(data_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
    svg_failures = {}
    for img_id, img_url in tqdm(img_urls.items()):
        img_path = os.path.join(img_dir, img_url[4:].replace('/', '|'))
        if not os.path.exists(img_path):
            urllib.request.urlretrieve(
                img_base_url + urllib.parse.quote(img_url), img_path)
        if img_path.endswith('svg'):
            try:
                svg_img_path = img_path
                png_img_path = img_path[:-3] + 'png'
                img_paths[img_id] = png_img_path
                if os.path.exists(png_img_path):
                    continue
                cmd_list = [
                    'rsvg-convert', '--format=png', 
                    '--output=%s' % png_img_path, svg_img_path]
                p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
                out, err = p.communicate()
                
                if p.returncode:
                    raise Exception('Error: %s. Out: %s' % ((err or '?'), (out or '?')))
            except Exception as e:
                svg_failures[img_id] = e
                print('Warning: skipping svg image "%s" due to error: %s' % (img_id, e))
                del img_paths[img_id]
        else:
            img_paths[img_id] = img_path
    
    if svg_failures:
        print('Had %d errors when converting svg images.' % len(svg_failures))
        print('Unique failures:')
        for failure in set([str(e) for e in svg_failures.values()]):
            print(failure)
    return img_paths


def get_img_features(data_dir, img_paths):
    print('Getting image features.')
    img_features_dir = os.path.join(data_dir, 'img_features')
    os.makedirs(img_features_dir, exist_ok=True)
    feature_extractor = None
    img_feature_filenames = {}
    for img_id, img_path in tqdm(img_paths.items()):
        if img_id in img_feature_filenames:
            continue
        filename =  os.path.splitext(os.path.basename(img_path))[0] + '.pickle'
        full_filename = os.path.join(img_features_dir, filename)
        img_feature_filenames[img_id] = full_filename
        if os.path.exists(full_filename):
            continue
        if feature_extractor is None:
            feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(img_path)
        pickle.dump(features, open(full_filename, 'wb'))
    print('Done extracting image features.')
    return img_feature_filenames


def process_raw_question(question, question_type, mode, img_id, hop=0, sep_token='[SEP]'):
    if mode in ['implicit_decomp', 'auto_routing']:
        processed_question = (
            f'{question_type} {sep_token} '
            f'HOP={hop} {sep_token} '
            f'{question} {sep_token} '
            f'{img_id}'
        )
    elif mode == 'context_only':
        processed_question = f'{img_id}'
    else:
        raise ValueError('Unsupported mode: %s' % mode)
    return processed_question


def process_context_and_qas(context, qas):
    docs = {doc['id']: doc for doc in context['documents']}
    
    info = []
    img_urls = {}
    for qa in qas:
        metadata = qa['metadata']
        question_type = metadata['type']
        if question_type == 'ImageQ':
            img_ids = [metadata['wiki_entities_in_question'][0]['wiki_title']]
            answers = [qa['answers'][0]['answer']]
            img_urls[img_ids[0]] = metadata['image_url']
        elif question_type == 'ImageListQ':
            entities = metadata['wiki_entities_in_question_names']
            correct_answers = {am['answer'] for am in qa['answers']}
            assert len(metadata['image_urls']) == len(entities)
            img_ids, answers = [], []
            for img_url, img_id in zip(metadata['image_urls'], entities):
                answer = IS_ANSWER_TOK if img_id in correct_answers else IS_NOT_ANSWER_TOK
                img_ids.append(img_id)
                answers.append(answer)
                img_urls[img_id] = img_url
        elif question_type in [
            'Compose(ImageQ,TextQ)', 'Compose(ImageQ,TableQ)']:
            assert len(metadata['arg2_meta']['wiki_entities_in_question']) == 1
            assert len(qa['answers']) == 1
            img_ids = [metadata['arg2_meta']['wiki_entities_in_question'][0]['wiki_title']]
            answers = [qa['answers'][0]['answer']]
            img_urls[img_ids[0]] = metadata['arg2_meta']['image_url']
        elif question_type in {
                'Compose(TextQ,ImageListQ)',
                'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
                'Compare(Compose(TableQ,ImageQ),TableQ)',
                'Compose(TableQ,ImageListQ)',
                'Intersect(ImageListQ,TableQ)',
                'Intersect(ImageListQ,TextQ)',
                }:
            try:
                arg = 'arg2' if question_type == 'Intersect(ImageListQ,TextQ)' else 'arg1'
                assert metadata[arg+'_modality'][0] == 'image'
                entities = metadata[arg+'_meta']['wiki_entities_in_question_names']
                correct_answers = {am['answer'] for am in metadata[arg+'_answers']}
                if question_type in QUESTION_TYPES_WITH_SINGLE_CORRECT_ANS:
                    assert len(correct_answers) == 1
                img_ids, answers = [], []
                urls = metadata[arg+'_meta']['image_urls']
                assert len(urls) == len(entities), 'Unexpected length mismatch: entities %d, urls: %d' % (len(entities), len(urls))
                for img_id, img_url in zip(entities, urls):
                    img_urls[img_id] = img_url
                    answer = IS_ANSWER_TOK if img_id in correct_answers else IS_NOT_ANSWER_TOK
                    img_ids.append(img_id)
                    answers.append(answer)
            except Exception as e:
                print(traceback.format_exc())
                import pdb; pdb.set_trace()
        elif 'image' in question_type.lower():
            print(question_type)
            import pdb; pdb.set_trace()
            raise ValueError('Question type %s should be supported.' % question_type)
        else:
            # Non image types. Add to image urls just to be sure.
            for doc_id in metadata['image_doc_ids']:
                doc = docs[doc_id]
                img_id = doc['image']['title']
                img_urls[img_id] =  doc['image']['url']
            continue
         
        if question_type in IMAGE_AS_SECOND_HOP_QUESTION_TYPES:
            hop = 1
            if question_type in ['Compose(ImageQ,TableQ)', 'Compose(ImageQ,TextQ)']:
                golden_bridge_entities = [img_ids[0]]
            elif question_type in ['Intersect(ImageListQ,TextQ)']:
                golden_bridge_texts = {am['answer'] for am in metadata['arg2_answers']}
                text_to_title = {e['text']: e['wiki_title'] for e in metadata['wiki_entities_in_question']}
                golden_bridge_entities = {
                    text_to_title[text] for text in golden_bridge_texts
                    if text in text_to_title and text_to_title[text] in img_ids
                }
                for entity in metadata['wiki_entities_in_answers']:
                    golden_bridge_entities.add(entity['wiki_title'])
                golden_bridge_entities = list(golden_bridge_entities)
                for entity in golden_bridge_entities:
                    assert entity in img_ids
            else:
                raise NotImplementedError('Unsupported second hop type: %s' % question_type)
        else:
            hop = 0
            golden_bridge_entities = []
        
        # Add distractors.
        for doc_id in metadata['image_doc_ids']:
            doc = docs[doc_id]
            img_id = doc['image']['title']
            if img_id in img_ids:
                continue
            img_url = doc['image']['url']
            img_urls[img_id] = img_url
            img_ids.append(img_id)
            answers.append(DISTRACTOR_TOK)
        
        if not img_ids:
            import pdb; pdb.set_trace()
        info.append({
            'img_ids': img_ids,
            'answers': answers,
            'question': qa['question'],
            'question_type': question_type,
            'question_id': qa['qid'],
            'golden_bridge_entities': golden_bridge_entities,
            'hop': hop,
        })
    return info, img_urls
    

def get_raw_dataset(data_dir, try_cache=True, splits=['train', 'dev', 'test']):
    cache_dir = os.path.join(data_dir, 'cache')
    use_cache = try_cache
    if not os.path.exists(os.path.join(cache_dir, 'vocab.pickle')):
        use_cache = False
    for split in splits:
        if not os.path.exists(os.path.join(cache_dir, split + '.pickle')):
            use_cache = False
    if use_cache:
        print('Found cached data, loading from it.')
        dataset = {
            split:  pickle.load(open(os.path.join(cache_dir, split + '.pickle'), 'rb'))
            for split in splits
        }
        idx2answer, answer2idx = pickle.load(open(os.path.join(cache_dir, 'vocab.pickle'), 'rb'))
        return dataset, idx2answer, answer2idx
    os.makedirs(cache_dir, exist_ok=True)
    img_local_paths = download_images(data_dir, splits)
    img_feature_filenames = get_img_features(data_dir, img_local_paths)
    dataset = {}

    all_answers = set()
    for split in splits:
        print('Loading data from %s split' % split)
        dataset[split] = {}
        filename = os.path.join(data_dir, 'MultiModalQA_%s.jsonl' % split)
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        raw_data = [json.loads(d) for d in lines]
        for raw_d in raw_data:
            context = raw_d['context']
            qas = raw_d['qas']
            processed_d, _ = process_context_and_qas(context, qas)
            for d in processed_d:
                question_id = d['question_id']
                assert len(d['img_ids']) == len(d['answers'])
                for answer in d['answers']:
                    all_answers.add(answer)
                filenames = [img_feature_filenames[img_id] for img_id in d['img_ids']]
                dataset[split][question_id] = d
                dataset[split][question_id]['img_features_files'] = filenames

        print('Done with split ' + split)
    print('Dumping pickled data.')
    for split in splits:
        pickle.dump(dataset[split],
            open(os.path.join(cache_dir, split + '.pickle'), 'wb'))

    idx2answer, answer2idx = {}, {}
    for i, answer in enumerate(sorted(list(all_answers))):
        idx2answer[i] = answer
        answer2idx[answer] = i

    pickle.dump((idx2answer, answer2idx),
        open(os.path.join(cache_dir, 'vocab.pickle'), 'wb'))
    return dataset, idx2answer, answer2idx


def process_raw_inputs(inputs, tokenizer, answer2idx, mode):
    assert mode in ['auto_routing', 'implicit_decomp', 'context_only']

    # image stuff
    if 'img_features_file' in inputs:
        raw_features, info = pickle.load(open(inputs['img_features_file'], 'rb'))
    else:
        raw_features, info = inputs['img_features']
    w, h = float(info['image_width']), float(info['image_height'])
    boxes = info['bbox']
    num_boxes = raw_features.shape[0]
    g_feat = np.sum(raw_features, axis=0, keepdims=True) / num_boxes
    num_boxes = num_boxes + 1
    img_features = np.concatenate([g_feat, raw_features], axis=0)
    img_loc = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    img_loc[:,:4] = boxes.copy()
    img_loc[:,4] = (img_loc[:,3] - img_loc[:,1]) * (img_loc[:,2] - img_loc[:,0]) / (w * h)
    img_loc[:,0] = img_loc[:,0] / w
    img_loc[:,1] = img_loc[:,1] / h
    img_loc[:,2] = img_loc[:,2] / w
    img_loc[:,3] = img_loc[:,3] / h
    g_location = np.array([0,0,1,1,1])
    img_loc = np.concatenate([np.expand_dims(g_location, axis=0), img_loc], axis=0)
    
    question  = process_raw_question(
        question=inputs['question'],
        question_type= inputs['question_type'],
        mode=mode,
        img_id=inputs['img_id'],
        hop=inputs['hop']
    )
    
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(question) + [tokenizer.sep_token]
    tokens = tokenizer.convert_tokens_to_ids(tokens)

    outputs = {
        'img_features': torch.tensor(img_features),
        'img_bboxes': torch.tensor(img_loc),
        'questions': tokens,
        'raw_question': question
    }
    if 'answer' in inputs:
        answer = inputs['answer']
        target = answer2idx[answer]
        outputs['answers'] = torch.tensor([target])

    extra_feature_names = ['question_id', 'question_type', 'img_id']
    for feature_name in extra_feature_names:
        outputs[feature_name] = inputs[feature_name]
            
    return outputs


def collate(batch, pad_id):
    outputs = {}
    outputs['img_features'] = torch.stack([el['img_features'] for el in batch]).float()
    outputs['img_bboxes'] = torch.stack([el['img_bboxes'] for el in batch]).float()
    if 'answers' in batch[0]:
        outputs['answers'] = torch.stack([el['answers'] for el in batch]).squeeze(1)

    questions = [el['questions'] for el in batch]
    lengths = [len(question) for question in questions]
    max_len = max(lengths)

    batch_size = len(questions)
    batched_questions = []
    mask = torch.zeros([batch_size, max_len])

    # Pad on front.
    # segment ids is always zero
    for i in range(batch_size):
        padd = max_len - lengths[i]
        batched_questions.append(torch.tensor([pad_id] * padd + questions[i]))
        mask[i] = torch.tensor([0.0] * padd + [1.0] * lengths[i])

    outputs['questions'] = torch.stack(batched_questions).long()
    outputs['masks'] = mask.long()

    for name in ['question_ids', 'question_types', 'img_ids']:
        if name in batch[0]:
            outputs[name] = [el[name] for el in batch]
    return outputs


class VQADataset(Dataset):
    def __init__(self, data, tokenizer, answer2idx, mode, sample_distractor_prob=0.3):
        assert mode in ['auto_routing', 'implicit_decomp', 'context_only']
        self.mode = mode
        self.sample_distractor_prob = sample_distractor_prob
        keep_fn = lambda d: mode != 'auto_routing' or d['question_type'] in AUTO_ROUTING_QTYPES
        self.data = {k: val for k, val in data.items() if keep_fn(val)}
        self.idx2qid = {i: k for i, k in enumerate(list(self.data.keys()))}

        self.tokenizer = tokenizer
        self.answer2idx = answer2idx
    
    def __getitem__(self, i):
        qid = self.idx2qid[i]
        img_ids = self.data[qid]['img_ids']
        answers = self.data[qid]['answers']
        assert len(img_ids) == len(answers)
        if len(img_ids) == 1:
            index = 0
        else:
            try:
                distractor_indexes = [
                    j for j, answer in enumerate(answers) if answer == DISTRACTOR_TOK]
                if distractor_indexes and random.random() < self.sample_distractor_prob:
                    index = random.choice(distractor_indexes)
                else:
                    index = random.choice([
                        j for j in range(len(answers)) if j not in distractor_indexes])
            except:
                import pdb; pdb.set_trace()
        
        raw_inputs = {
            'question_id': qid,
            'question_type': self.data[qid]['question_type'],
            'question': self.data[qid]['question'],
            'img_features_file': self.data[qid]['img_features_files'][index],
            'img_id': self.data[qid]['img_ids'][index],
            'answer': self.data[qid]['answers'][index],
            'hop': self.data[qid]['hop'],
        }
        
        return process_raw_inputs(raw_inputs, self.tokenizer, self.answer2idx, self.mode)

    def __len__(self):
        return len(self.idx2qid)

    def collate(self, batch):
        return collate(batch, pad_id=self.tokenizer.pad_token_id)


if __name__ == '__main__':
    datadir = os.path.join(basedir, '../../data')
    get_raw_dataset(datadir, try_cache=False, splits=['train', 'dev', 'test'])

    