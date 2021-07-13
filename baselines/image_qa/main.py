import collections
import os
import pickle
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch

from train_and_eval import train_and_eval
from dataset import get_raw_dataset, VQADataset
import inference
from baselines.common_utils import IMAGE_AS_SECOND_HOP_QUESTION_TYPES


from pytorch_transformers import BertTokenizer

import argparse

import matplotlib.pyplot as plt

basedir = os.path.dirname(os.path.abspath(__file__))


def f1_score(predictions, ground_truths):
    if len(ground_truths) == 0:
        return float(len(predictions) == 0)
    recall_vec = [
        1.0 if gt in predictions else 0.0 for gt in ground_truths]
    recall = np.average(recall_vec)
    if len(predictions) == 0:
        precision = 1.0
    else:
        precision_vec = [
            1.0 if pred in ground_truths else 0.0 for pred in predictions]
        precision = np.average(precision_vec)
    if precision + recall == 0:
        return 0.0
    list_f1 = (2 * precision * recall) / (precision + recall)
    return list_f1

def exact_match(predictions, ground_truths):
    match = (sorted(predictions) == sorted(ground_truths))
    return match


def analyse(model, dataset, device, idx2answer, use_distractors, mode):
    model.eval()
    model = model.to(device)

    exact_match_sum = collections.defaultdict(int)
    f1_sum = collections.defaultdict(int)
    count = collections.defaultdict(int)
    with torch.no_grad():
        for data in tqdm(dataset.data.values()):
            question_type = data['question_type']
            
            if question_type not in IMAGE_AS_SECOND_HOP_QUESTION_TYPES:
                bridge_entities = None
            elif mode == 'auto_routing':
                # Use all available entities
                bridge_entities = data['img_ids']
            else:
                # Use only ground truth
                bridge_entities = data['golden_bridge_entities']

            predicted_answers, ground_truth_answers, raw_questions = inference.get_answers(
                model=model,
                raw_data=data,
                question_type=data['question_type'],
                device=device,
                tokenizer=dataset.tokenizer,
                answer2idx=dataset.answer2idx,
                idx2answer=idx2answer,
                bridge_entities=bridge_entities,
                mode=mode,
                use_distractors=use_distractors)
            f1 = f1_score(predicted_answers, ground_truth_answers)
            em = exact_match(predicted_answers, ground_truth_answers)

            if random.random() > 0.95:
                print('='*50)
                print('Printing example (' + data['question_type'] + ')')
                print('All img ids:', data['img_ids'])
                print('Bridge entities:', bridge_entities)
                print('Questions:')
                for question in raw_questions:
                    print(question)
                print('Predicted answers:', predicted_answers)
                print('Ground Truth answers:', ground_truth_answers)


            exact_match_sum[question_type] += em
            f1_sum[question_type] += f1
            count[question_type] += 1.

    return exact_match_sum, f1_sum, count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="The batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Which device to use",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.01,
        help="Dropout rate"
    )
    parser.add_argument(
        "--frozen_epochs",
        type=float,
        default=25,
        help="Number of epochs to train with frozen backbone"
    )
    parser.add_argument(
        "--frozen_lr",
        type=float,
        default=2e-5,
        help="Learning rate to train with frozen backbone"
    )
    parser.add_argument(
        "--frozen_lr_step_size",
        type=float,
        default=15,
        help="Learning rate step size with frozen backbone"
    )
    parser.add_argument(
        "--unfrozen_epochs",
        type=float,
        default=50,
        help="Number of epochs to train with full model"
    )
    parser.add_argument(
        "--unfrozen_lr",
        type=float,
        default=2e-6,
        help="Learning rate to train with full model"
    )
    parser.add_argument(
        "--unfrozen_lr_step_size",
        type=float,
        default=25,
        help="Learning rate step size with full model"
    )
    parser.add_argument(
        "--sample_distractor_prob",
        type=float,
        default=0.3,
        help="Learning rate step size with full model"
    )
    parser.add_argument(
        "--mask_vis",
        type=bool,
        default=False,
        help="Mask visual information"
    )
    parser.add_argument(
        "--mask_lang",
        type=bool,
        default=False,
        help="Mask textual contents in context"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='implicit_decomp',
        help="Which mode to use. One of 'implicit_decomp', 'context_only', 'auto_routing'"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(basedir, '../downloads/MMQA/'),
        help="Directory where data is stored."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    hparams = {
        'batch_size': args.batch_size,
        'device': args.device,
        'dropout_prob': args.dropout_prob,
        'frozen_epochs': args.frozen_epochs,
        'frozen_lr': args.frozen_lr,
        'frozen_lr_step_size': args.frozen_lr_step_size,
        'unfrozen_epochs': args.unfrozen_epochs,
        'unfrozen_lr': args.unfrozen_lr,
        'unfrozen_lr_step_size': args.unfrozen_lr_step_size,
        'sample_distractor_prob': args.sample_distractor_prob,
        'force_data_reload': False,
        'mask_vis': args.mask_vis,
        'mask_lang': args.mask_lang
    }
    print('Using hparams:')
    for hparam, val in hparams.items():
        print('   %s: %s' % (hparam, val))

    datadir = args.data_dir
    analysis_dir = os.path.join(basedir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    training_stats_dir = os.path.join(basedir, 'training_stats')
    os.makedirs(training_stats_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    time_signature = datetime.now().strftime("%d|%m|%Y_%H:%M:%S")
    mode = args.mode
    assert mode in ['implicit_decomp', 'context_only', 'auto_routing']
    print('='*50)
    print('Starting training with mode %s' % mode)
    hparams['mode'] = mode

    model_suffix = mode + '_' + time_signature
    
    model, training_stats, _ = train_and_eval(hparams, model_suffix)

    datasets, idx2answer, answer2idx = get_raw_dataset(datadir)
    dev_dataset = VQADataset(
        datasets['test'], tokenizer, answer2idx, mode=hparams['mode'])

    for stat_name in ['train_acc', 'train_loss', 'val_acc', 'val_loss']:
        fig = plt.figure()
        plt.plot(training_stats['epoch'], training_stats[stat_name])
        filename = os.path.join(training_stats_dir, stat_name + '_' + model_suffix + '.png')
        print('Saving stat %s to file %s' % (stat_name, filename))
        plt.savefig(filename)

    print('='*100)
    print('Done training (mode=%s). Initiating analysis.' % mode)
    all_stats = {
        'with_distractors': analyse(model, dev_dataset, device, idx2answer, True, mode),
        'no_distractors': analyse(model, dev_dataset, device, idx2answer, False, mode)
    }
    for stats_name, stats in all_stats.items():
        print('Stats breakdown: (%s)' % stats_name)
        exact_match_sums, f1_sums, counts = stats
        total_em, total_f1, total_count = 0.0, 0.0, 0.0
        for question_type in counts:
            count = counts[question_type]
            total_em += exact_match_sums[question_type]
            total_f1 += f1_sums[question_type]
            total_count += count
            if count == 0:
                print('  %s: F1=nan, EM=nan (0 occurrences)' % question_type)
            else:
                print('  %s: F1=%.2f, EM=%.2f (%d occurrences)' % (
                    question_type,
                    100*f1_sums[question_type]/count,
                    100*exact_match_sums[question_type]/count,
                    count))
        print('  TOTAL: F1=%.2f, EM=%.2f (%d occurrences)' % (
                    100*total_f1/total_count,
                    100*total_em/total_count,
                    total_count))
        print('*' * 100)

    stats_filename = os.path.join(analysis_dir, 'stats_' + model_suffix + '.pickle')
    all_stats['training_statistics'] = training_stats
    pickle.dump(all_stats, open(stats_filename, 'wb'))
