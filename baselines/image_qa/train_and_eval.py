import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dataset import get_raw_dataset, VQADataset
from model import VilbertForMQA, BertConfig, get_config
from pytorch_transformers import BertTokenizer

from tqdm import tqdm


basedir = os.path.dirname(os.path.abspath(__file__))


def train(epoch, loader, model, optimizer, device):
    model.train()
    model = model.to(device)

    cum_loss, count = 0.0, 0
    correct_sum = 0
    
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        model.zero_grad()

        feats = batch['img_features'].to(device)
        boxes = batch['img_bboxes'].to(device)
        questions = batch['questions'].to(device)
        mask = batch['masks'].to(device)
        targets = batch['answers']

        logits = model(feats, boxes, questions, mask).cpu()
        loss = criterion(logits, targets)
        loss.backward()

        batch_size = logits.shape[0]
        cum_loss += loss.item() * batch_size
        count += batch_size

        _, pred = logits.max(1)      
        correct = (pred == targets).float().sum().numpy()
        correct_sum += correct

        optimizer.step()

    loss = (cum_loss / count)
    acc = (correct_sum / count)
    return loss, acc
        
def validate(epoch, loader, model, device):
    model.eval()
    model = model.to(device)

    cum_loss, count = 0.0, 0
    correct_sum = 0.0
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            feats = batch['img_features'].to(device)
            boxes = batch['img_bboxes'].to(device)
            questions = batch['questions'].to(device)
            mask = batch['masks'].to(device)
            targets = batch['answers']

            logits = model(feats, boxes, questions, mask).cpu()
            loss = criterion(logits, targets)
            
            batch_size = logits.shape[0]
            cum_loss += loss.item() * batch_size
            count += batch_size
            
            _, pred = logits.max(1)
            
            correct = (pred == targets).float().sum().numpy()
            correct_sum += correct
            
    loss = (cum_loss / count)
    acc = (correct_sum / count)
    return loss, acc

def train_and_eval(hparams, model_suffix='', use_tqdm=True):
    datadir = os.path.join(basedir, '../../dataset')
    checkpoint_dir = os.path.join(basedir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    datasets, idx2answer, answer2idx = get_raw_dataset(datadir, try_cache=not hparams['force_data_reload'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    train_dataset = VQADataset(
        datasets['train'], tokenizer, answer2idx,
        mode=hparams['mode'],
        sample_distractor_prob=hparams['sample_distractor_prob'])
    train_loader = DataLoader(
        train_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=train_dataset.collate)
    val_dataset = VQADataset(datasets['dev'], tokenizer, answer2idx, hparams['mode'])
    val_loader = DataLoader(
        val_dataset, batch_size=hparams['batch_size'], shuffle=False, collate_fn=val_dataset.collate)

    num_labels = len(idx2answer)

    vilbert_dir = os.path.join(basedir, '../../deps/vilbert-multi-task')
    model = VilbertForMQA(
        vilbert_pretrained_model_name_or_path=os.path.join(vilbert_dir, "multi_task_model.bin"),
        config=get_config(vilbert_dir),
        num_labels=num_labels,
        mask_vis=hparams['mask_vis'],
        mask_lang=hparams['mask_lang'],
        dropout_prob=hparams['dropout_prob'])

    best_val_loss = 1.0e10

    model_name = 'vilbert'
    if hparams['mask_lang']: model_name += '_masklang'
    if hparams['mask_vis']: model_name += '_maskvis'
    model_name += '_' + model_suffix
    checkpoint_filename = os.path.join(checkpoint_dir, model_name + '.pt')
    
    stats = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Train frozen
    print('Training with frozen backbone.')
    model.freeze()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['frozen_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams['frozen_lr_step_size'], gamma=0.1)
    t = tqdm(range(hparams['frozen_epochs']))
    for i in t:
        train_loss, train_acc = train(i, train_loader, model, optimizer, hparams['device'])
        val_loss, val_acc = validate(i, val_loader, model, hparams['device'])
        scheduler.step()
        t.set_description((
            f'epoch: {i + 1}; '
            f'train loss: {train_loss:.3f}; '
            f'train acc: {100*train_acc:.2f}; '
            f'dev loss: {val_loss:.3f}; '
            f'dev acc: {100*val_acc:.2f}; '
        ))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_filename)
        stats['epoch'].append(i+1)
        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)

    # Train unfrozen
    print('Training full model.')
    model.unfreeze()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['unfrozen_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams['unfrozen_lr_step_size'], gamma=0.1)
    t = tqdm(range(hparams['frozen_epochs'], hparams['frozen_epochs'] + hparams['unfrozen_epochs']))
    for i in t:
        train_loss, train_acc = train(i, train_loader, model, optimizer, hparams['device'])
        val_loss, val_acc = validate(i, val_loader, model, hparams['device'])
        scheduler.step()
        t.set_description((
            f'epoch: {i + 1}; '
            f'train loss: {train_loss:.3f}; '
            f'train acc: {100*train_acc:.2f}; '
            f'dev loss: {val_loss:.3f}; '
            f'dev acc: {100*val_acc:.2f}; '
        ))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_filename)
        stats['epoch'].append(i+1)
        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)

    final_checkpoint_filename = os.path.join(checkpoint_dir, model_name + '_final.pt')
    torch.save(model.state_dict(), final_checkpoint_filename)
    
    return model, stats, checkpoint_filename
    

