import sys
import os
import yaml

from vilbert.vilbert import VILBertForVLTasks, BertPreTrainedModel, BertConfig, BertForMultiModalPreTraining, SimpleClassifier
from easydict import EasyDict as edict
import torch
from torch import nn
from types import SimpleNamespace


class VilbertForMQA(nn.Module):
    def __init__(self, vilbert_pretrained_model_name_or_path, config, num_labels, mask_vis=False, mask_lang=False, dropout_prob=0.1):
        super(VilbertForMQA, self).__init__()
        self.num_labels = num_labels
        self.mask_vis = mask_vis
        self.mask_lang = mask_lang
        self.task_id = 1

        self.vilbert, loading_info = VILBertForVLTasks.from_pretrained(
            vilbert_pretrained_model_name_or_path, config=config, num_labels=num_labels, output_loading_info=True)
        for k, v in loading_info.items():
            if v: print(k,':', v)
        self.dropout = nn.Dropout(dropout_prob)
        self.cls = SimpleClassifier(
            config.bi_hidden_size, config.bi_hidden_size * 2, num_labels, 0.5
        )
        self.fusion_method = config.fusion_method

    def forward(self, img_fts, img_boxes, questions, question_masks):
        device = img_fts.device
        batch_size = img_fts.shape[0]
        segment_ids = torch.zeros(questions.shape, dtype=torch.long).to(device)
        if self.mask_vis:
            img_fts = torch.zeros_like(img_fts)
            img_boxes = torch.zeros_like(img_boxes)
            image_mask = torch.zeros(img_fts.shape[:2], dtype=torch.long).to(device)
        else:
            image_mask = torch.ones(img_fts.shape[:2], dtype=torch.long).to(device)
        if self.mask_lang:
            questions = torch.zeros_like(questions)
            question_masks = torch.zeros_like(question_masks)
        
        co_attention_shape = (batch_size, img_fts.shape[1], questions.shape[1])
        if self.mask_vis or self.mask_lang:
            co_attention_mask = torch.zeros(co_attention_shape).long().to(device)
        else:
            co_attention_mask = torch.ones(co_attention_shape, dtype=torch.long).to(device)
        task_tokens = self.task_id * torch.ones((batch_size,1), dtype=torch.long).to(device)
    
    
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.vilbert.bert(
            questions,
            img_fts,
            img_boxes,
            segment_ids,
            question_masks,
            image_mask,
            co_attention_mask,
            task_tokens
        )
        
        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False

        logits = self.cls(pooled_output)
        
        return logits

    def freeze(self):
        for name, p in self.named_parameters():
            if 'vilbert' in name:
                p.requires_grad = False

    def unfreeze(self):
        for name, p in self.named_parameters():
            p.requires_grad = True



def get_config(vilbert_dir):
    args = SimpleNamespace(
        from_pretrained=os.path.join(vilbert_dir, "multi_task_model.bin"),
        bert_model="bert-base-uncased",
        config_file=os.path.join(vilbert_dir, "bert_base_6layer_6conect.json"),
        max_seq_length=101,
        train_batch_size=1,
        do_lower_case=True,
        predict_feature=False,
        seed=42,
        num_workers=0,
        baseline=False,
        img_weight=1,
        distributed=False,
        objective=1,
        visual_target=0,
        dynamic_attention=False,
        task_specific_tokens=True,
        tasks='1',
        save_name='',
        in_memory=False,
        batch_size=1,
        local_rank=-1,
        split='mteval',
        clean_train_sets=True
    )
    config = BertConfig.from_json_file(args.config_file)
    with open(os.path.join(vilbert_dir, 'vilbert_tasks.yml'), 'r') as f:
        task_cfg = edict(yaml.safe_load(f))

    task_names = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)

    timeStamp = args.from_pretrained.split('/')[-1] + '-' + args.save_name
    config = BertConfig.from_json_file(args.config_file)

    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.task_specific_tokens:
        config.task_specific_tokens = True    

    if args.dynamic_attention:
        config.dynamic_attention = True

    config.visualization = True
    return config
