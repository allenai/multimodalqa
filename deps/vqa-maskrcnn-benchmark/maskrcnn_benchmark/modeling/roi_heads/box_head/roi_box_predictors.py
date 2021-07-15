# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.return_feats = cfg.MODEL.ROI_BOX_HEAD.RETURN_FC_FEATS
        self.has_attributes = cfg.MODEL.ROI_BOX_HEAD.ATTR

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if self.has_attributes:
            self.cls_embed = nn.Embedding(num_classes, 256)
            self.attr_linear1 = nn.Linear(representation_size + 256, 512)
            self.attr_linear2 = nn.Linear(512, 400)

            nn.init.normal_(self.cls_embed.weight, std=0.01)
            nn.init.normal_(self.attr_linear1.weight, std=0.01)
            nn.init.normal_(self.attr_linear2.weight, std=0.01)
            nn.init.constant_(self.attr_linear1.bias, 0)
            nn.init.constant_(self.attr_linear2.bias, 0)

    def forward(self, x, proposals=None):
        if isinstance(x, dict):
            in_feat = x["fc7"]
        else:
            in_feat = x

        scores = self.cls_score(in_feat)
        bbox_deltas = self.bbox_pred(in_feat)

        if self.return_feats:
            x["scores"] = scores
            x["bbox_deltas"] = bbox_deltas

            if self.has_attributes:
                assert proposals is not None, "Proposals are None while attr=True"

                # get labels and indices of proposals with foreground
                all_labels = cat([prop.get_field("labels") for prop in proposals], dim=0)
                fg_idx = all_labels > 0
                fg_labels = all_labels[fg_idx]

                # slice fc7 for those indices
                fc7_fg = in_feat[fg_idx]

                # get embeddings of indices using gt cls labels
                cls_embed_out = self.cls_embed(fg_labels)

                # concat with fc7 feats
                concat_attr = cat([fc7_fg, cls_embed_out], dim=1)

                # pass through attr head layers
                fc_attr = self.attr_linear1(concat_attr)
                attr_score = F.relu(self.attr_linear2(fc_attr))
                x["attr_score"] = attr_score

            return x

        return scores, bbox_deltas


_ROI_BOX_PREDICTOR = {"FastRCNNPredictor": FastRCNNPredictor, "FPNPredictor": FPNPredictor}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
