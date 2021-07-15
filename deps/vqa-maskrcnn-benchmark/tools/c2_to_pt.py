import torch as t

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format

path = "FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl"
config_name = 'modified_for_conversion_e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml'
base_path = '/private/home/meetshah/detectron/vmb/configs/visual_genome_vqa/c2/'
cfg.merge_from_file(base_path + config_name)

_d = load_c2_format(cfg, path)
newdict = _d
t.save(newdict, "model_final.pth")
