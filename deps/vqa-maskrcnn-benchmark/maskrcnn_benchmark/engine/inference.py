# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.layers import nms

from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize


def process_feature_extraction(output,
                               im_scales,
                               feat_name='fc6',
                               conf_thresh=0.2):
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    feat_list = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_scales[i]
        scores = score_list[i]

        max_conf = torch.zeros((scores.shape[0])).to(cur_device)

        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                         cls_scores[keep],
                                         max_conf[keep])

        keep_boxes = torch.argsort(max_conf, descending=True)[:100]
        feat_list.append(feats[i][keep_boxes])
    return feat_list


def compute_on_dataset(model, data_loader, device, save_path=None,
                       feat_name='fc6', group_id=1, n_groups=0):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    split = data_loader.dataset.get_img_info(0)["file_name"].split("_")[1]

    get_file_name = lambda x: data_loader.dataset.get_img_info(x)['file_name']

    base_path = os.path.join(save_path, '{}/{}'.format(feat_name, split))
    if not os.path.exists(base_path):
        os.makedirs(base_path)


    for i, batch in enumerate(tqdm(data_loader)):
        if i % group_id == n_groups:
            images, targets, image_ids = batch
            images = images.to(device)
            with torch.no_grad():
                output = model(images)

                # only run for feature extraction
                if isinstance(output, tuple):
                    im_scales = [x.get_field('im_scale') for x in targets]
                    feats = process_feature_extraction(output,
                                                       im_scales=im_scales,
                                                       feat_name=feat_name)

                    for img_id, feat in zip(image_ids, feats):
                        coco_id = get_file_name(img_id).split("_")[-1][:-4]
                        _feat_name = "COCO_{}_{}".format(split, coco_id)
                        _internal_path = "{}/{}/{}".format(feat_name,
                                                           split,
                                                           _feat_name)
                        feat_save_path = os.path.join(save_path, _internal_path)
                        np.save(feat_save_path, feat.cpu().numpy())

                    output = output[1]

                output = [o.to(cpu_device) for o in output]
            results_dict.update({img_id: result \
                                 for img_id, result in zip(image_ids, output)})
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
    model,
    data_loader,
    dataset_name,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    save_path=None,
    feat_name=None,
    group_id=1,
    n_groups=0,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device,
                                     save_path, feat_name, group_id, n_groups)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args
    )
