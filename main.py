import argparse
import contextlib
import io
import itertools
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from tqdm import tqdm

from confusion_matrix import ConfusionMatrix


def per_class_AR_table(coco_eval, class_names=[], headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=[], headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def run_coco(gt_path, pred_path):
    cocoGT = COCO(gt_path)
    cocoDt = cocoGT.loadRes(pred_path)
    cocoEval = COCOeval(cocoGT, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cat_names = []
    with open(gt_path, mode="r") as gtj:
        data = json.load(gtj)
        cats = data["categories"]
        for cat in cats:
            cat_names.append(cat["name"])
    info = ""
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        cocoEval.summarize()
    info += redirect_string.getvalue()
    AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
    info += "\nper class AP:\n" + AP_table + "\n"
    AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
    info += "\nper class AR:\n" + AR_table + "\n"
    return info


def plot_conf_matrix(gt_path, pred_path, cm_save_dir, conf, iou):
    names = None
    all_pred_dict = defaultdict(list)
    all_gt_dict = defaultdict(list)

    if not os.path.exists(cm_save_dir):
        os.makedirs(cm_save_dir)

    with open(gt_path, mode="r") as gtj, open(pred_path, mode="r") as predj:
        pred_data = json.load(predj)
        gt_data = json.load(gtj)
        for pred in pred_data:
            x = pred["bbox"][0]
            y = pred["bbox"][1]
            w = pred["bbox"][2]
            h = pred["bbox"][3]
            score = pred["score"]
            cls_id = pred["category_id"]-1
            all_pred_dict[pred["image_id"]].append([x, y, w+x, h+y, score, cls_id])

        for gt in gt_data["annotations"]:
            if not gt["ignore"]:
                x = gt["bbox"][0]
                y = gt["bbox"][1]
                w = gt["bbox"][2]
                h = gt["bbox"][3]
                cls_id = gt["category_id"]-1
                all_gt_dict[gt["image_id"]].append([cls_id, x, y, w+x, h+y])

        names = [x["name"] for x in gt_data["categories"]]

    num_classes = len(names)
    conf_mat = ConfusionMatrix(num_classes=num_classes, CONF_THRESHOLD=conf, IOU_THRESHOLD=iou)

    # process image by image
    for img_id in (pbar := tqdm(all_gt_dict.keys())):
        pbar.set_description(f"Processing Image {img_id}")
        rel_gts = all_gt_dict[img_id]
        rel_preds = all_pred_dict[img_id]
        conf_mat.process_batch(np.array(rel_preds), np.array(rel_gts))

    conf_mat.plot(save_dir=cm_save_dir, fig_name=f"confusion_matrix_{num_classes}.png", names=names)
    conf_mat_matrix = conf_mat.return_matrix()

    with open(os.path.join(cm_save_dir, f"cm_{num_classes}.txt"), mode="w") as out:
        for i in range(num_classes):
            out.write(f"Class: {names[i]}")
            current_col = conf_mat_matrix[:, i][:-1]
            gt_count = int(conf_mat_matrix[:, i].sum())
            tp_count = int(conf_mat_matrix[i, i])
            missed = int(conf_mat_matrix[:, i][-1])

            out.write(f"\nTotal GT counts: {gt_count}")
            out.write(f"\nTotal TP counts: {tp_count}")
            out.write(f"\nTotal FN counts: {gt_count-tp_count} with {missed} missed")

            skip_i_row = np.delete(current_col.copy(), i)
            top3 = np.argpartition(skip_i_row, -3)[-3:]
            top3 = top3[np.argsort(skip_i_row[top3])[::-1]]

            new_top3 = []
            for idx in top3:
                if idx < i:
                    new_top3.append(idx)
                else:
                    new_top3.append(idx + 1)

            out_str = ""
            for x, y in zip(np.array(names)[new_top3], skip_i_row[top3]):
                out_str += f" {x}:{int(y)} "
            out.write(f"\nTop confused classes with counts: {out_str}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COCO Results Analysis Toolkit.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to groundtruth JSON file.')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to prediction JSON file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to output folder.')

    args = parser.parse_args()
    gt_path = args.gt_path
    pred_path = args.pred_path
    save_dir = args.save_dir

    dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
    logger.add(f"{save_dir}/{dt_string}_console.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
               backtrace=True, diagnose=True)
    logger.info(f"Eval {pred_path} with {gt_path}")
    eval_out_info = run_coco(gt_path, pred_path)
    logger.info(eval_out_info)
    conf = 0.3
    iou = 0.5
    plot_conf_matrix(gt_path, pred_path, save_dir, conf, iou)
