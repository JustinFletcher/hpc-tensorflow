"""
Original author:
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python3 calculate_mean_ap_pa.py --option 

Will display a plot of precision vs recall curves at 10 distinct IoU thresholds as well as output
summary information regarding the average precision and mAP scores.

NOTE: Requires the files `ground_truth_boxes.json` and `predicted_boxes.json` which can be
downloaded from this gist.

Modified to accept `detections.pickle` file input

The detections dictionary:
detections_dict = {}
detections_dict['image_name'] = []
detections_dict['num_detections'] = []
detections_dict['detection_classes'] = []
detections_dict['detection_scores'] = []
detections_dict['detection_boxes'] = []
detections_dict['ground_truth_class_id'] = []
detections_dict['ground_truth_boxes'] = []

Also modified to perform additional performance analyses(pa), such as:
1) false positive analysis
2) false negative analysis

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import argparse
import os
import sys
import json
import glob
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

sns.set_style('white')
sns.set_context('poster')

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def loadDetectionsDataFromPickle(pickle_file):
    # Load data (deserialize)
    print('Loading pickle file...')
    with open(pickle_file, 'rb') as handle:
        unserialized_data = pickle.load(handle)
        handle.close()
        # return unserialized_data
        detections_dict = unserialized_data
        return detections_dict

def removeLowConfDetections(detections_dict, score_threshold = 0.5):

    print('removeLowConfDetections:')

    scores = detections_dict['detection_scores']
    boxes = detections_dict['detection_boxes']

    for k,s in enumerate(scores):
        # print('score:')
        # print(score)
        indices = [i for i,v in enumerate(s) if v>=score_threshold]
        # print('')
        # print('indices:')
        # print(indices)
        s = [s[i] for i in indices]
        # print('score > score_threshold:')
        # print(score)
        # sys.exit()
        scores[k] = s
        b = boxes[k]
        b = [b[i] for i in indices]
        boxes[k] = b

    detections_dict['detection_scores'] = scores
    detections_dict['detection_boxes'] = boxes

    # two_satellites = '05.20.2015_sat_26761.0013-2.jpg'
    # names = detections_dict['image_name']
    # indexes = [i for i,v in enumerate(names) if v == two_satellites]
    # print('05.20.2015_sat_26761.0013-2.jpg satellite indexes:')
    # print(indexes)
    # idx = indexes[0]

    # print('reduced scores:',detections_dict['detection_scores'][idx])
    # print('reduced boxes:',detections_dict['detection_boxes'][idx])

    # sys.exit()

    return detections_dict        

def get_boxes(detections_dict):
    pred_boxes = {}
    gt_boxes = {}

    pred_boxes = {k:{'boxes':v1,'scores':v2} for k,v1,v2 in zip(detections_dict['image_name'],detections_dict['detection_boxes'],detections_dict['detection_scores'])}
    gt_boxes = {k:v for k,v in zip(detections_dict['image_name'],detections_dict['ground_truth_boxes'])}

    # print('pred_boxes[04.22.2015_sat_28884.0002-1.jpg]:')
    # print(pred_boxes['04.22.2015_sat_28884.0002-1.jpg'])

    return pred_boxes, gt_boxes

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = 0
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos  += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def calc_tp_fp_fn(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    return (true_pos, false_pos, false_neg)

def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)

    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes']  = np.array(pred_boxes[img_id]['boxes' ])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    true_poss = []
    false_poss = []
    false_negs = []

    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes']  = pred_boxes_pruned[img_id]['boxes' ][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec  = calc_precision_recall(img_results)
        tp, fp, fn = calc_tp_fp_fn(img_results)
        true_poss.append(tp)
        false_poss.append(fp)
        false_negs.append(fn)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs,
        'true_poss': true_poss,
        'false_poss': false_poss,
        'false_negs': false_negs}


def plot_pr_curve(
    precisions, recalls, category='Satellite', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax


def mAP(detections_dict):

    pred_boxes, gt_boxes = get_boxes(detections_dict)

    # print('done mAP()')

    # sys.exit()

    # Run for one IoU threshold
    iou_thr = 0.7
    start_time = time.time()
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('avg precision: {:.4f}'.format(data['avg_prec']))

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls = data['recalls']
        ax = plot_pr_curve(precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

        print('recalls, precisions:')
        print(recalls)
        print(precisions)

    # prettify for printing:
    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs  = [float('{:.4f}'.format(thr)) for thr in iou_thrs]

    print('map: {:.2f}'.format(100*np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)

    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.show()

def false_pos_images(detections_dict):
    image_list = detections_dict['image_name']
    detection_boxes_list = detections_dict['detection_boxes']
    ground_truth_boxes_list = detections_dict['ground_truth_boxes']

    false_pos_images_list = []

    for i,image in enumerate(image_list):
        if(len(detection_boxes_list[i]) > len(ground_truth_boxes_list[i])):
            false_pos_images_list.append(image)

    false_pos_images_list = sorted(false_pos_images_list)

    nLen = len(detections_dict['image_name'])
    nFP = len(false_pos_images_list)
    nFP_percent = float(nFP)/float(nLen)*100.

    print('false_pos_images_list:',nFP,'%:',nFP_percent)
    for i,image in enumerate(false_pos_images_list):
        print(i+1,image)

def false_neg_images(detections_dict):
    image_list = detections_dict['image_name']
    detection_boxes_list = detections_dict['detection_boxes']
    ground_truth_boxes_list = detections_dict['ground_truth_boxes']

    false_neg_images_list = []

    for i,image in enumerate(image_list):
        if(len(detection_boxes_list[i]) < len(ground_truth_boxes_list[i])):
            false_neg_images_list.append(image)

    false_neg_images_list = sorted(false_neg_images_list)

    nLen = len(detections_dict['image_name'])
    nFN = len(false_neg_images_list)
    nFN_percent = float(nFN)/float(nLen)*100.

    print('false_neg_images_list:',nFN,'%:',nFN_percent)
    for i,image in enumerate(false_neg_images_list):
        print(i+1,image)
    
def main(unused_argv):

    pickle_file = FLAGS.pickle_file
    pickle_file_path = os.path.expanduser(pickle_file)
    print(pickle_file_path)

    detections_dict = loadDetectionsDataFromPickle(pickle_file_path)

    # Remove useless detections. That is, 100 detections were requested. Many are bogus.
    detections_dict = removeLowConfDetections(detections_dict)

    nLen = len(detections_dict['image_name'])
    print('len(detections_dict):', nLen)

    if(FLAGS.map == True):
        print('perform mAP:')
        mAP(detections_dict)

    elif(FLAGS.fp == True):
        print('find all false positive images:')
        false_pos_images(detections_dict)

    elif(FLAGS.fn == True):
        print('find all false negative images:')
        false_neg_images(detections_dict)

    else:
        print('no option selected')

    print('main: Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file',
                        type=str, 
                        default='~/tensorflow/mAP/satnet/trial_3/test/detections.pickle', 
                        help='Directory to input data pickle file')
    parser.add_argument("--map", action="store_true", help="perform mAP")
    parser.add_argument("--fp", action="store_true", help="list all the false positive images")
    parser.add_argument("--fn", action="store_true", help="list all the false negative images")
    FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main(unparsed)


