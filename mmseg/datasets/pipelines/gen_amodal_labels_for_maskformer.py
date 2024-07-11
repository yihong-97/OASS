# --------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------------------------------


import numpy as np
from ..builder import PIPELINES
from blendpassscripts.helpers.labels import id2label, labels
from tools.panoptic_deeplab.utils import rgb2id
from mmdet.core import BitmapMasks
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
def isValidBox(box):
    isValid = False
    x1, y1, x2, y2 = box
    if x1 < x2 and y1 < y2:
        isValid = True
    return isValid

def get_bbox_coord(mask):
    # bbox computation for a segment
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    x = hor_idx[0]
    width = hor_idx[-1] - x + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    y = vert_idx[0]
    height = vert_idx[-1] - y + 1
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + int(width) - 1
    y2 = y1 + int(height) - 1
    bbox = [x1, y1, x2, y2]
    return bbox

@PIPELINES.register_module()
class GenAPSKITTIAmodalLabelsForMaskFormer(object):

    def __init__(self, sigma, mode, num_classes=19, gen_instance_classids_from_zero=True):
        self.ignore_label = 255
        self.label_divisor = 1000
        self.thing_list = [11, 12, 13, 14, 15, 16, 17]
        self.thing_list_mapids = {11:0, 12:1, 13:2, 14:3, 15:4, 16:5, 17:6}
        self.ignore_stuff_in_offset = True
        self.small_instance_area = 4096 # not using currently
        self.small_instance_weight = 3  # not using currently
        self.ignore_crowd_in_semantic = True
        self.ignore_crowd_in_instance = True
        self.sigma = sigma
        self.mode = mode
        self.num_classes = num_classes
        self.gen_instance_classids_from_zero = gen_instance_classids_from_zero
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def _map_instance_class_ids(self, catid):
        if self.gen_instance_classids_from_zero:
            return self.thing_list_mapids[catid]
        else:
            return catid

    def __call__(self, results):
        panoptic = results['gt_panoptic_seg']
        amodal = results['gt_amodal_segs']
        segments = results['ann_info']['segments_info']
        panoptic = rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        panoptic_only_thing_classes = np.zeros(panoptic.shape)
        max_inst_per_class = np.zeros(len(self.thing_list))
        class_id_tracker = {}
        for cid in self.thing_list:
            class_id_tracker[cid] = 1
        gt_masks = []
        gt_amasks = []
        gt_labels = []
        gt_amotoins = []
        gt_bboxes = []
        gt_abboxes = []
        gt_bboxes_ignore = np.empty([0, 4], dtype=np.float32)
        gt_abboxes_ignore = np.empty([0, 4], dtype=np.float32)
        for seg in segments:
            cat_id = seg["category_id"]
            if self.mode == 'val':
                labelInfo = id2label[cat_id]
                cat_id = labelInfo.trainId
            if self.ignore_crowd_in_semantic:
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id
            mask = (panoptic == seg["id"])
            amask_id = seg["amodal_id"]
            if not mask.sum() == 0:
                if self.ignore_crowd_in_instance:
                    if not seg['iscrowd']:
                        # gt_masks_all.append(mask.astype(np.uint8))
                        if cat_id in self.thing_list:
                            amask = amodal[amask_id]
                            if not amask.sum() == 0:
                                box = get_bbox_coord(mask)
                                abox = get_bbox_coord(amask)
                                if isValidBox(box) and isValidBox(abox):
                                    gt_masks.append(mask.astype(np.uint8))
                                    gt_amasks.append(amask)
                                    gt_labels.append(self._map_instance_class_ids(cat_id))
                                    gt_amotoins.append(str(seg["amodal_id"])+'_' + str(seg["id"]))
                                    gt_bboxes.append(box)
                                    gt_abboxes.append(abox)
                                    panoptic_only_thing_classes[panoptic == seg["id"]] = cat_id * self.label_divisor + class_id_tracker[cat_id]
                                    class_id_tracker[cat_id] += 1
                            # else:
                            #     print_log('--' * 10)
                            #     print_log('amodal mask error')
                            #     Imgname = results['ori_filename']
                            #     print_log(f'Imgname: {Imgname}, amodalid: {seg["amodal_id"]}',logger=get_root_logger())
                            #     print_log(f'mask: {mask.sum()}, amask: {amask.sum()}', logger=get_root_logger())
                            #     print_log('--' * 10)
                else:
                    if cat_id in self.thing_list:
                        amask = amodal[amask_id]
                        if not amask.sum() == 0:
                            box = get_bbox_coord(mask)
                            abox = get_bbox_coord(amask)
                            if isValidBox(box) and isValidBox(abox):
                                gt_masks.append(mask.astype(np.uint8))
                                gt_amasks.append(amask)
                                gt_amotoins.append(str(seg["amodal_id"]) + '_' + str(seg["id"]))
                                gt_labels.append(self._map_instance_class_ids(cat_id))
                                gt_bboxes.append(box)
                                gt_abboxes.append(abox)
                                panoptic_only_thing_classes[panoptic == seg["id"]] = cat_id * self.label_divisor + class_id_tracker[cat_id]
                                class_id_tracker[cat_id] += 1
        for cid in list(class_id_tracker.keys()):
            max_inst_per_class[self._map_instance_class_ids(cid)] = class_id_tracker[cid]
        gt_masks = BitmapMasks(gt_masks, height, width)
        gt_amasks = BitmapMasks(gt_amasks, height, width)
        results['gt_masks'] = gt_masks
        results['gt_amasks'] = gt_amasks
        results['gt_semantic_seg'] = semantic.astype('long')
        results['gt_panoptic_only_thing_classes'] = panoptic_only_thing_classes.astype('long')
        results['gt_labels'] = np.asarray(gt_labels).astype('long')
        results['gt_amodaltoinstance'] = gt_amotoins
        results['max_inst_per_class'] = max_inst_per_class.astype('long')
        results['gt_bboxes'] = np.asarray(gt_bboxes).astype(np.float32)
        results['gt_abboxes'] = np.asarray(gt_abboxes).astype(np.float32)
        results['gt_bboxes_ignore'] = gt_bboxes_ignore
        results['gt_abboxes_ignore'] = gt_abboxes_ignore
        # adding the fields
        results['bbox_fields'] = ['gt_bboxes_ignore', 'gt_bboxes']
        results['abbox_fields'] = ['gt_bboxes_ignore', 'gt_abboxes']
        results['mask_fields'] = ['gt_masks']
        results['amask_fields'] = ['gt_amasks']
        results['seg_fields'] = ['gt_semantic_seg']
        results['pan_fields'] = ['gt_panoptic_only_thing_classes']
        results['maxinst_fields'] = ['max_inst_per_class']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ignore_label={self.ignore_label}, ' \
                    f'(thing_list={self.thing_list}, ' \
                    f'(ignore_stuff_in_offset={self.ignore_stuff_in_offset}, ' \
                    f'(small_instance_area={self.small_instance_area}, ' \
                    f'(small_instance_weight={self.small_instance_weight}, ' \
                    f'(sigma={self.sigma}, ' \
                    f'(g={self.g}, '
        return repr_str
