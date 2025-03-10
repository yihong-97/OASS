# ------------------------------------------------------------------------------
# Post-processing to get instance and panoptic segmentation results.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from .semantic_post_processing import get_semantic_segmentation

__all__ = ['find_amodal_center', 'get_amodal_segmentation', 'get_amodalpanoptic_segmentation']


def find_amodal_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:  # False: ctr_hmp: tensor(1,1,1024,2048)
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1) # ctr_hmp: tensor(1,1,1024,2048)

    # NMS
    nms_padding = int((nms_kernel - 1) // 2) # nms_padding:3
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding) # ctr_hmp_max_pooled: tensor(1,1,1024,2048)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1 # ctr_hmp: tensor(1,1,1024,2048)

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze() # ctr_hmp: tensor(1024,2048) # this is only applicable for single batch, we need to do it for batch size 2
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0) # ctr_all: tensor (N,2), N is the no of center > 0
    if top_k is None: # False
        return ctr_all
    elif ctr_all.size(0) < top_k: # True, e.g. ctr_all: tensor(17,2), and top_k = 200,
        return ctr_all
    else:
        # find top k centers.
        # ctr_hmp: tensor(1024,2048), top_k=200, torch.flatten(ctr_hmp) gives you a tensor of shape 1024x2048, top_k_scores: tensor(k,)
        # having scores in descending order, the top scoreis the first element e.g. 0.94, 0.92, 0.89
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])


def group_pixels(ctr, offsets):
    """
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0) # offsets:tensor:(1,2,1024,2048)
    height, width = offsets.size()[1:] # offsets:tensor:(2,1024,2048)

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0) # coord:tensor:(2,1024,2048)

    ctr_loc = coord + offsets  # coord:tensor:(2,1024,2048) , offsets:tensor:(2,1024,2048), ctr_loc:tensor:(2,1024,2048)
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0) # ctr_loc:tensor: (1024x2048, 2)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1) # ctr:tensor: (K,1,2) ,K is no. of centers
    ctr_loc = ctr_loc.unsqueeze(0) # ctr_loc:tensor: (1, H*W, 2)

    # distance: [K, H*W], K is the no. of centers, e.g. [17, 1024*2048]
    distance = torch.norm(ctr - ctr_loc, dim=-1)

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    # torch.argmin(distance, dim=0) returns a tensor of shape [H*W]
    # it returns the center id (between 0 to K-1) for the center which has the minimum distance from a pixel in ctr_loc
    # e.g. if there are 17 centers i.e. K=17, then distance is a tensor of shape H*W each element in this center has a value between 0 and 16 (i.e. 17-1)
    # now if you reshape this tensor H*W to [1,H,W] then you get the K segments, each segment has group of pixels with segment id starting from 0 to 16
    # you can offset the segment id by addining a offset value, e.g. 1
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    return instance_id


def get_amodal_segmentation(sem_seg, ctr_hmp, offsets, thing_list, threshold=0.1, nms_kernel=3, top_k=None,
                              thing_seg=None):
    """
    Post-processing for instance segmentation, gets class agnostic instance id map.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        thing_seg: A Tensor of shape [1, H, W], predicted foreground mask, if not provided, inference from
            semantic prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
        A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if thing_seg is None: # True
        # gets foreground segmentation
        thing_seg = torch.zeros_like(sem_seg) # sem_seg: tensor:(1,1024,2048)
        for thing_class in thing_list:
            thing_seg[sem_seg == thing_class] = 1  # finding the thing mask from the semantic prediction : thing_seg: (1,1024,2048)

    ctr = find_amodal_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k) # ctr:tensor:(N,2), N is the no of center
    if ctr.size(0) == 0:
        return torch.zeros_like(sem_seg), ctr.unsqueeze(0)
    ins_seg = group_pixels(ctr, offsets)
    return thing_seg * ins_seg, ctr.unsqueeze(0)


def merge_semantic_and_amodal_v2(sem_seg, pan_seg_thing_classes, label_divisor, thing_list, stuff_area, void_label):
    thing_seg = pan_seg_thing_classes != void_label
    pan_seg = torch.zeros_like(sem_seg) + void_label
    # paste thing by majority voting
    instance_ids = torch.unique(pan_seg_thing_classes)
    for ins_id in instance_ids:
        if ins_id == 255000:
            continue
        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = (pan_seg_thing_classes == ins_id)
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        pan_seg[thing_mask] = ins_id
    # paste stuff to unoccupied area
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id in thing_list:
            # thing class
            continue
        # calculate stuff area
        stuff_mask = (sem_seg == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor
    return pan_seg

def merge_semantic_and_amodal(sem_seg, amo_segs, label_divisor, thing_list, stuff_area, void_label):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation label and class agnostic
        instance segmentation label.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ins_seg: A Tensor of shape [1, H, W], predicted instance label.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        thing_list: A List of thing class id.
        stuff_area: An Integer, remove stuff whose area is less tan stuff_area.
        void_label: An Integer, indicates the region has no confident prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    # In case thing mask does not align with semantic prediction
    amopan_segs = []
    thing_seg = torch.zeros_like(sem_seg)
    for i, amo_seg in enumerate(amo_segs):
        thing_seg[amo_seg > 0] = 1
    thing_seg = thing_seg > 0

    thing_overlap_seg = torch.zeros_like(sem_seg)
    for i, amo_seg in enumerate(amo_segs):
        thing_overlap_seg[amo_seg > 0] += 1
    thing_overlap_seg = thing_overlap_seg > 1

    semantic_thing_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        semantic_thing_seg[sem_seg == thing_class] = 1

    # paste stuff to unoccupied area
    amopan_semseg = torch.zeros_like(sem_seg) + void_label
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            # thing class
            continue
        # calculate stuff area
        stuff_mask = (sem_seg == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area:
            amopan_semseg[stuff_mask] = class_id * label_divisor
    ## adopt panoptic update, in custom.py Line 1017 panoptic_metric.update
    amopan_semseg[amopan_semseg == void_label] = 0
    amopan_segs.append(amopan_semseg)

    # keep track of instance id for each class
    class_id_tracker = {}

    # paste thing by majority voting
    for amo_seg in amo_segs:

        # if torch.unique(amo_seg) == 0:
        #     continue
        amopan_seg = torch.zeros_like(sem_seg) + void_label
        # Make sure only do majority voting within semantic_thing_seg
        ## if occluded area is large than unoccluded area
        thing_mask = (amo_seg != 0) & (semantic_thing_seg == 1)
        thing_class_mask = (amo_seg != 0) & (semantic_thing_seg == 1) & (~thing_overlap_seg)
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        if torch.nonzero(thing_class_mask).size(0) == 0:
            class_id, _ = torch.mode(sem_seg[thing_mask].view(-1, ))
        else:
            class_id, _ = torch.mode(sem_seg[thing_class_mask].view(-1, ))
        if class_id.item() in class_id_tracker:
            new_ins_id = class_id_tracker[class_id.item()]
        else:
            class_id_tracker[class_id.item()] = 1
            new_ins_id = 1
        class_id_tracker[class_id.item()] += 1
        amopan_seg[thing_mask] = class_id * label_divisor + new_ins_id
        amopan_segs[0][thing_mask] = void_label
        amopan_segs.append(amopan_seg)



    return amopan_segs

def get_amodalpanoptic_segmentation(sem, ctr_hmp, offsets, thing_list, label_divisor, stuff_area, void_label,
                              threshold=0.1, nms_kernel=3, top_k=None, foreground_mask=None):
    """
    Post-processing for panoptic segmentation.
    Arguments:
        sem: A Tensor of shape [N, C, H, W] of raw semantic output, where N is the batch size, for consistent,
            we only support N=1. Or, a processed Tensor of shape [1, H, W].
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        stuff_area: An Integer, remove stuff whose area is less tan stuff_area.
        void_label: An Integer, indicates the region has no confident prediction.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        foreground_mask: A Tensor of shape [N, 2, H, W] of raw foreground mask, where N is the batch size,
            we only support N=1. Or, a processed Tensor of shape [1, H, W].
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel), int64.
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem.dim() != 4 and sem.dim() != 3:
        raise ValueError('Semantic prediction with un-supported dimension: {}.'.format(sem.dim()))
    if sem.dim() == 4 and sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if foreground_mask is not None:
        if foreground_mask.dim() != 4 and foreground_mask.dim() != 3:
            raise ValueError('Foreground prediction with un-supported dimension: {}.'.format(sem.dim()))

    if sem.dim() == 4: # False
        semantic = get_semantic_segmentation(sem)
    else:
        semantic = sem

    if foreground_mask is not None: # false
        if foreground_mask.dim() == 4:
            thing_seg = get_semantic_segmentation(foreground_mask)
        else:
            thing_seg = foreground_mask
    else:
        thing_seg = None # True executed

    instance, center = get_amodal_segmentation(semantic, ctr_hmp, offsets, thing_list, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k, thing_seg=thing_seg)
    panoptic = merge_semantic_and_amodal(semantic, instance, label_divisor, thing_list, stuff_area, void_label)

    return panoptic, center
