# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/sem_seg_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import logging
from collections import OrderedDict

import numpy as np

from fvcore.common.file_io import PathManager
# from ctrl.utils.panoptic_deeplab import save_annotation
from tools.panoptic_deeplab.save_annotations import save_annotation

APS2DEN_CATEGORIES = [{"color": [128, 64, 128], "id": 0, "isthing": 0, "name": "road", "supercategory": "flat"},
    {"color": [244, 35, 232], "id": 1, "isthing": 0, "name": "sidewalk", "supercategory": "flat"},
    {"color": [70, 70, 70], "id": 2, "isthing": 0, "name": "building", "supercategory": "construction"},
    {"color": [102, 102, 156], "id": 3, "isthing": 0, "name": "wall", "supercategory": "construction"},
    {"color": [190, 153, 153], "id": 4, "isthing": 0, "name": "fence", "supercategory": "construction"},
    {"color": [153, 153, 153], "id": 5, "isthing": 0, "name": "pole", "supercategory": "object"},
    {"color": [250, 170, 30], "id": 6, "isthing": 0, "name": "traffic light", "supercategory": "object"},
    {"color": [220, 220, 0], "id": 7, "isthing": 0, "name": "traffic sign", "supercategory": "object"},
    {"color": [107, 142, 35], "id": 8, "isthing": 0, "name": "vegetation", "supercategory": "nature"},
    {"color": [152, 251, 152], "id": 9, "isthing": 0, "name": "terrain", "supercategory": "nature"},
    {"color": [70, 130, 180], "id": 10, "isthing": 0, "name": "sky", "supercategory": "sky"},
    {"color": [220, 20, 60], "id": 11, "isthing": 1, "name": "pedestrians", "supercategory": "human"},
    {"color": [255, 0, 0], "id": 12, "isthing": 1, "name": "cyclists", "supercategory": "human"},
    {"color": [0, 0, 142], "id": 13, "isthing": 1, "name": "car", "supercategory": "vehicle"},
    {"color": [0, 0, 70], "id": 14, "isthing": 1, "name": "truck", "supercategory": "vehicle"},
    {"color": [0, 60, 100], "id": 15, "isthing": 1, "name": "other vehicles", "supercategory": "vehicle"},
    {"color": [0, 80, 100], "id": 16, "isthing": 1, "name": "van", "supercategory": "vehicle"},
    {"color": [0, 0, 230], "id": 17, "isthing": 1, "name": "two-wheeler", "supercategory": "vehicle"}]
def print_results(results, logger, categories=APS2DEN_CATEGORIES):
    logger.info("#" * 41)
    logger.info("- Semantic")
    logger.info("-" * 41)
    logger.info("{:14s}| {:>5s}".format("Category", "IoU"))
    # labels = sorted(results['classIoU'].keys())
    for label in range(len(categories)):
        logger.info("{:14s}| {:5.2f}".format(
            categories[label]['name'],
            results['classIoU'][label],
        ))
    logger.info("-" * 41)
    logger.info("{:14s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "mIoU", "fwIoU", "mACC", "pACC"))

    logger.info("{:14s}| {:5.2f}  {:5.2f}  {:5.2f} {:5.2f}".format(
        'All',
        results['mIoU'],
        results['fwIoU'],
        results['mACC'],
        results['pACC']
    ))
    logger.info("#" * 41)
class SemanticEvaluator:
    """
    Evaluate semantic segmentation
    """
    def __init__(self, num_classes, ignore_label=255, output_dir=None, train_id_to_eval_id=None, logger=None, dataset_name='cityscapes'):
        """
        Args:
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
        """
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1  # store ignore label in the last class
        self._train_id_to_eval_id = train_id_to_eval_id

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._logger = logger
        self._logger.info('tools/panoptic_deeplab/eval/semantic.py -->  def __init__() --> self._logger : {}'.format(self._logger))
        self.dataset_name = dataset_name

    @staticmethod
    def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
        """Converts the predicted label for evaluation.
        There are cases where the training labels are not equal to the evaluation
        labels. This function is used to perform the conversion so that we could
        evaluate the results on the evaluation server.
        Args:
            prediction: Semantic segmentation prediction.
            train_id_to_eval_id (list): maps training id to evaluation id.
        Returns:
            Semantic segmentation prediction whose labels have been changed.
        """
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(train_id_to_eval_id):
            converted_prediction[prediction == train_id] = eval_id

        return converted_prediction

    def update(self, pred, gt, image_filename=None, debug=False, logger=None):
        pred = pred.astype(int)
        gt = gt.astype(int)
        gt[gt == self._ignore_label] = self._num_classes
        self._conf_matrix += np.bincount(self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2).reshape(self._N, self._N)
        if self._output_dir:
            if self._train_id_to_eval_id is not None:
                pred = self._convert_train_id_to_eval_id(pred, self._train_id_to_eval_id)
            if image_filename is None:
                raise ValueError('Need to provide filename to save.')
            save_annotation(pred, self._output_dir, image_filename, add_colormap=False, dataset_name=self.dataset_name, debug=debug, logger=logger)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        acc = np.zeros(self._num_classes, dtype=float)
        iou = np.zeros(self._num_classes, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res['classIoU'] = 100 * iou
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        results = OrderedDict({"sem_seg": res})
        # self._logger.info(results)
        return results
class BlendPASSSemanticEvaluator:
    """
    Evaluate semantic segmentation
    """
    def __init__(self, num_classes, ignore_label=255, output_dir=None, train_id_to_eval_id=None, logger=None, dataset_name='cityscapes'):
        """
        Args:
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
        """
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1  # store ignore label in the last class
        self._train_id_to_eval_id = train_id_to_eval_id

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._logger = logger
        self._logger.info('tools/panoptic_deeplab/eval/semantic.py -->  def __init__() --> self._logger : {}'.format(self._logger))
        self.dataset_name = dataset_name

    @staticmethod
    def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
        """Converts the predicted label for evaluation.
        There are cases where the training labels are not equal to the evaluation
        labels. This function is used to perform the conversion so that we could
        evaluate the results on the evaluation server.
        Args:
            prediction: Semantic segmentation prediction.
            train_id_to_eval_id (list): maps training id to evaluation id.
        Returns:
            Semantic segmentation prediction whose labels have been changed.
        """
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(train_id_to_eval_id):
            converted_prediction[prediction == train_id] = eval_id

        return converted_prediction

    def update(self, pred, gt, image_filename=None, debug=False, logger=None):
        pred = pred.astype(int)
        gt = gt.astype(int)
        gt[gt == self._ignore_label] = self._num_classes
        self._conf_matrix += np.bincount(self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2).reshape(self._N, self._N)
        if self._output_dir:
            if self._train_id_to_eval_id is not None:
                pred = self._convert_train_id_to_eval_id(pred, self._train_id_to_eval_id)
            if image_filename is None:
                raise ValueError('Need to provide filename to save.')
            save_annotation(pred, self._output_dir, image_filename, add_colormap=False, dataset_name=self.dataset_name, debug=debug, logger=logger)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        acc = np.zeros(self._num_classes, dtype=float)
        iou = np.zeros(self._num_classes, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res['classIoU'] = 100 * iou
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        results = OrderedDict({"sem_seg": res})
        # self._logger.info(results)
        print_results(results["sem_seg"], self._logger)
        return results
