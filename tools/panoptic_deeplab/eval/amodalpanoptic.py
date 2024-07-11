# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/panoptic_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import contextlib
import io
import logging
from collections import OrderedDict
import os
import json

import numpy as np

# from fvcore.common.file_io import PathManager
# from ctrl.utils.panoptic_deeplab import save_annotation

from fvcore.common.file_io import PathManager
from tools.panoptic_deeplab.save_annotations import save_annotation


class BlendPASSAmodalPanopticEvaluator:
    """
    Evaluate panoptic segmentation
    """
    def __init__(
                    self, output_dir=None,
                    train_id_to_eval_id=None,
                    thing_list_mapevalids=None,
                    label_divisor=1000,
                    void_label=255000,
                    gt_dir='./datasets/cityscapes',
                    split='val',
                    num_classes=19,
                    panoptic_josn_file=None,
                    panoptic_json_folder=None,
                    debug=None,
                    target_dataset_name=None,
                    input_image_size=None,
                    mapillary_dataloading_style='OURS',
                    logger=None
                ):
        """
        Args:
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
            label_divisor (int):
            void_label (int):
            gt_dir (str): path to ground truth annotations.
            split (str): evaluation split.
            num_classes (int): number of classes.
        """
        self.debug = debug
        if output_dir is None:
            raise ValueError('Must provide a output directory.')
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._panoptic_dir = os.path.join(self._output_dir, 'predictions')
        if self._panoptic_dir:
            PathManager.mkdirs(self._panoptic_dir)

        self._panoptic_gt_dir = os.path.join(self._output_dir, 'gts')
        if self._panoptic_gt_dir:
            PathManager.mkdirs(self._panoptic_gt_dir)

        self._predictions = []
        self._predictions_json = os.path.join(output_dir, 'predictions.json')

        self._save_gts = []
        self._save_gts_json = os.path.join(output_dir, 'gt_amodal_panoptic.json')

        self._train_id_to_eval_id = train_id_to_eval_id
        self._thing_list_mapevalids = thing_list_mapevalids
        self._label_divisor = label_divisor
        self._void_label = void_label
        self._num_classes = num_classes
        self.dataset_name = target_dataset_name
        self.input_image_size = input_image_size
        self.mapillary_dataloading_style = mapillary_dataloading_style

        self._logger = logger

        self._logger.info('tools/panoptic_deeplab/eval/amodalpanoptic.py --> class BlendPASSAmodalPanopticEvaluator: --> def __init__() --> self._logger : {}'.format(self._logger))

        self._gt_json_file = os.path.join(gt_dir, panoptic_josn_file)
        self._gt_folder = os.path.join(gt_dir, panoptic_json_folder)

        # if 'cityscapes' in target_dataset_name:
        #     self._gt_json_file = os.path.join(gt_dir, panoptic_josn_file)
        #     self._gt_folder = os.path.join(gt_dir, panoptic_json_folder)
        #
        # elif 'mapillary' in target_dataset_name:
        #     self._gt_json_file = os.path.join(gt_dir, panoptic_josn_file)
        #     self._gt_folder = os.path.join(gt_dir, panoptic_json_folder)
        # else:
        #     NotImplementedError('no implmentation error --> ctrl/eval_panop/panoptic.py --> def __init__() --> class CityscapesPanopticEvaluator()')

        self._pred_json_file = os.path.join(output_dir, 'predictions.json')
        self._pred_folder = self._panoptic_dir
        self._resultsFile = os.path.join(output_dir, 'resultPanopticSemanticLabeling.json')

    @staticmethod
    def id2rgb(id_map):
        if isinstance(id_map, np.ndarray):
            id_map_copy = id_map.copy()
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map_copy % 256
                id_map_copy //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return color

    def update(self, panoptics, gt_sem, gt_amodal, image_filename=None, image_id=None, debug=False, logger=None):
        if image_filename is None:
            raise ValueError('Need to provide image_filename.')
        if image_id is None:
            raise ValueError('Need to provide image_id.')

        gt_sem_copy = gt_sem.copy()
        ## predict
        segments_info = []
        save_file_names = []
        for pan_lab in np.unique(panoptics[0]):
            if pan_lab == self._void_label:
                continue
            pred_class = pan_lab // self._label_divisor
            if self._train_id_to_eval_id is not None:
                pred_class = self._train_id_to_eval_id[pred_class]
            segments_info.append({'id': int(pan_lab), 'category_id': int(pred_class), })
        sem_filename = image_filename + '_sem'
        save_file_names.append(sem_filename + '.png')
        save_annotation(self.id2rgb(panoptics[0]), self._panoptic_dir, sem_filename, add_colormap=False, debug=debug,
                        logger=logger)

        for panoptic in panoptics[1:]:
            # Change void region.
            # panoptic[panoptic == self._void_label] = 0

            for pan_lab in np.unique(panoptic):
                if pan_lab == self._void_label:
                    continue
                pred_class = pan_lab // self._label_divisor
                if self._train_id_to_eval_id is not None:
                    pred_class = self._train_id_to_eval_id[pred_class]
                segments_info.append(
                    {
                        'id': int(pan_lab),
                        'category_id': int(pred_class),
                    }
                )
                amodal_filename = image_filename+ '_' + str(int(pan_lab)) + '_amodal'
                save_file_names.append(amodal_filename + '.png')
                save_annotation(self.id2rgb(panoptic), self._panoptic_dir, amodal_filename, add_colormap=False, debug=debug, logger=logger)
        self._predictions.append(
            {
                'image_id': image_id,
                'file_name': image_filename + '.png',
                'save_file_names': save_file_names,
                'segments_info': segments_info,
            }
        )


        ## gt
        segments_gt_info = []
        save_gt_file_names = []
        gtlabels, gtlabels_cnt = np.unique(gt_sem_copy, return_counts=True)
        for sem_class, sem_area in zip(gtlabels, gtlabels_cnt):
            if self._thing_list_mapevalids is not None:
                if sem_class in self._thing_list_mapevalids:
                    gt_sem_copy[gt_sem_copy==sem_class] = self._void_label
                    continue
            # pan_sem_lab = sem_class * self._label_divisor
            if sem_class == (self._void_label/self._label_divisor):
                continue
            if self._train_id_to_eval_id is not None:
                sem_class = self._train_id_to_eval_id[sem_class]
            # area = np.sum(gt_sem_copy[])
            segments_gt_info.append({'id': int(sem_class), 'category_id': int(sem_class),  "iscrowd": 0, 'area':int(sem_area),})
        # gt_sem_copy[gt_sem_copy==(self._void_label/self._label_divisor)] == self._void_label
        sem_filename = image_filename + '_sem'
        save_gt_file_names.append(sem_filename + '.png')
        save_annotation(self.id2rgb(gt_sem_copy), self._panoptic_gt_dir, sem_filename, add_colormap=False, debug=debug,
                        logger=logger)


        for amask, gt_class, amo2ins in zip(gt_amodal['amasks'], gt_amodal['labels'], gt_amodal['amodaltoinstance']):
            if self._thing_list_mapevalids is not None:
                gt_class = self._thing_list_mapevalids[gt_class]

            pan_lab = int(amo2ins.split('_')[-1])
            assert gt_class == (pan_lab // self._label_divisor), 'GT label is not equl in amodalpanoptic'
            amask_area = np.sum(amask)
            segments_gt_info.append({'id': int(pan_lab), 'category_id': int(gt_class), "iscrowd": 0,  'area':int(amask_area),})

            amodal_gt_filename = image_filename + '_' + str(int(pan_lab)) + '_amodal'
            save_gt_file_names.append(amodal_gt_filename + '.png')
            panoptic_gt = amask.astype(np.int64) * pan_lab
            panoptic_gt[panoptic_gt==0] = self._void_label
            save_annotation(self.id2rgb(panoptic_gt), self._panoptic_gt_dir, amodal_gt_filename, add_colormap=False, debug=debug,
                            logger=logger)
        self._save_gts.append(
            {'image_id': image_id, 'file_name': image_filename + '.png', 'save_file_names': save_gt_file_names,
                'segments_info': segments_gt_info, })

    def evaluate(self, logger):
        import blendpassscripts.evaluation.evalAmodalPanopticSemanticLabeling as blendpass_eval

        read_gt_json_file = self._gt_json_file
        gt_json_file = self._save_gts_json
        gt_folder = self._panoptic_gt_dir
        pred_json_file = self._pred_json_file
        pred_folder = self._pred_folder
        resultsFile = self._resultsFile

        with open(read_gt_json_file, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions
        with PathManager.open(self._predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        json_data["annotations"] = self._save_gts
        with PathManager.open(self._save_gts_json, "w") as f:
            f.write(json.dumps(json_data))

        with contextlib.redirect_stdout(io.StringIO()):
            results = blendpass_eval.evaluateAmodalPanoptic(
                                                        gt_json_file,
                                                        gt_folder,
                                                        pred_json_file,
                                                        pred_folder,
                                                        resultsFile,
                                                        debug=self.debug,
                                                        dataset_name=self.dataset_name,
                                                        input_image_size=self.input_image_size,
                                                        mapillary_dataloading_style=self.mapillary_dataloading_style,
                                                        logger=logger
                                                    )

        self._logger.info(results)
        return results
