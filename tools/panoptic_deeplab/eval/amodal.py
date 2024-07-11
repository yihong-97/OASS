# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/cityscapes_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------
import logging
from collections import OrderedDict
import os
import glob
# from fvcore.common.file_io import PathManager
# from ctrl.utils.panoptic_deeplab import save_annotation
from fvcore.common.file_io import PathManager
from tools.panoptic_deeplab.save_annotations import save_annotation
import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval
# import blendpassscripts.evaluation.evalInstanceLevelSemanticLabeling as blendpass_eval
import blendpassscripts.evaluation.evalInstanceAmodalLevelSemanticLabeling as blendpass_eval
from blendpassscripts.helpers.labels import labels

class BlendPASSAmodalEvaluator:
    """
    Evaluate cityscapes instance segmentation
    """
    def __init__(
                        self,
                        output_dir=None,
                        train_id_to_eval_id=None,
                        thing_list_mapevalids=None,
                        gt_dir=None,
                        num_classes=19,
                        DEBUG=None,
                        num_samples=12,
                        dataset_name='cityscapes',
                        rgb2id=None,
                        input_image_size=None,
                        mapillary_dataloading_style='OURS',
                        logger=None
                    ):
        """
        Args:
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
            gt_dir (str): path to ground truth annotations (gtFine).
        """
        assert gt_dir, 'gt_dir must not be none !'
        self.debug = DEBUG
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        if output_dir is None:
            raise ValueError('Must provide a output directory.')
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._mask_dir = os.path.join(self._output_dir, 'amask')
        if self._mask_dir:
            PathManager.mkdirs(self._mask_dir)
        self._gt_amask_dir = os.path.join(self._output_dir, 'gt_amask')
        if self._gt_amask_dir:
            PathManager.mkdirs(self._gt_amask_dir)
        self._train_id_to_eval_id = train_id_to_eval_id
        self._thing_list_mapevalids = thing_list_mapevalids
        self.input_image_size = input_image_size
        self.mapillary_dataloading_style = mapillary_dataloading_style
        self.rgb2id = rgb2id

        self.logger = logger
        self.logger.info('tools/panoptic_deeplab/eval/amodal.py --> class BlendPASSAmodalInstanceEvaluator: --> def __init__() --> self.logger : {}'.format(self.logger))

        self._gt_dir = gt_dir
        # self.logger.info('tools/panoptic_deeplab/eval/instance.py --> class CityscapesInstanceEvaluator: --> def __init__() --> self._gt_dir:{}'.format(self._gt_dir))
        self.num_classes = num_classes

    def update(self, instances, gt_amodal, image_filename=None, debug=False, logger=None):
        pred_txt = os.path.join(self._output_dir, image_filename + "_pred.txt")
        num_instances = len(instances)

        with open(pred_txt, "w") as fout:
            for i in range(num_instances):
                pred_class = instances[i]['pred_class']
                if self._train_id_to_eval_id is not None:
                    pred_class = self._train_id_to_eval_id[pred_class]

                score = instances[i]['score']
                mask = instances[i]['pred_mask'].astype("uint8")
                png_filename = os.path.join(self._mask_dir, image_filename + "_{}_{}.png".format(i, pred_class))
                save_annotation(mask, self._mask_dir, image_filename + "_{}_{}".format(i, pred_class), add_colormap=False, scale_values=True, debug=False, logger=logger)
                fout.write("{} {} {}\n".format(os.path.join('amask', os.path.basename(png_filename)), pred_class, score))
        amask_count = 0
        gt_txt = os.path.join(self._output_dir, image_filename + "_gt.txt")
        with open(gt_txt, "w") as fout:
            for amask, gt_class, amo2ins in zip(gt_amodal['amasks'], gt_amodal['labels'], gt_amodal['amodaltoinstance']):
                if self._thing_list_mapevalids is not None:
                    gt_class = self._thing_list_mapevalids[gt_class]

                amask = amask.astype("uint8")
                png_filename = os.path.join(self._gt_amask_dir, image_filename + "_{}_{}.png".format(amask_count, gt_class))
                save_annotation(amask, self._gt_amask_dir, image_filename + "_{}_{}".format(amask_count, gt_class), add_colormap=False,
                                scale_values=True, debug=False, logger=logger)
                fout.write("{} {} {}\n".format(os.path.join('gt_amask', os.path.basename(png_filename)), gt_class, amo2ins))
                amask_count += 1
        if debug:
            logger.info(f'File saved at: {pred_txt}')
            logger.info(f'File saved at: {self._mask_dir}')
            logger.info('* There are multiple mask PNG files for each val image *')

    def evaluate(self, img_list_debug=None,  eval_type=None):
        if self._gt_dir is None:
            raise ValueError('Must provide cityscapes or mapillary path for evaluation.')

        self.logger.info("Evaluating results under {} ...".format(self._output_dir))
        # set some global states in cityscapes evaluation API, before evaluating
        blendpass_eval.args.predictionPath = os.path.abspath(self._output_dir)
        blendpass_eval.args.gtAmodalsPath = os.path.abspath(self._output_dir)
        blendpass_eval.args.predictionWalk = None
        blendpass_eval.args.JSONOutput = False
        blendpass_eval.args.colorized = False
        blendpass_eval.args.gtAmodalFile = os.path.join(self._output_dir, "gtAmodals.json")
        blendpass_eval.args.gtAmodalFile = os.path.join(self._output_dir, "gtAmodals.json")
        blendpass_eval.args.labels = labels

        gt_inst_dir = PathManager.get_local_path(self._gt_dir)
        blendpass_eval.args.gtInstancePath = gt_inst_dir
        self.logger.info('tools/panoptic_deeplab/eval/amodal.py --> class CityscapesInstanceEvaluator: --> def evaluate()')
        self.logger.info(f'gt_pano_dir: {gt_inst_dir}')
        self.logger.info(f'self.dataset_name: {self.dataset_name}')

        if 'cityscapes' in self.dataset_name:
            groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
            self.logger.info('len(groundTruthImgList)={}'.format(len(groundTruthImgList)))

        elif 'mapillary' in self.dataset_name:
            groundTruthImgList = glob.glob(os.path.join(gt_dir, "*.png"))
        elif 'blendpass' in self.dataset_name:
            groundTruthImgList = glob.glob(os.path.join(self._output_dir, "*gt.txt"))
            self.logger.info('len(groundTruthImgList)={}'.format(len(groundTruthImgList)))
        else:
            raise NotImplementedError(f'dataset name {self.dataset_name} is not recognised !')

        # during debug we are making smaller groundTruthImgList and  predictionImgList of 12 or 13 samples
        # so that we can check the complete eval cycle quicker
        # the groundTruthImgList is contains the cityscapes gtFine/val/*_gtFine_instanceIds.png or mapillary's panoptic GT labels
        if self.debug:
            groundTruthImgListTemp = []
            self.logger.info('groundTruthImgList:')
            for groundTruthImg in groundTruthImgList:
                if 'cityscapes' in self.dataset_name:
                    str1 = groundTruthImg.split('/')[5:]
                    str2 = str1[0].split('_')[:3]
                    str3 = "_".join(str2) + '_gtFine_panoptic'
                elif 'blendpass' in self.dataset_name:
                    str1 = groundTruthImg.split('/')[-1]
                    str3 = str1.replace('_gt.txt','')
                elif 'mapillary' in self.dataset_name:
                    str2 = groundTruthImg.split('/')[3]
                    str3 = str2.split('.')[0]
                else:
                    raise NotImplementedError(f'dataset name {self.dataset_name} is not recognised !')
                if str3 in img_list_debug:
                    groundTruthImgListTemp.append(groundTruthImg)
            groundTruthImgList = groundTruthImgListTemp
            assert len(groundTruthImgList) == len(img_list_debug), 'during debug we need to have groundTruthImgList eual to img_list_debug'

        assert len(groundTruthImgList), "Cannot find any ground truth images to use for evaluation"

        # generate the predictionImgList from the text files panoptic_eva/../instance/***_pred.txt
        predictionImgList = []
        if 'cityscapes' in self.dataset_name:
            for gt in groundTruthImgList:
                predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        elif 'blendpass' in self.dataset_name:
            for gt in groundTruthImgList:
                predictionImgList.append(blendpass_eval.getPrediction(gt, blendpass_eval.args))
        elif 'mapillary' in self.dataset_name:
            from os import listdir
            from os.path import isfile, join
            predictionImgList = [join(self._output_dir, f) for f in listdir(self._output_dir) if isfile(join(self._output_dir, f))]
            # each entry (or filename) in the groundTruthImgList and predictionImgList must match for the correct instance evaluation
            # so, picking each entry from groundTruthImgList and check the corresponding matched entry in the predictionImgList
            # and create an arranged predictionImgList
            predictionImgList_arranged = []
            for gt in groundTruthImgList:
                gt_base = os.path.basename(gt).split('.')[0] # get the filename from the path and remove extension .png
                for pd in predictionImgList:
                    pd_base = os.path.basename(pd)
                    if gt_base in pd_base:
                        predictionImgList_arranged.append(pd)
            predictionImgList = predictionImgList_arranged
        else:
            raise NotImplementedError(f'implementation not fond for dataset {self.dataset_name}')

        # make sure that the GT image file name is matched with the prediction image file name
        for (gt, pd) in zip(groundTruthImgList, predictionImgList):
            gt_base = os.path.basename(gt).split('.')[0]
            pd_base = os.path.basename(pd)
            if 'cityscapes' in self.dataset_name:
                strSplits = gt_base.split('_')
                gt_base = strSplits[0] + '_' + strSplits[1] + '_'  + strSplits[2] + '_'  + strSplits[3]
            elif 'blendpass' in self.dataset_name:
                gt_base = gt_base[:-len('_gt.txt')]
            assert gt_base in pd_base, 'GT image file name must match with the predicted image file name'


        if self.debug:
            self.logger.info('predictionImgList:')
            for (gt, pd) in zip(groundTruthImgList, predictionImgList):
                self.logger.info(f'groundTruthImg: {gt}')
                self.logger.info(f'predictionImg: {pd}')


        results = blendpass_eval.evaluateImgLists(
                                                    predictionImgList,
                                                    groundTruthImgList,
                                                    blendpass_eval.args,
                                                    dataset_name=self.dataset_name,
                                                    rgb2id=self.rgb2id,
                                                    input_image_size=self.input_image_size,
                                                    mapillary_dataloading_style=self.mapillary_dataloading_style,
                                                    logger=self.logger,
                                                    debug=self.debug,
                                                    eval_type=eval_type
                                                )["averages"]

        ret = OrderedDict()
        ret["amodal"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        return ret

