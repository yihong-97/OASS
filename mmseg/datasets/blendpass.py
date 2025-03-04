# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import tempfile
import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image
from .builder import DATASETS
from .custom import CustomDataset
import torch
from . import ApsKittiDataset

@DATASETS.register_module()
class BlendPASS(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ApsKittiDataset.CLASSES
    PALETTE = ApsKittiDataset.PALETTE


    def __init__(self, **kwargs):
        super(BlendPASS, self).__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_panoptic.png',
            # seg_map_suffix='_gtFine_labelTrainIds.png', # original
            **kwargs)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import blendpassscripts.helpers.labels as DSLabels
        result_copy = result.copy()
        for trainId, label in DSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import blendpassscripts.helpers.labels as DSLabels
            palette = np.zeros((len(DSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in DSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir


    def evaluate(self, results, metric='mIoU', logger=None, imgfile_prefix=None, efficient_test=False,
                 eval_type=None, oass_eval_folder=None, oass_eval_temp_folder=None, dataset_name=None,
                 gt_dir=None, debug=None, num_samples_debug=None, gt_dir_insta=None, gt_dir_panop=None, gt_dir_amodal=None,
                 post_proccess_params=None, visuals_pan_eval=None, visuals_all_eval=None, out_dir=None, evalScale=None,
                 evaluate_from_saved_numpy_predictions=None, evaluate_from_saved_png_predictions=None):

        cuda = torch.device('cuda')
        """Evaluation in Cityscapes/default protocol.
        
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        print(f'####### eval_type={eval_type} #######')
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            if eval_type == 'daformer':
                eval_results.update(super(BlendPASS,self).evaluate(results, metrics, logger, efficient_test, evalScale))
            elif eval_type == 'panop_deeplab':
                eval_results.update(
                    super(BlendPASS, self).evaluate_panoptic(results, cuda, oass_eval_temp_folder, dataset_name,
                        gt_dir, debug, num_samples_debug, gt_dir_insta, gt_dir_panop, gt_dir_amodal, logger,
                        post_proccess_params, visuals_pan_eval, visuals_all_eval, evalScale, metric))

            elif eval_type == 'maskformer': # eval mask based mIoU, mPQ, mAP
                eval_results.update(
                    super(BlendPASS, self).evaluate_panoptic_for_maskformer(
                        results, cuda, oass_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_panop, logger, post_proccess_params, visuals_pan_eval, out_dir,
                    )
                )
            elif eval_type == 'maskrcnn': # only eval inst seg. mAP
                eval_results.update(
                    super(BlendPASS, self).evaluate_instance_for_maskrcnn(
                        results, cuda, oass_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_panop, logger, post_proccess_params, visuals_pan_eval, out_dir, evalScale
                                                                                )
                )
            elif eval_type == 'maskrcnn_oass':
                eval_results.update(
                    super(BlendPASS, self).evaluate_oass_for_maskrcnn(
                        results, cuda, oass_eval_temp_folder, dataset_name, gt_dir, debug, num_samples_debug,
                        gt_dir_insta, gt_dir_panop, gt_dir_amodal, logger, post_proccess_params, visuals_pan_eval, visuals_all_eval, out_dir, metric, evalScale
                                                                                )
                )
            else:
                raise NotImplementedError(f'implementation not found for eval_type={eval_type}')

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_files, tmp_dir = self.format_results(results, imgfile_prefix)

        if tmp_dir is None:
            result_dir = imgfile_prefix
        else:
            result_dir = tmp_dir.name

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in mmcv.scandir(
                self.ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results
