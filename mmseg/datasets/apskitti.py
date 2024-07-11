from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ApsKittiDataset(CustomDataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'pedestrians', 'cyclists', 'car', 'truck', 'other vehicles', 'van', 'two-wheeler')
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230]]

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(ApsKittiDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_panoptic.png',
            # seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
