# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

dataset_type = 'BlendPASS'
data_root = '../../../datasets/BlendPASS'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (376, 376)
num_classes = 18
apskitti_train_pipeline = [
                            dict(type='LoadImageFromFile'),
                            dict(type='LoadAmodalAnnotations'),
                            dict(type='Resize', img_scale=(1408, 376)),
                            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                            dict(type='RandomFlip', prob=0.5),
                            dict(type='GenAPSKITTIAmodalLabelsForMaskFormer',
                                 sigma=8,
                                 mode='train',
                                 num_classes=num_classes,
                                 gen_instance_classids_from_zero=True,
                                 ),
                            dict(type='Normalize', **img_norm_cfg),
                            dict(type='DefaultFormatBundleMmdet'),
                            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_abboxes', 'gt_labels', 'gt_masks', 'gt_amasks', 'gt_semantic_seg', 'gt_panoptic_only_thing_classes', 'max_inst_per_class']),
                        ]

blendpass_train_pipeline = [
                            dict(type='LoadImageFromFile'),
                            # dict(type='LoadPanopticAnnotations'),
                            dict(type='Resize', img_scale=(2048, 400)),
                            dict(type='RandomCrop', crop_size=crop_size),
                            dict(type='RandomFlip', prob=0.5),
                            dict(type='Normalize', **img_norm_cfg),
                            dict(type='DefaultFormatBundleMmdet'),
                            dict(type='Collect', keys=['img']),
                        ]

test_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(type='MultiScaleFlipAug',
                    img_scale=(2048, 400),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img']),
                                ]
                         )
                ]

data = dict(
            samples_per_gpu=4,
            workers_per_gpu=4,
            train=dict(
                        type='UDADataset',
                        source=dict(
                            type='ApsKittiDataset',
                            data_root='../../../datasets/APS_KITTI_360',
                            img_dir='Image/train',
                            depth_dir='',  # not in use
                            pano_dir='Panoptic/train',
                            ann_file='annotations/train.json',
                            pipeline=apskitti_train_pipeline),
                        target=dict(
                            type=dataset_type,
                            data_root=data_root,
                            img_dir='leftImg8bit/train',
                            depth_dir='', # not in use
                            pano_dir='Panoptic/blendpass_panoptic_train',
                            pipeline=blendpass_train_pipeline,
                        )
                    ),
            val=dict(
                type=dataset_type,
                data_root=data_root,
                img_dir='leftImg8bit/val',
                depth_dir='', # not in use
                ann_file='annotations/val.json',
                pano_dir='Panoptic/val_c17_trainId',
                pipeline=test_pipeline
                    ),
            test=dict(
                type=dataset_type,
                data_root=data_root,
                img_dir='leftImg8bit/val',
                depth_dir='', # not in use
                ann_file='annotations/val.json',
                pano_dir='Panoptic/val_c17_trainId',
                pipeline=test_pipeline
                    )
            )