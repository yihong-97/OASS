# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

debug = False
_base_ = [
		"../_base_/default_runtime_mmdet_mr.py",
		"../_base_/models/maskrcnn_sepaspp_amodalpassb5.py",
		"../_base_/datasets/uda_kitti360aps_to_blendpass_maskrcnn_oass.py",
		"../_base_/uda/dacsam_a999_fdthings.py",
		"../_base_/schedules/adamw.py",
		"../_base_/schedules/poly10warm.py"
]

n_gpus = 1
gpu_mtotal = 23000
total_train_time =  "21:00:00"
n_cpus = 16
mem_per_cpu = 16000
machine = "local"
# euler_template_fname = "euler_template_slurm_syn2city.sh"
resume_from = None
load_from = None
only_eval = False
only_train = False
activate_auto_scale_lr = False
auto_scale_lr = dict(enable=False, base_batch_size=16)
print_layer_wise_lr = False
file_sys = "Slurm"
launcher = None
generate_only_visuals_without_eval = False
dump_predictions_to_disk = False
evaluate_from_saved_png_predictions = False
oass_eval_temp_folder_previous = None
seed = 0
uda = dict(
        pseudo_weight_ignore_top=11,
        pseudo_weight_ignore_bottom=88,
        share_src_backward=True,
        debug_img_interval=5000,
        imnet_feature_dist_lambda=0.005,
        alpha=0.999,
        pseudo_threshold=0.968,
        disable_mix_masks=False
)

data = dict(
	    samples_per_gpu=1, # batchsize (2 source  + 2 target images)
        workers_per_gpu=1,
        train=dict(
                rare_class_sampling=dict(
                                        min_pixels=3000,
                                        class_temp=0.01,
                                        min_crop_ratio=0.5
                                        )
                ),
        val=dict(
                pano_dir="Panoptic/val_c17_trainId",
                data_root="../../../datasets/BlendPASS"
        )
)
optimizer_config =  None

optimizer = dict(
        lr=6e-05,
        paramwise_cfg=dict(
            custom_keys=dict(
                neck=dict(lr_mult=10.0), head=dict(lr_mult=10.0), pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0)
            )
        )
)

evaluation = dict(
    interval=2000,
    metric=["mIoU", "mPQ", "mAP", "mAAP", "mAPQ"],
    eval_type="maskrcnn_oass",
    dataset_name="blendpass",
    gt_dir="../../../datasets/BlendPASS/Semantic/val_c17",
    gt_dir_insta="../../../datasets/BlendPASS/Instance/val_c17",
    gt_dir_panop="../../../datasets/BlendPASS/Panoptic/",
    num_samples_debug=12,
    post_proccess_params=dict(
        num_classes=18,
        ignore_label=255,
        mapillary_dataloading_style="OURS",
        label_divisor=1000,
        train_id_to_eval_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        thing_list=[11, 12, 13, 14, 15, 16, 17],
        mask_score_threshold=0.95,
        amask_score_threshold=0.95,
        debug=False,
        dump_semantic_pred_as_numpy_array=False,
        load_semantic_pred_as_numpy_array=False,
        semantic_pred_numpy_array_location=None,
        use_semantic_decoder_for_instance_labeling=False,
        use_semantic_decoder_for_panoptic_labeling=False,
        nms_th=None,
        intersec_th=None,
        upsnet_mask_pruning=False,
        generate_thing_cls_panoptic_from_instance_pred=False
    ),
    visuals_pan_eval=False, # exp6 False
    visuals_all_eval=True, # exp6 False
    evalScale="2048x400", # exp6 None
    oass_eval_folder="UnmaskFormer_test/oass_eval",
    oass_eval_temp_folder="UnmaskFormer_test/oass_eval",
    debug=False,
    out_dir="UnmaskFormer_test/oass_eval/visuals"
)

runner = dict(
    type="IterBasedRunner",
    max_iters= 20000
)

checkpoint_config = dict(
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=10
)

log_config = dict(
    interval=50,
)
name =  "UnmaskFormer_test"
exp = 6
exp_root = "unmaskformer_experiments"
exp_sub = "exp-00006"
name_dataset =  "kitti360aps2blendpass"
name_architecture =  "maskrcnn_amodalpassb5"
name_encoder =  "amodalpassb5"
name_decoder =  "maskrcnn"
name_uda =  "dacs_rcs0.01_cpl"
name_opt =  "adamw_6e-05_pmTrue_poly10warm_1x1_40k"
checkpoint_path = None,
checkpoint_file_path = "./pretrained_edaps/UnmaskFormer.pth"
work_dir =  "UnmaskFormer_test"
git_rev =  ""

