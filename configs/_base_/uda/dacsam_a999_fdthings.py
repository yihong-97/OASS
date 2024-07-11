_base_ = ['dacsam.py']
uda = dict(
    alpha=0.999,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[11,12,13,14,15,16,17],
    imnet_feature_dist_scale_min_ratio=0.75,
)
