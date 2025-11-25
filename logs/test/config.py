data_root = '/nas3/Datasets/SSCBenchKITTI360'
ann_file = '/nas3/Datasets/SSCBenchKITTI360/labels'
stereo_depth_root = '/nas3/Datasets/SSCBenchKITTI360/depth'
camera_used = ['left']
dataset_type = 'KITTI360Dataset'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]
voxel_x = 0.2
voxel_y = 0.2
voxel_z = 0.2
voxel_size = [0.2, 0.2, 0.2]
grid_config = dict(
    xbound=[0, 51.2, 0.4],
    ybound=[-25.6, 25.6, 0.4],
    zbound=[-2, 4.4, 0.4],
    dbound=[2.0, 58.0, 0.5])
empty_idx = 0
kitti360_class_frequencies = [
    2264087502, 20098728, 104972, 96297, 1149426, 4051087, 125103, 105540713,
    16292249, 45297267, 14454132, 110397082, 6766219, 295883213, 50037503,
    1561069, 406330, 30516166, 1950115
]
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'terrain', 'pole', 'traffic-sign',
    'other-structure', 'other-object'
]
num_class = 19
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0)
data_config = dict(
    input_size=(384, 1408),
    resize=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_KITTI360',
        data_config=dict(
            input_size=(384, 1408),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        load_stereo_depth=True,
        is_train=True,
        color_jitter=(0.4, 0.4, 0.4)),
    dict(
        type='CreateDepthFromLiDAR_KITTI360',
        data_root='/nas3/Datasets/SSCBenchKITTI360',
        dataset='kitti360'),
    dict(
        type='LoadKITTI360Annotation',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0),
        apply_bda=False,
        is_train=True,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_occ'],
        meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth'])
]
trainset_config = dict(
    type='KITTI360Dataset',
    stereo_depth_root='/nas3/Datasets/SSCBenchKITTI360/depth',
    data_root='/nas3/Datasets/SSCBenchKITTI360',
    ann_file='/nas3/Datasets/SSCBenchKITTI360/labels',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_KITTI360',
            data_config=dict(
                input_size=(384, 1408),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            load_stereo_depth=True,
            is_train=True,
            color_jitter=(0.4, 0.4, 0.4)),
        dict(
            type='CreateDepthFromLiDAR_KITTI360',
            data_root='/nas3/Datasets/SSCBenchKITTI360',
            dataset='kitti360'),
        dict(
            type='LoadKITTI360Annotation',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0),
            apply_bda=False,
            is_train=True,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='CollectData',
            keys=['img_inputs', 'gt_occ'],
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth'])
    ],
    split='train',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
    test_mode=False)
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_KITTI360',
        data_config=dict(
            input_size=(384, 1408),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        load_stereo_depth=True,
        is_train=False,
        color_jitter=None),
    dict(
        type='CreateDepthFromLiDAR_KITTI360',
        data_root='/nas3/Datasets/SSCBenchKITTI360',
        dataset='kitti360'),
    dict(
        type='LoadKITTI360Annotation',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0),
        apply_bda=False,
        is_train=False,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_occ'],
        meta_keys=[
            'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
            'stereo_depth'
        ])
]
testset_config = dict(
    type='KITTI360Dataset',
    stereo_depth_root='/nas3/Datasets/SSCBenchKITTI360/depth',
    data_root='/nas3/Datasets/SSCBenchKITTI360',
    ann_file='/nas3/Datasets/SSCBenchKITTI360/labels',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_KITTI360',
            data_config=dict(
                input_size=(384, 1408),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            load_stereo_depth=True,
            is_train=False,
            color_jitter=None),
        dict(
            type='CreateDepthFromLiDAR_KITTI360',
            data_root='/nas3/Datasets/SSCBenchKITTI360',
            dataset='kitti360'),
        dict(
            type='LoadKITTI360Annotation',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0),
            apply_bda=False,
            is_train=False,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='CollectData',
            keys=['img_inputs', 'gt_occ'],
            meta_keys=[
                'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                'stereo_depth'
            ])
    ],
    split='test',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4])
valset_config = dict(
    type='KITTI360Dataset',
    stereo_depth_root='/nas3/Datasets/SSCBenchKITTI360/depth',
    data_root='/nas3/Datasets/SSCBenchKITTI360',
    ann_file='/nas3/Datasets/SSCBenchKITTI360/labels',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_KITTI360',
            data_config=dict(
                input_size=(384, 1408),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            load_stereo_depth=True,
            is_train=False,
            color_jitter=None),
        dict(
            type='CreateDepthFromLiDAR_KITTI360',
            data_root='/nas3/Datasets/SSCBenchKITTI360',
            dataset='kitti360'),
        dict(
            type='LoadKITTI360Annotation',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0),
            apply_bda=False,
            is_train=False,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='CollectData',
            keys=['img_inputs', 'gt_occ'],
            meta_keys=[
                'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                'stereo_depth'
            ])
    ],
    split='val',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4])
data = dict(
    train=dict(
        type='KITTI360Dataset',
        stereo_depth_root='/nas3/Datasets/SSCBenchKITTI360/depth',
        data_root='/nas3/Datasets/SSCBenchKITTI360',
        ann_file='/nas3/Datasets/SSCBenchKITTI360/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_KITTI360',
                data_config=dict(
                    input_size=(384, 1408),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=True,
                color_jitter=(0.4, 0.4, 0.4)),
            dict(
                type='CreateDepthFromLiDAR_KITTI360',
                data_root='/nas3/Datasets/SSCBenchKITTI360',
                dataset='kitti360'),
            dict(
                type='LoadKITTI360Annotation',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=True,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=['img_inputs', 'gt_occ'],
                meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth'])
        ],
        split='train',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        test_mode=False),
    val=dict(
        type='KITTI360Dataset',
        stereo_depth_root='/nas3/Datasets/SSCBenchKITTI360/depth',
        data_root='/nas3/Datasets/SSCBenchKITTI360',
        ann_file='/nas3/Datasets/SSCBenchKITTI360/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_KITTI360',
                data_config=dict(
                    input_size=(384, 1408),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=False,
                color_jitter=None),
            dict(
                type='CreateDepthFromLiDAR_KITTI360',
                data_root='/nas3/Datasets/SSCBenchKITTI360',
                dataset='kitti360'),
            dict(
                type='LoadKITTI360Annotation',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=['img_inputs', 'gt_occ'],
                meta_keys=[
                    'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                    'stereo_depth'
                ])
        ],
        split='val',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    test=dict(
        type='KITTI360Dataset',
        stereo_depth_root='/nas3/Datasets/SSCBenchKITTI360/depth',
        data_root='/nas3/Datasets/SSCBenchKITTI360',
        ann_file='/nas3/Datasets/SSCBenchKITTI360/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_KITTI360',
                data_config=dict(
                    input_size=(384, 1408),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=False,
                color_jitter=None),
            dict(
                type='CreateDepthFromLiDAR_KITTI360',
                data_root='/nas3/Datasets/SSCBenchKITTI360',
                dataset='kitti360'),
            dict(
                type='LoadKITTI360Annotation',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=['img_inputs', 'gt_occ'],
                meta_keys=[
                    'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                    'stereo_depth'
                ])
        ],
        split='test',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]))
train_dataloader_config = dict(batch_size=1, num_workers=4)
test_dataloader_config = dict(batch_size=1, num_workers=4)
numC_Trans = 128
voxel_channels = [128, 256, 512]
voxel_out_indices = (0, 1, 2)
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 128
_pos_dim_ = 64
_num_layers_self_ = 2
_num_points_self_ = 8
model = dict(
    type='ScanSSC',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=
            '/nas1/hamyo/model_weights/EfficientNet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'
        )),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4],
        out_channels=[128, 128, 128, 128, 128]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=640,
        numC_Trans=128,
        cam_channels=33,
        grid_config=dict(
            xbound=[0, 51.2, 0.4],
            ybound=[-25.6, 25.6, 0.4],
            zbound=[-2, 4.4, 0.4],
            dbound=[2.0, 58.0, 0.5]),
        loss_depth_type='kld',
        loss_depth_weight=0.0001),
    img_view_transformer=dict(
        type='ViewTransformerLSS',
        downsample=8,
        grid_config=dict(
            xbound=[0, 51.2, 0.4],
            ybound=[-25.6, 25.6, 0.4],
            zbound=[-2, 4.4, 0.4],
            dbound=[2.0, 58.0, 0.5]),
        data_config=dict(
            input_size=(384, 1408),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    proposal_layer=dict(
        type='VoxelProposalLayer',
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[128, 128, 16],
        data_config=dict(
            input_size=(384, 1408),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        init_cfg=None),
    VoxFormer_head=dict(
        type='VoxFormerHead',
        volume_h=128,
        volume_w=128,
        volume_z=16,
        data_config=dict(
            input_size=(384, 1408),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        embed_dims=128,
        cross_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=1,
            encoder=dict(
                type='VoxFormerEncoder_DFA3D',
                num_layers=3,
                pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                data_config=dict(
                    input_size=(384, 1408),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformCrossAttention_DFA3D',
                            pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                            num_cams=1,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D_DFA3D',
                                embed_dims=128,
                                num_points=8,
                                num_levels=1),
                            embed_dims=128)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        self_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=1,
            use_level_embeds=False,
            use_cams_embeds=False,
            encoder=dict(
                type='VoxFormerEncoder',
                num_layers=2,
                pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                data_config=dict(
                    input_size=(384, 1408),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformSelfAttention',
                            embed_dims=128,
                            num_levels=1,
                            num_points=8)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=64,
            row_num_embed=512,
            col_num_embed=512),
        mlp_prior=True),
    occ_encoder_backbone=dict(
        type='Fuser',
        embed_dims=128,
        spatial_mixing_net=dict(
            type='Spatial_mixing_net',
            local_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=128,
                num_layer=[2, 2, 2],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2]),
            local_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[128, 128, 128],
                out_channels=128,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='trilinear', align_corners=False)))),
    pts_bbox_head=dict(
        type='OccHead',
        in_channels=[128],
        out_channel=19,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_scan_weight=1.0),
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=[
            2264087502, 20098728, 104972, 96297, 1149426, 4051087, 125103,
            105540713, 16292249, 45297267, 14454132, 110397082, 6766219,
            295883213, 50037503, 1561069, 406330, 30516166, 1950115
        ]))
learning_rate = 0.0003
training_steps = 54000
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.01)
lr_scheduler = dict(
    type='MultiStepLR', milestones=[44000], interval='step', frequency=1)
load_from = '/nas1/hamyo/model_weights/geodepth/pretrain_geodepth.pth'
config_path = 'configs/kitti360_ScanSSC.py'
ckpt_path = '/nas1/hamyo/model_weights/ScanSSC_KITTI360/modified_weight.ckpt'
seed = 7240
log_folder = 'test'
save_path = None
test_mapping = False
submit = False
eval = True
log_every_n_steps = 100
