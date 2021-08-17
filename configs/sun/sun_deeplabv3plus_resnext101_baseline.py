import cv2

# 1. configuration for inference
nclasses = 37
ignore_label = 255
size_h = 531
size_w = 730
batch_size_per_gpu = 4
data_channels = ['rgb', 'hha']  # ['rgb', 'hha', 'depth']
image_pad_value = ()
norm_mean = ()
norm_std = ()
if 'rgb' in data_channels:
    image_pad_value += (123.675, 116.280, 103.530)
    norm_mean += (0.485, 0.456, 0.406)
    norm_std += (0.229, 0.224, 0.225)
if 'hha' in data_channels:
    image_pad_value += (123.675, 116.280, 103.530)
    norm_mean += (0.485, 0.456, 0.406)
    norm_std += (0.229, 0.224, 0.225)
if 'depth' in data_channels:
    image_pad_value += (123.675, )
    norm_mean += (0.485, )
    norm_std += (0.229, )

# img_norm_cfg = dict(mean=norm_mean,
#                     std=norm_std,
#                     max_pixel_value=255.0)
conv_cfg = dict(type='Conv')    # Conv, ShapeConv
norm_cfg = dict(type='SyncBN')      # 'FRN', 'BN', 'SyncBN', 'GN'
act_cfg = dict(type='Relu', inplace=True)    # Relu, Tlu
multi_label = False

inference = dict(
    gpu_id='0,1,2,3',
    multi_label=multi_label,
    transforms=[
        dict(type='PadIfNeeded', min_height=size_h, min_width=size_w,
             value=image_pad_value, mask_value=ignore_label),
        # dict(type='Normalize', **img_norm_cfg),
        dict(type='ToTensor'),
    ],
    model=dict(
        # model/encoder
        encoder=dict(
            backbone=dict(
                type='ResNet',
                arch='resnext101_32x8d',   # resnext101_32x8d, resnext50_32x4d, resnet152, resnet101, resnet50
                replace_stride_with_dilation=[False, False, True],
                multi_grid=[1, 2, 4],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                input_type=data_channels
            ),
            enhance=dict(
                type='ASPP',
                from_layer='c5',
                to_layer='enhance',
                in_channels=2048,
                out_channels=256,
                atrous_rates=[6, 12, 18],
                mode='bilinear',
                align_corners=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dropout=0.1,
            ),
        ),
        # model/decoder
        decoder=dict(
            type='GFPN',
            # model/decoder/blocks
            neck=[
                # model/decoder/blocks/block1
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='enhance',
                        adapt_upsample=True,
                    ),
                    lateral=dict(
                        from_layer='c2',
                        type='ConvModule',
                        in_channels=256,
                        out_channels=48,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                    post=None,
                    to_layer='p5',
                ),  # 4
            ],
        ),
        # model/head
        head=dict(
            type='Head',
            in_channels=304,
            inter_channels=256,
            out_channels=nclasses,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            num_convs=2,
            upsample=dict(
                type='Upsample',
                size=(size_h, size_w),
                mode='bilinear',
                align_corners=True,
            ),
        )
    )
)

# 2. configuration for train/test
root_workdir = '/home/leon/Summarys'
dataset_type = 'SUNDataset'
dataset_root = '/home/leon/Datasets/sun_rgbd'

common = dict(
    seed=0,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metrics=[
        dict(type='IoU', num_classes=nclasses),
        dict(type='MIoU', num_classes=nclasses, average='equal'),
        dict(type='MIoU', num_classes=nclasses, average='frequency_weighted'),
        dict(type='Accuracy', num_classes=nclasses, average='pixel'),
        dict(type='Accuracy', num_classes=nclasses, average='class'),
    ],
    dist_params=dict(backend='nccl')
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            classes=nclasses,
            imglist_name='test_list.txt',
            channels=data_channels,
            multi_label=multi_label,
        ),
        transforms=inference['transforms'],
        sampler=dict(
            type='DefaultSampler',
        ),
        dataloader=dict(
            type='DataLoader',
            samples_per_gpu=batch_size_per_gpu,
            workers_per_gpu=2,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    ),
    tta=dict(
        scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        biases=[None, None, None, None, None, None],    # bias may change the size ratio
        flip=True,
    ),
    # save_pred=True,
)

## 2.2 configuration for train
max_epochs = 800

train = dict(
    data=dict(
        train=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                imglist_name='train_list.txt',
                channels=data_channels,
                multi_label=multi_label,
            ),
            transforms=[
                dict(type='RandomScale', scale_limit=(0.5, 2), scale_step=0.25,
                     interpolation=cv2.INTER_LINEAR),
                dict(type='PadIfNeeded', min_height=size_h, min_width=size_w,
                     value=image_pad_value, mask_value=ignore_label),
                dict(type='RandomCrop', height=size_h, width=size_w),
                dict(type='HorizontalFlip', p=0.5),
                # dict(type='Normalize', **img_norm_cfg),
                dict(type='ToTensor'),
            ],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=batch_size_per_gpu,
                workers_per_gpu=2,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        val=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                imglist_name='test_list.txt',
                channels=data_channels,
                multi_label=multi_label,
            ),
            transforms=inference['transforms'],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=batch_size_per_gpu,
                workers_per_gpu=2,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            ),
        ),
    ),
    resume=None,
    criterion=dict(type='CrossEntropyLoss', ignore_index=ignore_label),
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001),
    lr_scheduler=dict(type='PolyLR', max_epochs=max_epochs, end_lr=0.002),
    max_epochs=max_epochs,
    trainval_ratio=10,
    log_interval=10,
    snapshot_interval=1000,
    save_best=True,
)
