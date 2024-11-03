import os
# ============= PATH ===================
proj_name = os.path.basename(__file__)[:-3]
exp_dir = '/mnt/data-2/users/jeff/exp/'  # PATH TO YOUR EXPERIMENT FOLDER
project_dir = os.path.join(exp_dir, proj_name)
train_data = '/mnt/pfs/datasets/nuscenes/v1.0-trainval/cam_all_train/v0.0.2'
test_data = '/mnt/pfs/datasets/nuscenes/v1.0-trainval/cam_all_val/v0.0.2'
ckpt_2d = ['YOUR_PATH_HERE/exp/drivedreamer-img_sd15_corners_hdmap_res448/models/checkpoint_epoch_20_step_27980/drivedreamer']
# ============= Data Parameters =================
img_width = 448
num_frames = 32
num_cams = 1  # 1 for single view, 6 for surrounding view
hz_factor = 1  # fps = 12 / hz_factor, 12 is the fps of raw nusc camera data
video_split_rate = 2  # e.g., sample N-frame videos, the first video sampled from 1 to N, the next video sampled from N/video_split_rate to N/video_split_rate+N
mv_video = False
view = 'cam_front'  # 'ALL' for surrounding view, 'cam_front' for single view, do not support others
# ============= Model Parameters =================
pos_name = 'corner'  # different control inputs for foreground agents, the choises are [box, corner, box_image, corner_image]
hdmap_dim = 8  # hidden dim of hdmap feature
max_objs_num = 100  # max number of objects in one frame
# ============= Train Parameters =================
num_machines = 1
gpu_ids = [0,1,2,3,4,5,6,7]
distributed_type = None  # DEEPSPEED
deepspeed_config = None # dict(deepspeed_config_file=os.path.join(os.path.dirname(__file__), '..', '..', 'accelerate_configs/zero2.json'))
activation_checkpointing = True
max_epochs = 20
batch_size = 1
batch_size_frame_per_gpu = batch_size * num_frames * num_cams
gradient_accumulation_steps = 1
resume = False
with_ema = True
tune_all_unet_params = False  # tune all unet 2D+3D params (better) or only tune the 3D parts (faster) 
# ============= Test Parameters =================
guidance_scale = 7.5
weight_path = None  # None is the last ckpt you have trained
# ============= Config ===================
config = dict(
    project_dir=project_dir,
    launch=dict(
        gpu_ids=gpu_ids,
        num_machines=num_machines,
        distributed_type=distributed_type,
        deepspeed_config=deepspeed_config,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=train_data,
            batch_size_per_gpu=batch_size_frame_per_gpu,
            num_workers=2,
            transform=dict(
                type='DriveDreamerTransform',
                dst_size=img_width,
                mode='long',
                pos_name=pos_name,
                max_objs=max_objs_num,
                random_choice=True,
                default_prompt='a realistic driving scene.',
                prompt_name='sd',
                dd_name='image_hdmap',
                is_train=True,
            ),
            sampler=dict(
                type='NuscVideoSampler',
                cam_num=num_cams,
                frame_num=num_frames,
                hz_factor=hz_factor,
                video_split_rate=video_split_rate,
                mv_video=mv_video,
                view=view,
            ),
        ),
        test=dict(
            data_or_config=test_data,
            batch_size_per_gpu=num_frames * num_cams,
            num_workers=0,
            transform=dict(
                type='DriveDreamerTransform',
                dst_size=img_width,
                mode='long',
                pos_name=pos_name,
                max_objs=max_objs_num,
                random_choice=False,
                prompt_name='sd',
                default_prompt='a realistic driving scene.',
                dd_name='image_hdmap',
                is_train=False,
            ),
            sampler=dict(
                type='NuscVideoSampler',
                cam_num=num_cams,
                frame_num=num_frames,
                hz_factor=hz_factor,
                video_split_rate=1,
                mv_video=mv_video,
                view=view,
            ),
        ),
    ),
    models=dict(
        drivedreamer=dict(
            unet_type='UNet2DConditionModel',
            noise_scheduler_type='DDPMScheduler',
            tune_all_unet_params=tune_all_unet_params,
            unet_from_2d_to_3d=True,
            num_frames=num_frames,
            num_cams=num_cams,
            position_net_cfg=dict(
                type='PositionNet',
                in_dim=768,
                mid_dim=512,
                box_dim=16 if 'corner' in pos_name else 4,
                feature_type='text_image' if 'image' in pos_name else 'text_only',
            ),
            grounding_downsampler_cfg=dict(
                type='GroundingDownSampler',
                in_dim=3,
                mid_dim=4,
                out_dim=hdmap_dim,
            ),
            add_in_channels=hdmap_dim,
        ),
        pretrained='runwayml/stable-diffusion-v1-5',
        pipeline_name='StableDiffusionControlPipeline',
        checkpoint=ckpt_2d,
        with_ema=with_ema,
        weight_path=weight_path,
    ),
    optimizers=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=0.0,
    ),
    schedulers=dict(
        name='constant',
        num_warmup_steps=100,
    ),
    train=dict(
        max_epochs=max_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16',
        checkpoint_interval=1,
        checkpoint_total_limit=10,
        log_with='tensorboard',
        log_interval=100,
        activation_checkpointing=activation_checkpointing,
        resume=resume,
        with_ema=with_ema,
        # max_grad_norm=1.0,
    ),
    test=dict(
        mixed_precision='fp16',
        save_dir=os.path.join(project_dir, 'vis'),
        guidance_scale=guidance_scale,
    ),
)
