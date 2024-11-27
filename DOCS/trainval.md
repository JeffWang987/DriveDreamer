**1. Train DriveDreamer-image Model (3D box and HDMap as conditions).**
```
python ./dreamer-train/projects/launch.py \
        --project_name DriveDreamer \
        --config_name drivedreamer-img_sd15_corners_hdmap_res448 \
        --runners drivedreamer.DriveDreamerTrainer
```

**2. Test DriveDreamer-image model (3D box and HDMap as conditions), and make visualizations.**
```
python ./dreamer-train/projects/launch.py \
        --project_name DriveDreamer \
        --config_name drivedreamer-img_sd15_corners_hdmap_res448 \
        --runners drivedreamer.DriveDreamerTester
```

**3. Train DriveDreamer-video model (3D box and HDMap as conditions).**
```
python ./dreamer-train/projects/launch.py \
        --project_name DriveDreamer \
        --config_name drivedreamer-video_sd15_corners_hdmap_res448-f32 \
        --runners drivedreamer.DriveDreamerTrainer
```

**4. Test DriveDreamer-video model (3D box and HDMap as conditions), and make visualizations.**
```
python ./dreamer-train/projects/launch.py \
        --project_name DriveDreamer \
        --config_name drivedreamer-video_sd15_corners_hdmap_res448-f32 \
        --runners drivedreamer.DriveDreamerTester
```

## Basic information of config file

<div align="center">
  
| Name |  Info |
| :----: | :----: |
| exp_dir         | Path to save logs and checkpoints |
| train_data      | The converted train dataset path (e.g., .../cam_all_train/v0.0.2) |
| test_data       | The converted test dataset path (e.g., .../cam_all_val/v0.0.2) |
| ckpt_2d         | The stage-1 trained DriveDreamer-image model path |
| hz_factor       | The video fps = 12 / hz_factor, 12 is the fps of raw nusc camera data |
| video_split_rate| To sample N-frame videos, the first video: 1 \~ N. The next video: N/video_split_rate \~ N/video_split_rate+N |
| pos_name        | Control formats for foreground objects, choises: [box, corner, box_image, corner_image], box is 2D box coordinates, corner is 3D box coordinates, box_image is 2D box image, corner_image is 3D box image |
| max_objs_num         | Maximum number of foreground objects in one frame |
| weight_path     | Specify your weight path during testing. None is the last ckpt you trained|
  
</div>
