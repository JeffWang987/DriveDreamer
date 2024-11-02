
# Prepare nuScenes-Occupancy
**1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Folder structure:**
```
nuscenes/
|── maps/
│   ├── expansion/
├── samples/
├── sweeps/
├── lidarseg/
├── v1.0-test/
├── v1.0-trainval/
```


**2. Set up the environment, edit the ENV.py:**
```
os.environ['TORCH_HOME'] = $PATH_TO_TORCH_HOME
os.environ['TRANSFORMERS_CACHE'] = $PATH_TO_TRANSFORMERS_CACHE
os.environ['HUGGINGFACE_HUB_CACHE'] = $PATH_TO_HUGGINGFACE_HUB_CACHE
os.environ['XDG_CACHE_HOME'] = $PATH_TO_XDG_CACHE
```


**3. Convert raw nuScenes to our format for fast training (v1.0-trainval take days, v1.0-mini take hours):**
```
python ./dreamer-datasets/dd_scripts/converters/nuscenes_converter.py \
    --nusc_version v1.0-trainval \
    --data_root $ROOT_PATH_RAW_NUSCENES_DATA \
    --save_root $SAVE_PATH_PROCESSED_NUSCENES_DATA \
```
**Folder structure:**
```
$SAVE_ROOT
├── nuscenes/
│   ├── v1.0-trainval/
│   │   ├── cam_all_train/
│   │   │   ├── v0.0.1/
│   │   │   ├── v0.0.2/
│   ├── v1.0-mini/ (Optional)
```
