import math
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dreamer_datasets import DefaultSampler

def custom_collate_fn(batch):
    frame_idx = [item['frame_idx'] for item in batch]
    cam_type = [item['cam_type'] for item in batch]
    video_length = [item['video_length'] for item in batch]
    multiview_start_idx = [item['multiview_start_idx'] if 'multiview_start_idx' in item else None for item in batch]
    return {
        'frame_idx': frame_idx,
        'cam_type': cam_type,
        'video_length': video_length,
        'multiview_start_idx': multiview_start_idx
    }


class NuscVideoSampler(DefaultSampler):
    def __init__(self, dataset, batch_size=None, cam_num=6, frame_num=32, hz_factor=1, video_split_rate=2, mv_video=False, view='cam_front', shuffle=False, infinite=True, seed=6666, logger=None, 
                 resample_num_workers=8, resample_batch_size=64):
        super(NuscVideoSampler, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, infinite=infinite, seed=seed)
        self.view = view
        self.mv_video = mv_video
        self.hz_factor = hz_factor
        self.cam_names = ['cam_front_left', 'cam_front_right', 'cam_back_right', 'cam_back', 'cam_back_left']
        self.cam_num = cam_num
        self.frame_num = frame_num
        self.data_num_per_batch = cam_num * frame_num
        self.img_batch_size = batch_size
        video_batch_size = int(self.img_batch_size / self.data_num_per_batch)
        
        # process index according to frame_num
        logger.info('Sampling video data from image dataset (depends on num_frames, hz_factor, video_split_rate), this may take minutes...')
        logger.info('For faster debugging, please use mini-version of nuscene.')
        dataloader = DataLoader(dataset, batch_size=resample_batch_size, num_workers=resample_num_workers, collate_fn=custom_collate_fn)
        video_frame_len = hz_factor * frame_num
        video_first_frame_flag = []
        video_front_view_idxes = []
        multiview_start_idxes = []
        offset_idx = 0
        for batch in tqdm(dataloader):
            frame_idxs = np.array(batch['frame_idx'])
            video_lengths = np.array(batch['video_length'])
            flags = (frame_idxs % (video_frame_len // video_split_rate) == 0) & (frame_idxs + video_frame_len <= video_lengths)
            video_first_frame_flag.extend(flags.tolist())
            front_idx = np.where(np.array(batch['cam_type']) == 'cam_front')[0]
            video_front_view_idxes.extend((offset_idx + front_idx).tolist())
            offset_idx += len(batch['frame_idx'])
            if mv_video:
                multiview_start_idxes.extend(batch['multiview_start_idx'])
                    
        # multiview frames: [FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, ....]
        if mv_video:
            front_indexes = [i for i in video_front_view_idxes if video_first_frame_flag[i]]
            self.index = []
            for front_idx in front_indexes:
                this_idx = [multiview_start_idxes[cam_name] for cam_name in self.cam_names]
                this_idx.insert(1, front_idx)
                self.index.extend(this_idx)
                
        # single-view frames: [C1, C2, C3, ...]
        else:
            if view != 'ALL':
                self.index = [i for i in video_front_view_idxes if video_first_frame_flag[i]]
            else:
                self.index = [i for i in range(len(dataset)) if video_first_frame_flag[i]]
            
        if self.mv_video:
            self.total_size = int(math.ceil(len(self.index) / self.cam_num / video_batch_size)) * video_batch_size * self.data_num_per_batch
        else:
            self.total_size = math.ceil(len(self.index) / video_batch_size) * self.frame_num * video_batch_size
            
        logger.info('Done sampling!')
            
    def __iter__(self):
        video_size = int(self.total_size/self.frame_num)
            
        while True:
            indices = self.index
            while len(indices) < video_size:
                indices_i = copy.deepcopy(indices)
                num_data = min(len(indices_i), video_size - len(indices))
                indices = np.hstack((indices, indices_i[:num_data]))
                
            if self.shuffle:
                # multiview init frames: [FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, FL,F,FR,BR,B,BL, ....]
                if self.mv_video:
                    indices = np.array(indices)
                    indices = indices.reshape(-1, self.cam_num)
                    indices = indices.tolist()
                    indices = np.random.permutation(indices)
                    indices = indices.reshape(-1).tolist()
                    
                # single-view init frames: [C1, C2, C3, xxx]
                else:
                    indices = np.random.permutation(indices)
            
            """ note bellow is ok for multi video, and the format is 
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            [FL0,1,2,.., F0,1,2,..., FR0,1,2,..., BR0,1,2,..., B0,1,2,..., BL0,1,2,...],
            ...
            """
            new_indices = np.stack([np.array(indices) + i*self.hz_factor for i in range(self.frame_num)]).T.reshape(-1)

            yield from new_indices
            if not self.infinite:
                break