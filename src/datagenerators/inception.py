#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:36:08 2021

@author: danish
"""

from torch.utils.data.dataset import Dataset
import torch
import pand as pd


class DataGeneratorI3D(Dataset):
    def __init__(self, features_path, feature_name, meta_path, device, 
                 pad_idx, get_full_feat, cfg):
        self.cfg = cfg
        self.features_path = features_path
        self.feature_name = f'{feature_name}_features'
        self.feature_names_list = [self.feature_name]
        self.device = torch.device(device)
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.get_full_feat = get_full_feat
        
        if self.feature_name == 'i3d_features':
            self.feature_size = 1024
        else:
            raise Exception(f'Inspect: "{self.feature_name}"')
    
    def __getitem__(self, indices):
        video_ids, captions, starts, ends, vid_stacks_rgb, vid_stacks_flow = [], [], [], [], [], []

        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.cfg, self.feature_names_list, video_id, start, end, duration, 
                self.pad_idx, self.get_full_feat
            )

            vid_stack_rgb, vid_stack_flow = stack['rgb'], stack['flow']
            
            # either both None or both are not None (Boolean Equivalence)
            both_are_None = vid_stack_rgb is None and vid_stack_flow is None
            none_is_None = vid_stack_rgb is not None and vid_stack_flow is not None
            assert both_are_None or none_is_None
            
            # # sometimes stack is empty after the filtering. we replace it with noise
            if both_are_None:
                # print(f'RGB and FLOW are None. Zero (1, D) @: {video_id}')
                vid_stack_rgb = fill_missing_features('zero', self.feature_size)
                vid_stack_flow = fill_missing_features('zero', self.feature_size)
    
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            
        vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
                
        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)

        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts.to(self.device),
            'ends': ends.to(self.device),
            'feature_stacks': {
                'rgb': vid_stacks_rgb.to(self.device),
                'flow': vid_stacks_flow.to(self.device),
            }
        }
        
        return batch_dict

    def __len__(self):
        return len(self.dataset)