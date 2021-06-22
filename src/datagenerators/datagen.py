#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:47:19 2021

@author: danish
"""

from torch.utils.data.dataset import Dataset
import spacy
import os
from torchtext import data
from inception import DataGeneratorI3D


class ActivityNetCaptionsDataset(Dataset):
    def __init__(self, path, batch_size, mode, get_full_feat, start_token='<s>',
                 end_token='</s>', pad_token='<blank>', min_freq=1, 
                 word_emb='glove.840B.300d', modality='audio_video',
                 vid_feature_name='i3d'):
        
        self.path = path
        self.mode = mode
        self.get_full_feat = get_full_feat
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.min_freq = min_freq
        self.word_emb = word_emb
        self.vid_feature_name = vid_feature_name


        
        if mode=='train':
            self.data_path = os.path.join(self.path, 'train.csv')
        elif mode=='val_1':
            self.data_path = os.path.join(self.path, 'val_1.csv')
        elif mode=='val_2':
            self.data_path = os.path.join(self.path, 'val_2.csv')
        # elif mode=='learned_props':
        #     self.data_path = val_prop_meta_path
        else:
            raise NotImplementedError

        # caption dataset *iterator*
        self.train_vocab, self.caption_loader = self.caption_iterator()
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[self.pad_token]
        self.start_idx = self.train_vocab.stoi[self.start_token]
        self.end_idx = self.train_vocab.stoi[self.end_token]
            
        if modality=='video':
            #self.vid_features_path
            self.features_dataset = DataGeneratorI3D(self.vid_feature_name, 
                                                       self.data_path, 
                                                       self.device, 
                                                       self.pad_idx, 
                                                       self.get_full_feat
                                                       )
        elif modality=='audio':
            self.features_dataset = VGGishFeaturesDataset(
                cfg.audio_features_path, cfg.audio_feature_name, self.meta_path, 
                torch.device(cfg.device), self.pad_idx, self.get_full_feat, cfg
            )
        elif modality=='audio_video':
            self.features_dataset = AudioVideoFeaturesDataset(
                cfg.video_features_path, cfg.video_feature_name, cfg.audio_features_path, 
                cfg.audio_feature_name, self.meta_path, torch.device(cfg.device), self.pad_idx, 
                self.get_full_feat, cfg
            )
        else:
            raise Exception(f'it is not implemented for modality: {cfg.modality}')
            
        # initialize the caption loader iterator
        self.caption_loader_iter = iter(self.caption_loader)
        
    def __getitem__(self, index):
        caption_data = next(self.caption_loader_iter)
        to_return = self.features_dataset[caption_data.idx]
        to_return['caption_data'] = caption_data

        return to_return

    def __len__(self):
        return len(self.caption_loader)
    
    def update_iterator(self):
        '''This should be called after every epoch'''
        self.caption_loader_iter = iter(self.caption_loader)
        
    def dont_collate(self, batch):
        return batch[0]
    
    def caption_iterator(self):
        print(f'Constructing caption_iterator for "{self.mode}" mode')
        spacy_en = spacy.load('en')
        
        def tokenize_en(txt):
            return [token.text for token in spacy_en.tokenizer(txt)]
        
        caption = data.ReversibleField(tokenize='spacy', 
                                       init_token=self.start_token, 
                                       eos_token=self.end_token, 
                                       pad_token=self.pad_token,
                                       lower=True, 
                                       batch_first=True, 
                                       is_target=True)
        index = data.Field(sequential=False, use_vocab=False, batch_first=True)
        
        # the order has to be the same as in the table
        fields = [('video_id', None), ('caption', caption), ('start', None), 
                  ('end', None), ('duration', None), ('phase', None), 
                  ('idx', index)]
    
        dataset = data.TabularDataset(path=self.data_path, format='tsv', 
                                      skip_header=True, fields=fields)
        caption.build_vocab(dataset.caption, min_freq=self.min_freq, 
                            vectors=self.word_emb)
        train_vocab = caption.vocab
        
        #making data tabular
        dataset = data.TabularDataset(path=self.data_path, format='tsv', 
                                      skip_header=True, fields=fields)
           
        # sort_key = lambda x: data.interleave_keys(len(x.caption), len(y.caption))
        datasetloader = data.BucketIterator(dataset, self.batch_size, 
                                            sort_key=lambda x: 0, 
                                            device=self.device, 
                                            repeat=False, 
                                            shuffle=True)
        return train_vocab, datasetloader