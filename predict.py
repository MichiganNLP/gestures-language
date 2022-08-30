import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch.optim import Adam, AdamW
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import logging
import lmdb as lmdb
from tqdm import tqdm
from collections import defaultdict
import pyarrow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from scipy.interpolate import interp1d
import math
from gesture_clip import GestureCLIP
from logger import *
logger = setup_logger("pose_clip", './', 0, filename = "probing.txt")

from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.stats import binom_test

# from motion_preprocessor import DataPreprocessor
def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))

from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.animation import FuncAnimation, PillowWriter  
import random 
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from tqdm import tqdm
from dataset import SpeechMotionDataset

# need to define a new dataset with more info
class SpeechMotionPredictionDataset(SpeechMotionDataset):
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, aux_info = sample

        def flatten_word_list(words, end_time=None):
            indexes = []
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(word[0])
            return indexes

        # to tensors
        word_seq = flatten_word_list(word_seq)
        text_input = self.tokenizer(" ".join(word_seq), padding='max_length', max_length=10, truncation=True, return_tensors='pt')

        pose_input = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
      
        return pose_input, text_input, aux_info, idx
    
val_dataset = SpeechMotionPredictionDataset('/home/abzaliev/clpp/data/lmdb_val', 'valid_spanish_norm', 12)
val_loader = DataLoader(dataset=val_dataset, batch_size=32,
                          shuffle=False, drop_last=True, num_workers=10, pin_memory=True)

model = GestureCLIP(embed_dim=768, context_length=15, transformer_width=768, transformer_heads=12, transformer_layers=12)
state_dict = torch.load('/local2/abzaliev/saved_models/saved_model/roberta_dif_lr_second_attempt_7_755360.pt')
model.load_state_dict(state_dict['model_state_dict'])
model = model.cuda()


    
texts = []
all_text_features = []
all_pose_features = []
all_is_right = []
all_preds = []
languages = []
log_softmax = torch.nn.LogSoftmax()
model.train(False)
val_step, val_loss, val_acc = 0, 0.0, 0.0
vid_ids = []
original_images = []
idcs = []

with torch.no_grad():
    for _ in range(1): # this was to originally shuffle the dataset in many ways to get a lot of random pairs between gestures and utterances. Not used for the embeddings.
        languages_local = []
        vid_ids_local = []
        idcs_local = []
        original_images_local = []
        all_is_right_local = []
        all_pose_features_local = []
        all_text_features_local = []
        texts_local = []
        _loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
        for val_step, batch in tqdm(enumerate(_loader)):
            input_images, input_texts, aux, idcs_batch = batch

            lang = aux['lang']
            vid_id = aux['vid']
            
            languages_local.append(lang)
            vid_ids_local.append(vid_id)
            idcs_local.append(idcs_batch)
            original_images_local.append(input_images)

            input_images = input_images.cuda()

            input_texts['input_ids'] = input_texts['input_ids'].squeeze(1).cuda()
            input_texts['attention_mask'] = input_texts['attention_mask'].squeeze(1).cuda()

            batch_text = val_loader.dataset.tokenizer.batch_decode(input_texts['input_ids'].cpu(), skip_special_tokens=True)

            pose_features = model.encode_pose(input_images)     
            text_features = model.encode_text(input_texts)

            # normalized features
            pose_features = pose_features / pose_features.norm(dim=-1, keepdim=True)

            # get logits and predictions
            logit_scale = model.logit_scale.exp()
            logits_per_pose = logit_scale * pose_features @ text_features.t()

            labels = torch.arange(len(logits_per_pose)).to(logits_per_pose.device)
            is_right = log_softmax(logits_per_pose).argmax(dim=1).cpu() == labels.cpu()

            all_is_right_local.append(is_right)
            all_pose_features_local.append(pose_features.cpu())
            all_text_features_local.append(text_features.cpu())
            texts_local.append(batch_text)
        
        # append it all
        languages.append(languages_local)
        vid_ids.append(vid_ids_local)
        idcs.append(idcs_local)
        original_images.append(original_images_local)
        all_is_right.append(all_is_right_local)
        all_pose_features.append(all_pose_features_local)
        all_text_features.append(all_text_features_local)
        texts.append(texts_local)
        
        
all_pose_embeddings_multi = []
is_correct_multi = []
all_texts_multi = []
text_lengths_multi = []
all_languages_multi = []
all_vid_ids_multi = []
all_images_multi = []
all_idcs_multi = []

for idx in range(len(all_pose_features)): # number of samplings
    
    all_idcs = idcs[idx]
    all_idcs = torch.cat(all_idcs).cpu().numpy() # this is my permutation list
                            
    # here simply concatenate all the information together
    all_pose_embeddings = torch.cat(all_pose_features[idx]).cpu().numpy()
    is_correct = torch.cat(all_is_right[idx]).cpu().numpy()
    all_texts = [j for i in texts[idx] for j in i]
    text_lengths = [len(i.split()) for i in all_texts]
    all_languages = np.concatenate(languages[idx])
    all_vid_ids = np.concatenate(vid_ids[idx])
    all_images = np.concatenate(original_images[idx])
    
    all_pose_embeddings_multi.append(all_pose_embeddings)
    is_correct_multi.append(is_correct)
    all_texts_multi.append(all_texts)
    text_lengths_multi.append(text_lengths)
    all_languages_multi.append(all_languages)
    all_vid_ids_multi.append(all_vid_ids)
    all_images_multi.append(all_images)
    all_idcs_multi.append(all_idcs)
    
d = {}
for ids, is_cor in zip(all_idcs_multi, all_pose_embeddings_multi):
    for i, j in zip(ids, is_cor):
        d[i] = j
        
all_pose_embeddings = d

d = {}
for ids, is_cor in zip(all_idcs_multi, all_images_multi):
    for i, j in zip(ids, is_cor):
        d[i] = j
        
all_images = d

d = {}
for ids, is_cor in zip(all_idcs_multi, all_vid_ids_multi):
    for i, j in zip(ids, is_cor):
        d[i] = j
        
all_vid_ids = d

d = {}
for ids, is_cor in zip(all_idcs_multi, all_languages_multi):
    for i, j in zip(ids, is_cor):
        d[i] = j
        
all_languages = d

d = {}
for ids, is_cor in zip(all_idcs_multi, all_texts_multi):
    for i, j in zip(ids, is_cor):
        d[i] = j
        
all_texts = d

# get the languages of the ids
vid_ids = []
langs = []

with torch.no_grad():
    for step, batch in tqdm(enumerate(_loader)):
        _, _, aux, ix = batch
        vid_id = aux['vid']
        lang = aux['lang']
        langs.append(lang)
        vid_ids.append(vid_id)
        
_ids = np.concatenate(vid_ids)
_langs = np.concatenate(langs)
id2lang = dict(zip(_ids, _langs))

# is_correct = np.array([i[1] for i in sorted(ensembled_is_correct.items(), key = lambda x: x[0])])
all_pose_embeddings = np.array([i[1] for i in sorted(all_pose_embeddings.items(), key = lambda x: x[0])])
all_texts = np.array([i[1] for i in sorted(all_texts.items(), key = lambda x: x[0])])
# text_lengths = np.array([i[1] for i in sorted(text_lengths.items(), key = lambda x: x[0])])

all_languages = np.array([i[1] for i in sorted(all_languages.items(), key = lambda x: x[0])])
all_vid_ids = np.array([i[1] for i in sorted(all_vid_ids.items(), key = lambda x: x[0])])
all_images = np.array([i[1] for i in sorted(all_images.items(), key = lambda x: x[0])])


# with open('is_correct.pickle', 'wb') as f:
#     pickle.dump(is_correct, f)
    
with open('all_pose_embeddings.pickle', 'wb') as f:
    pickle.dump(all_pose_embeddings, f)
    
with open('all_texts.pickle', 'wb') as f:
    pickle.dump(all_texts, f)
    
# with open('text_lengths.pickle', 'wb') as f:
#     pickle.dump(text_lengths, f)
    
with open('all_languages.pickle', 'wb') as f:
    pickle.dump(all_languages, f)
    
with open('all_vid_ids.pickle', 'wb') as f:
    pickle.dump(all_vid_ids, f)
    
with open('all_images.pickle', 'wb') as f:
    pickle.dump(all_images, f)
    
with open('id2lang.pickle', 'wb') as f:
    pickle.dump(id2lang, f)
