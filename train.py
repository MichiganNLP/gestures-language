import pandas as pd
import numpy as np
import torch
import math
import os

import torch.nn.functional as F

import datetime
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeechMotionDataset
from torch.utils.data import DataLoader

from logger import *
logger = setup_logger("gesture_clip", './', 0, filename = "log.txt")

import numpy as np
from scipy.stats import circvar
    
from gesture_clip import *
from torch.optim import Optimizer, AdamW
from scheduler import get_cosine_schedule_with_warmup
from utils import save_checkpoint
from typing import Callable, Iterable, Optional, Tuple, Union

train_dataset = SpeechMotionDataset(lmdb_dir='./data/lmdb_train', lmdb_dir_spanish='./data/train_spanish_norm', n_poses=12)

train_loader = DataLoader(dataset=train_dataset, batch_size=32,
                          shuffle=True, drop_last=True, num_workers=10, pin_memory=True)


val_dataset = SpeechMotionDataset('./data/lmdb_val', './data/valid_spanish_norm', 12)

val_loader = DataLoader(dataset=val_dataset, batch_size=32,
                          shuffle=False, drop_last=True, num_workers=10, pin_memory=True)


model = GestureCLIP(embed_dim=768, context_length=15, transformer_width=768, transformer_heads=12, transformer_layers=12)
model = model.cuda()


# total training iterations
t_total = len(train_loader) // 1 * 50

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)


num_warmup_steps = int(0.5 * t_total)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= t_total)

def validate(epoch, valid_loader):
    
    log_softmax = nn.LogSoftmax(dim=-1)
    gts = []
    pose_preds = []
    text_preds = []
    
    model.train(False)
    val_step, val_loss, val_acc = 0, 0.0, 0.0
    
    with torch.no_grad():
        for val_step, batch in enumerate(valid_loader):
            input_images, input_texts = batch

            input_images = input_images.cuda()
            input_texts['input_ids'] = input_texts['input_ids'].squeeze(1).cuda()
            input_texts['attention_mask'] = input_texts['attention_mask'].squeeze(1).cuda()

            logits_per_image, logits_per_text = model(input_images, input_texts)

            labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss  = F.cross_entropy(logits_per_text, labels)
            
            pose_preds.append(log_softmax(logits_per_image).argmax(dim=1).cpu())
            text_preds.append(log_softmax(logits_per_text).argmax(dim=1).cpu())
            gts.append(labels.cpu())
            
            loss = (image_loss + text_loss) / 2
            val_loss += loss.item()
            val_step += 1
            
    acc_text = (torch.cat(text_preds) == torch.cat(gts)).sum().item() / len(torch.cat(text_preds))
    acc_pose = (torch.cat(pose_preds) == torch.cat(gts)).sum().item() / len(torch.cat(pose_preds))
    random_acc = 1/input_texts['input_ids'].shape[0]
    
    logger.info("Validation accuracy with text: {:.4f}".format(acc_text))
    tb_writer.add_scalar('acc_text/validation', acc_text, epoch)
    logger.info("Validation accuracy with pose: {:.4f}".format(acc_pose))
    tb_writer.add_scalar('acc_pose/validation', acc_pose, epoch)
    logger.info("Random val accuracy: {:.4f}".format(random_acc))
    logger.info("Validation loss: {:.4f}".format(val_loss / val_step))
    tb_writer.add_scalar('loss/validation', val_loss / val_step, epoch)
                
    return val_loss / val_step



tb_writer = SummaryWriter(log_dir='./tb_runs/roberta_2_sec_around' + '_' + str(datetime.datetime.now().strftime('%d %B %H:%M')))
model = model.cuda()
prev_loss = 1000000
global_step, global_loss, global_acc = 0,  0.0, 0.0
model.zero_grad()
log_softmax = nn.LogSoftmax(dim=-1)

    
for epoch in range(int(50)):
    
    gts = []
    pose_preds = []
    text_preds = []
    model.train()
    
    for step, batch in enumerate(train_loader):
        input_images, input_texts = batch

        input_images = input_images.cuda()
        input_texts['input_ids'] = input_texts['input_ids'].squeeze(1).cuda()
        input_texts['attention_mask'] = input_texts['attention_mask'].squeeze(1).cuda()
    
        logits_per_image, logits_per_text = model(input_images, input_texts)

        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss  = F.cross_entropy(logits_per_text, labels)

        pose_preds.append(log_softmax(logits_per_image).argmax(dim=1).cpu())
        text_preds.append(log_softmax(logits_per_text).argmax(dim=1).cpu())
        gts.append(labels.cpu())
            
        loss = (image_loss + text_loss) / 2
        loss.backward()
        global_loss += loss.item()

        if (step + 1) % 1 == 0:
            global_step += 1
            optimizer.step()

            # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

            if scheduler:
                scheduler.step()

            model.zero_grad()

            if global_step % 1000 == 0:
                logger.info("Epoch: {}, step: {}, lr: {:.6f}, {:.4f}".format(epoch, global_step, 
                    optimizer.param_groups[0]["lr"], global_loss / global_step))
                tb_writer.add_scalar('detailed train loss', global_loss / global_step, global_step)
                tb_writer.add_scalar('detailed learning rate', optimizer.param_groups[0]["lr"], global_step)
                
    acc_text = (torch.cat(text_preds) == torch.cat(gts)).sum().item() / len(torch.cat(text_preds))
    acc_pose = (torch.cat(pose_preds) == torch.cat(gts)).sum().item() / len(torch.cat(pose_preds))
    random_acc = 1/input_texts['input_ids'].shape[0] 
    
    logger.info("Train accuracy with text: {:.4f}".format(acc_text))
    tb_writer.add_scalar('acc_text/train', acc_text, epoch)
    logger.info("Train accuracy with pose: {:.4f}".format(acc_pose))
    tb_writer.add_scalar('acc_pose/train', acc_pose, epoch)
    logger.info("Random train accuracy: {:.4f}".format(random_acc))
    
    logger.info("Train loss: {:.4f}".format(global_loss / global_step))
    tb_writer.add_scalar('loss/train', global_loss / global_step, epoch)
    
    # validation starts here
    v_loss = validate(epoch, val_loader)
    if v_loss < prev_loss:
        # saving checkpoint
        save_checkpoint(epoch, global_step, model, optimizer)
        prev_loss = v_loss
        
tb_writer.close()