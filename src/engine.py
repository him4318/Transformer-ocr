# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:15:54 2020

@author: himanshu.chaudhary
"""
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import time


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



def train(model, criterion, optimizer, scheduler, dataloader, vocab_length=100, device):

    model.train()
    total_loss = 0
    for batch, (imgs, labels_y,) in enumerate(dataloader):
          imgs = imgs.to(device)
          labels_y = labels_y.to(device)
    
          optimizer.zero_grad()
          output = model(imgs.float(),labels_y.long()[:,:-1])

          norm = (labels_y != 0).sum()
          loss = criterion(output.log_softmax(-1).contiguous().view(-1, vocab_length), labels_y[:,1:].contiguous().view(-1).long()) / norm

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
          optimizer.step()
          total_loss += (loss.item()*norm)

    return total_loss / len(dataloader),output

def evaluate(model, criterion, dataloader, vocab_length=100, device):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
      for batch, (imgs, labels_y,) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels_y = labels_y.to(device)

            output = model(imgs.float(),labels_y.long()[:,:-1])
              
            norm = (labels_y != 0).sum()
            loss = criterion(output.log_softmax(-1).contiguous().view(-1, vocab_length), labels_y[:,1:].contiguous().view(-1).long()) / norm
  
            epoch_loss += (loss.item()*norm)

    return epoch_loss / len(dataloader)

def get_memory(model, imgs):
    x = model.conv(model.get_feature(imgs))
    bs,_,H, W = x.shape
    pos = torch.cat([
            model.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            model.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

    return model.transformer.encoder(pos +  0.1 * x.flatten(2).permute(2, 0, 1))
    

def single_image_inference(model, img, tokenizer,device):
    '''
    Run inference on single image
    '''
    
    imgs = img.unsqueeze(0).float().to(device)
    with torch.no_grad():    
      memory = get_memory(model,imgs)
      out_indexes = [tokenizer.chars.index('SOS'), ]
      
      for i in range(128):
            mask = model.generate_square_subsequent_mask(i+1).to(device)
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
            output = model.vocab(model.transformer.decoder(model.query_pos(model.decoder(trg_tensor)), memory,tgt_mask=mask))
            out_token = output.argmax(2)[-1].item()
            if out_token == tokenizer.chars.index('EOS'):
                break
            
            out_indexes.append(out_token)

    pre = tokenizer.decode(out_token)
    return pre

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, target_path, device):
    '''
    run one epoch for a model
    '''
    best_valid_loss = np.inf
    c = 0
    for epoch in range(epochs):     
        print(f'Epoch: {epoch+1:02}','learning rate{}'.format(scheduler.get_last_lr()))
     
        start_time = time.time()
    
        train_loss,outputs = train(model,  criterion, optimizer, scheduler, train_loader, device)
        valid_loss = evaluate(model, criterion, val_loader, device)
     
        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        c+=1
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), target_path)
            c=0
     
        if c>4:
            scheduler.step()
            c=0
     
        print(f'Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val   Loss: {valid_loss:.3f}')    
    
    print(best_valid_loss)