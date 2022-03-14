from model import Net
from dataset import get_dataset_nih, renew_train_nih
from train_an_epoch import training_step
from valid_an_epoch import eval_step

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

import os
import random
import numpy as np

if __name__ == "__main__":
    # fix seed
    # seeds
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    BATCH = 4
    nih_nodule_dataset = 'dataset/{}/nodule/label.csv'
    nih_normal_dataset = 'dataset/{}/normal/label.csv'
    
    # init model
    model = Net()
    if torch.cuda.is_available():
        print('gpu detected, using gpu')
        model.cuda()
    
    print('model initialized, ResNet 18')
    
    # init dataloader
    train_loader, valid_loader, test_loader = get_dataset_nih(nih_nodule_dataset, nih_normal_dataset, batch=BATCH)
    print('dataloader initialized')
    
    # training settings
    loss = torch.nn.CrossEntropyLoss()
    LR = 1e-3
    warm_up_epoches = 10
    momentum=0.9
    EPOCH = 30
    
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum)
    
    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = CosineAnnealingLR(optimizer, EPOCH)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warm_up_epoches, after_scheduler=scheduler_steplr)
    
    # track training
    training_reports = {
        'tr_loss' : [],
        'val_loss' : [],
        'tr_auc' : [],
        'val_auc' : [],
        'tr_pr' : [],
        'val_pr' : [],
        'best_epoch' : -1,
        'max_val_pr' : 0
    }
    
    # where to save model
    save_model_path = "model_weights"
    
    # print training settings
    print('===========================================')
    print('training settings:')
    print('optimizer: SGD, momentum: {}'.format(momentum))
    print("LR: {}\nwarm_up_epoches: {}\nEpoch: {}".format(LR, warm_up_epoches, EPOCH))
    print("LR scheduler: CosineAnnealingLR")
    print('===========================================')
    
    ## train model for N epochs
    for epoch in range(1, EPOCH+1):
        # Train
        train_loss = training_step(epoch, model, loss, optimizer, scheduler_warmup, train_loader)

        # evaluate
        val_loss, tr_auc, val_auc, tr_pr, val_pr = eval_step(model, loss, train_loader, valid_loader)

        # print training status of this epoch
        print("Epoch:", epoch, "Training Loss: {}, training auc: {}, Valid auc: {}".format(train_loss, tr_auc, val_auc))

        # save model 
        if val_pr > training_reports['max_val_pr']:
            training_reports['max_val_pr'] = val_pr
            training_reports['best_epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best_model.pth'))
            print('Model saved!')

        # save last model
        torch.save(model.state_dict(), os.path.join(save_model_path, 'last_model.pth'))

        # renew trainset
        train_loader = renew_train_nih(nih_nodule_dataset, nih_normal_dataset, batch=BATCH)
        
    print('training finished')
