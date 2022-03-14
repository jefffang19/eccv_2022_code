from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from model import Net
from dataset import get_dataset_nih, renew_train_nih
from valid_an_epoch import test_step
import torch
import os

if __name__ == "__main__":
    nih_nodule_dataset = 'dataset/{}/nodule/label.csv'
    nih_normal_dataset = 'dataset/{}/normal/label.csv'
    
    # where to load model
    save_model_path = "model_weights"
    
    # init model
    model = Net()
    if torch.cuda.is_available():
        print('gpu detected, using gpu')
        model.cuda()
        
     # load model 
    model.load_state_dict(torch.load(os.path.join(save_model_path, 'weights_used_in_paper.pth'), map_location='cuda'))
    print('weights successfully loaded')
    
    # init dataloader
    train_loader, valid_loader, test_loader = get_dataset_nih(nih_nodule_dataset, nih_normal_dataset, batch=4)
    
    test_auc_score, test_pr_score = test_step(model, test_loader)
    
    print("AUC score on NIH test set : {}".format(test_auc_score))