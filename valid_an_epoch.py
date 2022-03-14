import torch

from validation_utils import get_report, valid_loss

def eval_step(model, loss_func, train_loader, valid_loader):
    model.eval()
    
    print('valid set loss calculation')
    val_loss = valid_loss(model, loss_func, valid_loader)
    
    torch.cuda.empty_cache()
    print('training set metrics calculation')
    train_auc_score, train_pr_score = get_report(model, train_loader)
    print('valid set metrics calculation')
    valid_auc_score, valid_pr_score = get_report(model, valid_loader)
        
    return val_loss, train_auc_score, valid_auc_score, train_pr_score, valid_pr_score


def test_step(model, valid_loader):
    model.eval()
    
    torch.cuda.empty_cache()
    print('valid set metrics calculation')
    valid_auc_score, valid_pr_score = get_report(model, valid_loader)
        
    return valid_auc_score, valid_pr_score
