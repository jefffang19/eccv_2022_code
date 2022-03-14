import torch
import sklearn
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def get_ys_pred(model, dataloader):
    '''
    model: pytorch network
    dataloader: pytorch dataloader
    '''
    
    # eval classification
    ys = np.array([])
    prds = np.array([])
    for x, y, _, _ in tqdm(dataloader):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        # fit model
        _, pred_cam, whole_predict = model(x, 1)
        
        # softmax
        whole_predict = whole_predict.softmax(dim=1)

        # eval with model
        ys = np.concatenate((ys, y.cpu().numpy())).astype(np.int32)
        # confidence of predict nodule class
        prds = np.concatenate((prds, whole_predict.cpu().numpy()[:, 1] )).astype(np.float32)
            
    return ys, prds


def get_report(model, dataloader):
    ys, prds = get_ys_pred(model, dataloader)
    
    _auc = sklearn.metrics.roc_auc_score(ys, prds, multi_class='ovr')
    _pr = sklearn.metrics.average_precision_score(ys, prds)

    return _auc, _pr

@torch.no_grad()
def valid_loss(model, loss_func, dataloader):
    valid_loss = []
    
    for x, y, _, _ in tqdm(dataloader): 
        if torch.cuda.is_available():
            x = x.cuda(); y = y.cuda()

        
        ## 1. forward propagationreal_labels
        ## -----------------------
        _, _, whole_predict = model(x, 1)
        ## 2. loss calculation
        ## -----------------------
        
        _loss = loss_func(whole_predict, y)
        
        valid_loss.append(_loss.item())
        
    return np.mean(valid_loss)
