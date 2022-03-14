from tqdm import tqdm
import numpy as np
import torch

def training_step(epoch, model, loss_func, optimizer, scheduler_warmup, train_loader):
    '''
    input:
    epoch: int
    model: pytorch model
    loss_func: pytorch loss function => cross entropy
    optimizer: pytorch optimizer
    scheduler_warmup: lr scheduler
    train_loader: training set dataloader
    
    return mean of all batch training loss
    type: float
    '''
    # step lr scheduler
    scheduler_warmup.step()
    
    model.train()
    train_loss = []
    
    print('training set loss calculation')
    
    torch.cuda.empty_cache()
    for x, y, _, _ in tqdm(train_loader): 
        if torch.cuda.is_available():
            x = x.cuda(); y = y.cuda()

        optimizer.zero_grad()
        _, _, whole_predict = model(x, 1) # model forward
        _loss = loss_func(whole_predict, y) # loss calculation
        _loss.backward()
        optimizer.step()
        train_loss.append(_loss.item()) # track loss
        
    return np.mean(train_loss)