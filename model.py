import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import ConcatDataset
from torchvision import transforms
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, load_model_path=None):
        super(Net, self).__init__()
        
        # backbone
        model = torchvision.models.resnet18(pretrained=True)
        self.feature_extract = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        
        self.avgpool = model.avgpool
        # classifiers
        self.wholeimg_fc = nn.Linear(in_features=512, out_features=2, bias=True)
        self.patchwise_fc = nn.Linear(in_features=512, out_features=2, bias=False)
        self.patch_aggr_fc = nn.Linear(in_features=512, out_features=2, bias=True)
        
    def forward(self, x, cls):
        
        batch_size = x.shape[0]
        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])
        
        # The incoming patches are always resized to 300x300
        input_size = (300, 300)
        
        # backbone
        features = self.feature_extract(x)
        x = self.avgpool(features).view(features.shape[0], features.shape[1])
        
        # reshape back to (b, 17, ...)
        x = x.view(batch_size, 17, 512)
        # whole image features
        whole_x = x[:, 0, :]
        # patch features
        patch_x = x[:, 1:17, :]
        
        # whole image logits
        whole_logits = self.wholeimg_fc(whole_x)
        
        # patch predictions
        # now we weighted sum patches features to form image level features
        patch_logits_collect = self.patchwise_fc(patch_x)
        weights = F.softmax(patch_logits_collect[:, :, 1], dim=1).unsqueeze(-1)
        patch_features = (patch_x * weights).sum(dim=1)
        patch_logists = self.patch_aggr_fc(patch_features)

        # combine whole image and patch predictions
        whole_predict = whole_logits + patch_logists
        
        # CAM
        cam_map = torch.sum(features * self.patchwise_fc.weight[0].view(512, 1, 1), dim=1)
        cam_map = F.interpolate(cam_map.unsqueeze(1), size=input_size, mode='bilinear', align_corners=False).squeeze(1)
        # reshape back to (b, 17, ...)
        cam_map = cam_map.view(batch_size, 17, *cam_map.shape[1:])
        
        return patch_logits_collect, cam_map, whole_predict