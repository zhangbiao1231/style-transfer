import torch
import torch.nn as nn

def Gram_matrix(X):
    b, c, h, w = X.size() #（3，64，224，224）
    features = X.view(b,c,h*w) # (3, 64, 224*244)
    G =torch.bmm(features,features.transpose(1,2)) # 批次矩阵乘法 (3, 64, 64)
    return G.div(c * h * w) # 归一化 # divide by (64*224*224)

class Contentloss(nn.Module):
    def __init__(self,target,weight=1):
        super(Contentloss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.criterion = nn.MSELoss()
    def forward(self,X):
        return self.criterion(X,self.target)* self.weight

class Styleloss(nn.Module):
    def __init__(self,target_feature,weight=1e3):
        super(Styleloss, self).__init__()
        self.target = Gram_matrix(target_feature).detach()
        self.weight = weight
        self.criterion = nn.MSELoss()
    def forward(self,X):
        G = Gram_matrix(X)
        return self.criterion(G,self.target)* self.weight

class TotalVariationLoss(nn.Module):
    def __init__(self,weight=1):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight
    def forward(self,X):
        h_variation = torch.mean(torch.abs(X[:,:,:-1,:] - X[:,:,1:,:]))
        w_variation = torch.mean(torch.abs(X[:,:,:,:-1] - X[:,:,:,1:]))
        loss = (h_variation+w_variation)* self.weight
        return loss
