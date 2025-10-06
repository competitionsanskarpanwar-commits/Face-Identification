#src/model.py
import torch
import torch.nn as nn
import math
from torch.nn.functional import one_hot
from torch.nn.modules import normalization
from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=512, backbone="resnet34", pretrained=False):
        super().__init__()
        if backbone == "resnet34":
            if pretrained:
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                self.backbone = models.resnet34(weights=None)

            in_features = self.backbone.fc.in_features
            if isinstance(in_features, tuple):   # 👈 safeguard
                in_features = in_features[0]
            
            # Remove the original fc layer and replace with identity
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Backbone not implemented")

        self.embedding = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)


    def forward(self, x):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = self.bn(emb)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace margin (additive angular margin)
    Produce logits for cross-entropy.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, label):
        # input: normalized embeddings [B, in_features]
        # weight: [num_classes, out_features] -> normalize rows
        normalized_weight = nn.functional.normalize(self.weight, dim=1)
        cosine = nn.functional.linear(input, normalized_weight) # [B, num_classes]
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # one-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1,1), 1.0)
        # apply margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= float(self.s)
        return output