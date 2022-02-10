import torch
import torch.nn as nn
from torch.nn import Module as Module
import timm
from torch.nn import Parameter
import torch.nn.functional as F

class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # create proxy weights
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        # cos_theta = F.cosine_similarity(x, self.weight.reshape(1, 20, -1), dim=2)
        cos_theta = F.normalize(x.view(x.shape[0], -1), dim=1).mm(F.normalize(self.weight, p=2, dim=0))
        return cos_theta.clamp(-1, 1)

    def get_centers(self):
        return torch.t(self.weight)

class TimmModelsWrapper(Module):
    def __init__(self,
                 model_name,
                 num_classes,
                 pretrained=False,
                 ml_decoder_used=False,
                 asl_use = False):

        super().__init__()
        self.pretrained = pretrained
        self.is_mobilenet = True if model_name in ["mobilenetv3_large_100_miil_in21k", "mobilenetv3_large_100_miil"] else False
        self.num_classes = num_classes

        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       num_classes=self.num_classes)
        self.num_head_features = self.model.num_features
        self.num_features = (self.model.conv_head.in_channels if self.is_mobilenet
                             else self.model.num_features)
        if ml_decoder_used:
            self.model.global_pool = torch.nn.Identity()
            self.model.classifier = torch.nn.Identity()
        else:
            if asl_use:
                self.model.act2 = nn.PReLU()
                self.model.classifier = AngleSimpleLinear(self.num_head_features, self.num_classes)
            else:
                self.model.classifier = self.model.get_classifier()

    def forward(self, x):
        return self.model(x)
