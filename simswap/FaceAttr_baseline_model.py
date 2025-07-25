from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import transforms, models

# you can add more models as you need.
__SUPPORT_MODEL__ = ["Resnet18", "Resnet101", "densenet121", "se_resnet101", "se_resnet50"]

"""
Adopt the pretrained resnet model to extract feature of the feature
"""
class FeatureExtraction(nn.Module):
    def __init__(self, pretrained, model_type = "Resnet18"):
        super(FeatureExtraction, self).__init__()
        if model_type == "Resnet18":
            self.model = models.resnet18(pretrained=pretrained)   
        if model_type == "Resnet101":
            self.model = models.resnet101(pretrained=pretrained)
        elif model_type == "Resnet152":
            self.model = models.resnet152(pretrained=pretrained)
        elif model_type == "Resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif model_type == "densenet121":
            self.model = models.densenet121(pretrained=pretrained)
        print("Has loaded the model {}".format(model_type))
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    def forward(self, image):
        return self.model(image)

"""
judge the attributes from the result of feature extraction
"""
class FeatureClassfier(nn.Module):
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfier, self).__init__()

        self.attrs_num = len(selected_attrs)
        self.selected_attrs = selected_attrs
        output_dim = len(selected_attrs)
        """build full connect layers for every attribute"""
        self.fc_set = {}

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        # res = self.fc(x)
        # res = self.sigmoid(res)
        return x


"""
conbime the extraction and classfier
"""
class FaceAttrModel(nn.Module):
    def __init__(self, model_type, pretrained, selected_attrs):
        super(FaceAttrModel, self).__init__()
        assert model_type in __SUPPORT_MODEL__
        self.featureExtractor = FeatureExtraction(pretrained, model_type)
        if model_type == "Resnet18":
            self.featureClassfier = FeatureClassfier(selected_attrs, input_dim=512)
        else:
            self.featureClassfier = FeatureClassfier(selected_attrs, input_dim=2048)
    
    def forward(self, image):
        features = self.featureExtractor(image)
        # results = self.featureClassfier(features)
        return features.view(image.size(0), -1)



