import timm
import torch


class DogClassifier(torch.nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super(DogClassifier, self).__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
