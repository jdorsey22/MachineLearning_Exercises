import torchvision.models as models
import torch.nn as nn
resnet18 = models.resnet18()
# model = models.densenet121(pretrained=True)

print(resnet18)
# fc = nn.Sequential(
#     nn.Linear(1024, 460),
#     nn.ReLU(),
#     nn.Dropout(0.4),
#
#     nn.Linear(460, 2),
#     nn.LogSoftmax(dim=1)
#
# )
# model.classifier = fc


print(resnet18)