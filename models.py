import torchvision.models as models
import torch.nn as nn

print(models.list_models())
# print(models.mvit_v1_base())
class CustomCNN(nn.Module):
    def __init__(self, num_classes, network=None):
        super(CustomCNN, self).__init__()
        if network == 'resnet18':
            self.features = models.resnet18(pretrained=True)
            self.features.fc = nn.Linear(512, num_classes)
        elif network == 'resnet34':
            self.features = models.resnet34(pretrained=True)
            self.features.fc = nn.Linear(512, num_classes)

        elif network == 'resnet50':
            self.features = models.resnet50(pretrained=True)
            self.features.fc = nn.Linear(2048, num_classes)

        elif network == 'resnet101':
            self.features = models.resnet101(pretrained=True)
            self.features.fc = nn.Linear(2048, num_classes)
        elif network == 'maxvit_t':
            self.features = models.maxvit_t(weights='DEFAULT')
            self.features.classifier[5] = nn.Linear(512, num_classes)
        elif network == 'vit_h_14':
            self.features = models.vit_h_14(weights='ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1')
            self.features.heads = nn.Linear(1280, num_classes)
        elif network == 'vit_l_16':
            self.features = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
            self.features.heads = nn.Linear(1024, num_classes)
        elif network == 'darknet':
            self.features = models.darknet(pretrained=True)
        elif network == 'convnext_large':
            self.features = models.convnext_large(pretrained=True)
            self.features.classifier[2] = nn.Linear(1536, num_classes)
        elif network == 'swin_b':
            self.features = models.swin_b(pretrained=True)
            self.features.head = nn.Linear(1024, num_classes)
        elif network == 'swin_v2_b':
            self.features = models.swin_v2_b(pretrained=True)
            self.features.head = nn.Linear(1024, num_classes)
        elif network == 'swin_s':
            self.features = models.swin_s(pretrained=True)
            self.features.head = nn.Linear(768, num_classes)
        elif network == 'swin_v2_s':
            self.features = models.swin_v2_s(pretrained=True)
            self.features.head = nn.Linear(768, num_classes)
        elif network == 'swin_t':
            self.features = models.swin_t(pretrained=True)
            self.features.head = nn.Linear(768, num_classes)
        elif network == 'swin_v2_t':
            self.features = models.swin_v2_t(pretrained=True)
            self.features.head = nn.Linear(768, num_classes)
        elif network == 'convnext_base':
            self.features = models.convnext_base(pretrained=True)
            self.features.classifier[2] = nn.Linear(1024, num_classes)
        elif network == 'convnext_small':
            self.features = models.convnext_small(pretrained=True)
            self.features.classifier[2] = nn.Linear(768, num_classes)
        elif network == 'convnext_tiny':
            self.features = models.convnext_tiny(pretrained=True)
            self.features.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x