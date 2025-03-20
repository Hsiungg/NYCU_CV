import torch.nn as nn
import timm


class ResNetForClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.model = timm.create_model('resnetv2_34d.ra4_e3600_r224_in1k',
        self.model = timm.create_model('cspresnet50.ra_in1k',
                                       pretrained=True,
                                       num_classes=num_classes,)

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
        return {"loss": loss, "logits": outputs}
