"""
Import packages
"""
from torch import nn
import timm
from timm.loss.cross_entropy import SoftTargetCrossEntropy


class ResNetForClassification(nn.Module):
    """Define the model architecture with Timm and nn."""

    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_r50_s16_384.orig_in21k_ft_in1k',
            pretrained=True,
            num_classes=num_classes,
        )
        '''
        self.model = timm.create_model('resnet50.a1_in1k',
                                       pretrained=True,
                                       num_classes=num_classes,)
        '''
        # change resolution into 384 x 384

    def forward(self, pixel_values, labels=None):
        """Forwrd process for model with loss function"""
        outputs = self.model(pixel_values)
        loss = None
        if labels is not None:
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = SoftTargetCrossEntropy()
            loss = loss_fn(outputs, labels)
        return {"loss": loss, "logits": outputs}
