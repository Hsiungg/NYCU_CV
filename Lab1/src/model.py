import torch.nn as nn
import timm


class ResNetForClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True)
        self.model.reset_classifier(num_classes)

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
        return {"loss": loss, "logits": outputs}
