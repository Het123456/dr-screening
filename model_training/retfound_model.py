import argparse

import timm
import torch
import torch.nn as nn
from torchvision import transforms

from base_model import BaseRetinalModel, Prediction
from dataset import NORM_MEAN, NORM_STD

NUM_LESIONS = 7
NUM_GRADES = 5
FEATURE_SIZE = 1024


class RETFoundNetwork(nn.Module):
    def __init__(self, weights_path=None, drop_path=0.0):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="avg",
            drop_path_rate=drop_path,
        )
        if weights_path is not None:
            self.load_retfound_weights(weights_path)

        self.lesion_head = nn.Linear(FEATURE_SIZE, NUM_LESIONS)
        self.grade_head = nn.Linear(FEATURE_SIZE, NUM_GRADES)

    def load_retfound_weights(self, weights_path):
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        result = self.backbone.load_state_dict(state, strict=False)
        print("RETFound weights loaded. Missing keys:", result.missing_keys)
        print("Unexpected keys:", result.unexpected_keys)

    def forward(self, images):
        features = self.backbone(images)
        return self.lesion_head(features), self.grade_head(features)


class RETFoundModel(BaseRetinalModel):
    def __init__(self, weights_path=None, device="cpu"):
        self.device = device
        self.network = RETFoundNetwork(weights_path).to(device)
        self.network.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])

    def predict(self, image):
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lesion_logits, grade_logits = self.network(tensor)

        lesion_scores = torch.sigmoid(lesion_logits)[0]
        grade = int(torch.argmax(grade_logits, dim=1).item())
        lesion_present = [1 if score > 0.5 else 0 for score in lesion_scores]
        return Prediction(grade, lesion_present, [round(s, 3) for s in lesion_scores.tolist()])


if __name__ == "__main__":
    from dataset import MMRDRDataset

    model = RETFoundModel(weights_path=None)
    data = MMRDRDataset("cfp", "train")
    image = data.load_image(0)

    result = model.predict(image)
    print(result)

    network = model.network
    total = sum(p.numel() for p in network.parameters())
    print("number of parameters:", f"{total:,}")
    lesion_logits, grade_logits = network(torch.zeros(2, 3, 224, 224))
    print("batch of 2 -> lesion output:", lesion_logits.shape, "| grade output:", grade_logits.shape)
