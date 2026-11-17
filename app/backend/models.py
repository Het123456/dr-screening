import io
from abc import ABC, abstractmethod
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from backend.settings import ImageLoadError, ModelFileError

LESION_NAMES = [
    "Microaneurysm",
    "Hard exudate",
    "Intraretinal hemorrhage",
    "VB / IRMA",
    "Neovascularization",
    "Vitreous hemorrhage",
    "Retinal detachment",
]

GRADE_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

REFERABLE_GRADE = 2

JPEG_QUALITY = 95
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

FAKE_GRID_SIDE = 14

NUM_LESIONS = 7
NUM_GRADES = 5
FEATURE_SIZE = 1024
PATCH_SIZE = 16


class Prediction:
    def __init__(self, grade_probs, lesion_scores):
        self.grade_probs = grade_probs
        self.lesion_scores = lesion_scores
        self.grade = self.find_most_likely_grade()

    def find_most_likely_grade(self):
        best_grade = 0
        for grade in range(len(self.grade_probs)):
            if self.grade_probs[grade] > self.grade_probs[best_grade]:
                best_grade = grade
        return best_grade

    def grade_name(self):
        return GRADE_NAMES[self.grade]

    def referral_score(self):
        total = 0.0
        for grade in range(REFERABLE_GRADE, len(self.grade_probs)):
            total = total + self.grade_probs[grade]
        return total


class ImagePreparer:
    def __init__(self, image_size):
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(NORM_MEAN, NORM_STD)

    def open_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
        except (OSError, ValueError):
            raise ImageLoadError("This file could not be opened as an image:\n" + str(image_path))
        return image

    def small_size(self, image):
        width, height = image.size
        shorter_side = min(width, height)
        scale = self.image_size / shorter_side
        new_width = round(width * scale)
        new_height = round(height * scale)
        return (new_width, new_height)

    def open_small_image(self, image_path):
        try:
            image = Image.open(image_path)
            new_size = self.small_size(image)
            image.draft("RGB", new_size)
            image = image.convert("RGB")
            image = image.resize(new_size, Image.LANCZOS)
        except (OSError, ValueError):
            raise ImageLoadError("This file could not be opened as an image:\n" + str(image_path))
        return self.save_and_reopen_as_jpeg(image)

    def save_and_reopen_as_jpeg(self, image):
        memory_file = io.BytesIO()
        image.save(memory_file, "JPEG", quality=JPEG_QUALITY)
        jpeg_image = Image.open(memory_file)
        return jpeg_image.convert("RGB")

    def make_tensor(self, small_image):
        square_image = small_image.resize((self.image_size, self.image_size))
        tensor = self.to_tensor(square_image)
        tensor = self.normalize(tensor)
        return tensor.unsqueeze(0)


class BaseRetinalModel(ABC):

    @abstractmethod
    def predict(self, small_image):
        pass

    @abstractmethod
    def lesion_maps(self, small_image):
        pass


class FakeModel(BaseRetinalModel):
    def predict(self, small_image):
        grade_probs = [0.10, 0.15, 0.55, 0.15, 0.05]
        lesion_scores = [0.91, 0.78, 0.40, 0.12, 0.05, 0.03, 0.01]
        return Prediction(grade_probs, lesion_scores)

    def lesion_maps(self, small_image):
        grid = []
        for row_number in range(FAKE_GRID_SIDE):
            row = []
            for column_number in range(FAKE_GRID_SIDE):
                distance = abs(row_number - 5) + abs(column_number - 9)
                row.append(max(0, 6 - distance))
            grid.append(row)
        maps = []
        for lesion_number in range(len(LESION_NAMES)):
            maps.append(grid)
        return maps


class RETFoundNetwork(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="avg", img_size=image_size
        )
        self.lesion_head = nn.Linear(FEATURE_SIZE, NUM_LESIONS)
        self.grade_head = nn.Linear(FEATURE_SIZE, NUM_GRADES)

    def forward(self, images):
        features = self.backbone(images)
        lesion_output = self.lesion_head(features)
        grade_output = self.grade_head(features)
        return lesion_output, grade_output

    def patch_scores(self, images):
        tokens = self.backbone.forward_features(images)
        patch_tokens = tokens[0, 1:]
        patch_tokens = self.backbone.fc_norm(patch_tokens)
        return self.lesion_head(patch_tokens)


class RETFoundModel(BaseRetinalModel):
    def __init__(self, checkpoint_path, use_flip, image_size):
        self.checkpoint_path = Path(checkpoint_path)
        self.use_flip = use_flip
        self.image_size = image_size
        self.device = self.pick_device()
        self.preparer = ImagePreparer(image_size)
        self.network = None

    def pick_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self):
        if not self.checkpoint_path.exists():
            raise ModelFileError("The model file is missing:\n" + str(self.checkpoint_path))
        network = RETFoundNetwork(self.image_size)
        try:
            weights = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            network.load_state_dict(weights)
        except Exception:
            raise ModelFileError("The model file could not be read:\n" + str(self.checkpoint_path))
        network = network.to(self.device)
        network.eval()
        self.network = network

    def run_network(self, tensor):
        with torch.no_grad():
            lesion_output, grade_output = self.network(tensor)
        lesion_scores = torch.sigmoid(lesion_output)[0]
        grade_probs = torch.softmax(grade_output, dim=1)[0]
        return lesion_scores, grade_probs

    def predict(self, small_image):
        if self.network is None:
            self.load()
        tensor = self.preparer.make_tensor(small_image).to(self.device)
        lesion_scores, grade_probs = self.run_network(tensor)

        if self.use_flip:
            mirror_tensor = torch.flip(tensor, dims=[3])
            mirror_lesions, mirror_grades = self.run_network(mirror_tensor)
            lesion_scores = (lesion_scores + mirror_lesions) / 2
            grade_probs = (grade_probs + mirror_grades) / 2

        return Prediction(grade_probs.tolist(), lesion_scores.tolist())

    def lesion_maps(self, small_image):
        if self.network is None:
            self.load()
        tensor = self.preparer.make_tensor(small_image).to(self.device)
        with torch.no_grad():
            scores = self.network.patch_scores(tensor)
        scores = scores.tolist()

        side = self.image_size // PATCH_SIZE
        maps = []
        for lesion_number in range(NUM_LESIONS):
            grid = []
            for row_number in range(side):
                row = []
                for column_number in range(side):
                    patch_number = row_number * side + column_number
                    row.append(scores[patch_number][lesion_number])
                grid.append(row)
            maps.append(grid)
        return maps
