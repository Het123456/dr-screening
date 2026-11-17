import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AppError(Exception):
    pass


class ImageLoadError(AppError):
    pass


class ModelFileError(AppError):
    pass


class UnknownModalityError(AppError):
    pass


class ModelSettings:
    def __init__(self, modality, checkpoint_folder, image_size, referral_file, lesion_file):
        self.modality = modality
        self.image_size = image_size
        self.checkpoint_path = PROJECT_ROOT / "checkpoints" / checkpoint_folder / "best.pt"
        self.referral_path = PROJECT_ROOT / "results" / referral_file
        self.lesion_path = PROJECT_ROOT / "results" / lesion_file
        self.cutoff = 0.5
        self.use_flip = False
        self.lesion_slopes = []
        self.lesion_intercepts = []
        self.read_referral_file()
        self.read_lesion_file()

    def read_referral_file(self):
        try:
            with open(self.referral_path) as file:
                values = json.load(file)
        except (OSError, ValueError):
            raise ModelFileError("The referral settings file could not be read:\n" + str(self.referral_path))
        self.cutoff = values["cutoff"]
        self.use_flip = values["use_flip"]

    def read_lesion_file(self):
        try:
            with open(self.lesion_path) as file:
                values = json.load(file)
        except (OSError, ValueError):
            raise ModelFileError("The lesion settings file could not be read:\n" + str(self.lesion_path))
        for lesion in values["lesions"]:
            self.lesion_slopes.append(lesion["slope"])
            self.lesion_intercepts.append(lesion["intercept"])
