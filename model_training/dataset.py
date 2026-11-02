import csv
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "MMRDR"

# Standard ImageNet values.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class Sample:
    def __init__(self, image_path, grade, lesions):
        self.image_path = image_path
        self.grade = grade
        self.lesions = lesions

    def __repr__(self):
        return f"Sample({self.image_path.name}, grade={self.grade}, lesions={self.lesions})"


class MMRDRDataset:
    def __init__(self, modality, split):
        # modality is either 'cfp' or 'uwf' and split is either 'train' or 'test'
        if modality == 'cfp':
            self.folder = DATA_ROOT / "MMRDR-CFP"
            csv_name = 'FP.csv'
        elif modality == 'uwf':
            self.folder = DATA_ROOT / "MMRDR-UWF"
            csv_name = 'UWF.csv'

        if split == 'train':
            prefix = 'tr'
        elif split == 'test':
            prefix = 'ts'

        self.samples = []

        with open(self.folder / csv_name, newline="") as f:
            for row in csv.DictReader(f):
                file_name = Path(row["image"]).name

                if file_name.startswith(prefix):
                    image_path = self.folder / row["image"]
                    grade = int(row["grade"])
                    lesions = json.loads(row["lesion"])
                    self.samples.append(Sample(image_path, grade, lesions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def load_image(self, index, image_size=224):
        # Open the picture, make sure it has 3 colour channels, and resize it to a square
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return image.resize((image_size, image_size))

    def get_item(self, index, image_size=224):
        # Everything the model needs for one image: image tensor, grade, and the 7 lesions
        image = self.load_image(index, image_size)
        to_tensor = transforms.Compose([
            transforms.ToTensor(),                           # picture -> tensor, values 0 to 1
            transforms.Normalize(NORM_MEAN, NORM_STD),       # rescale to what the model expects
        ])
        image_tensor = to_tensor(image)

        sample = self.samples[index]
        grade = torch.tensor(sample.grade)                              # a single whole number
        lesions = torch.tensor(sample.lesions, dtype=torch.float32)     # 7 values, 0.0 or 1.0
        return image_tensor, grade, lesions


if __name__ == "__main__":
    for modality in ("cfp", "uwf"):
        for split in ("train", "test"):
            data = MMRDRDataset(modality, split)
            print(modality, split, len(data))
    print(MMRDRDataset("cfp", "train")[0])

    # Check that images load and resize
    for modality in ("cfp", "uwf"):
        data = MMRDRDataset(modality, "train")
        original = Image.open(data[0].image_path)
        resized = data.load_image(0)
        print(modality, "original size:", original.size, "-> resized:", resized.size)

    data = MMRDRDataset("cfp", "train")
    index = next(i for i in range(len(data)) if sum(data[i].lesions) > 0)
    image_tensor, grade, lesions = data.get_item(index)
    print("image tensor shape:", image_tensor.shape)
    print("image value range:", round(image_tensor.min().item(), 2), "to", round(image_tensor.max().item(), 2))
    print("grade:", grade, "| lesions:", lesions, "| lesions shape:", lesions.shape)
