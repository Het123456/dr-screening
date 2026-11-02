import csv
import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "MMRDR"


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


if __name__ == "__main__":
    for modality in ("cfp", "uwf"):
        for split in ("train", "test"):
            data = MMRDRDataset(modality, split)
            print(modality, split, len(data))
    print(MMRDRDataset("cfp", "train")[0])
