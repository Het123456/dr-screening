from PIL import Image

from backend.models import GRADE_NAMES, ImagePreparer, LESION_NAMES
from backend.settings import UnknownModalityError

PICTURE_SIZE = 760
HIGHLIGHT_COLOUR = (0, 229, 255)
STRONGEST = 200
SHOWN_LABELS = ["Likely", "Possible"]


class HeatmapGenerator:
    def find_smallest(self, grid):
        smallest = grid[0][0]
        for row in grid:
            for value in row:
                if value < smallest:
                    smallest = value
        return smallest

    def find_largest(self, grid):
        largest = grid[0][0]
        for row in grid:
            for value in row:
                if value > largest:
                    largest = value
        return largest

    def strengths(self, grid):
        smallest = self.find_smallest(grid)
        largest = self.find_largest(grid)
        spread = largest - smallest
        if spread == 0:
            spread = 1
        numbers = []
        for row in grid:
            for value in row:
                share = (value - smallest) / spread
                share = share * share
                numbers.append(round(share * STRONGEST))
        return numbers

    def make_mask(self, grid, picture_size):
        rows = len(grid)
        columns = len(grid[0])
        small_mask = Image.new("L", (columns, rows))
        small_mask.putdata(self.strengths(grid))
        return small_mask.resize(picture_size, Image.BICUBIC)

    def shrink(self, image):
        picture = image.copy()
        picture.thumbnail((PICTURE_SIZE, PICTURE_SIZE))
        return picture

    def make_overlay(self, picture, grid):
        mask = self.make_mask(grid, picture.size)
        colour_layer = Image.new("RGB", picture.size, HIGHLIGHT_COLOUR)
        return Image.composite(colour_layer, picture, mask)

    def make_all(self, image, grids, lesion_labels):
        picture = self.shrink(image)
        overlays = []
        for lesion_number in range(len(grids)):
            if lesion_labels[lesion_number] in SHOWN_LABELS:
                overlays.append(self.make_overlay(picture, grids[lesion_number]))
            else:
                overlays.append(None)
        return overlays


class AnalysisResult:
    def __init__(self, image_path, modality, image, prediction, decision, lesion_scores, lesion_labels, heatmaps):
        self.image_path = image_path
        self.modality = modality
        self.image = image
        self.prediction = prediction
        self.decision = decision
        self.lesion_scores = lesion_scores
        self.lesion_labels = lesion_labels
        self.heatmaps = heatmaps

    def grade_text(self):
        return "Grade " + str(self.prediction.grade) + ": " + self.prediction.grade_name()

    def lesion_rows(self):
        rows = []
        for index in range(len(LESION_NAMES)):
            rows.append([LESION_NAMES[index], self.lesion_scores[index], self.lesion_labels[index]])
        return rows

    def grade_rows(self):
        rows = []
        for index in range(len(GRADE_NAMES)):
            name = str(index) + " " + GRADE_NAMES[index]
            probability = self.prediction.grade_probs[index]
            percent_text = str(round(probability * 100)) + "%"
            rows.append([name, probability, percent_text])
        return rows


class Analyzer:
    def __init__(self):
        self.models = {}
        self.policies = {}
        self.lesion_policies = {}
        self.preparers = {}
        self.heatmap_generator = HeatmapGenerator()

    def add_model(self, modality, model, policy, lesion_policy, image_size):
        self.models[modality] = model
        self.policies[modality] = policy
        self.lesion_policies[modality] = lesion_policy
        self.preparers[modality] = ImagePreparer(image_size)

    def analyze(self, image_path, modality):
        if modality not in self.models:
            raise UnknownModalityError("There is no model for this image type: " + str(modality))

        preparer = self.preparers[modality]
        image = preparer.open_image(image_path)
        small_image = preparer.open_small_image(image_path)
        prediction = self.models[modality].predict(small_image)
        decision = self.policies[modality].decide(prediction)
        lesion_policy = self.lesion_policies[modality]
        lesion_scores = lesion_policy.calibrate_all(prediction.lesion_scores)
        lesion_labels = lesion_policy.label_all(lesion_scores)
        grids = self.models[modality].lesion_maps(small_image)
        heatmaps = self.heatmap_generator.make_all(image, grids, lesion_labels)
        return AnalysisResult(image_path, modality, image, prediction, decision, lesion_scores, lesion_labels, heatmaps)
