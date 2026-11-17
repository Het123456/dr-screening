import json
from datetime import datetime
from pathlib import Path

from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from backend.models import LESION_NAMES
from backend.settings import AppError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULT_FILES = [
    ["CFP model on CFP images", "cfp_512_model_on_cfp_test_flip.json"],
    ["UWF model on UWF images", "uwf_v2_model_on_uwf_test_flip.json"],
    ["CFP model on UWF images", "cfp_512_model_on_uwf_test_flip.json"],
]

rcParams["font.family"] = ["Arial", "DejaVu Sans"]

MODALITY_NAMES = {"cfp": "Colour fundus photo (CFP)", "uwf": "Ultra-widefield photo (UWF)"}
BADGE_TEXTS = {"routine": "NO REFERRAL", "refer": "REFER", "urgent": "URGENT REFERRAL"}
MODEL_NAME = "RETFound"

BLACK = "#000000"
GREY = "#777777"
LIGHT_GREY = "#eeeeee"
WHITE = "#ffffff"

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.06
RIGHT = 0.94
MAX_FILE_NAME_LENGTH = 30


class ModelResult:
    def __init__(self, title, lesion_aucs, mean_lesion_auc, kappa, accuracy, num_images):
        self.title = title
        self.lesion_aucs = lesion_aucs
        self.mean_lesion_auc = mean_lesion_auc
        self.kappa = kappa
        self.accuracy = accuracy
        self.num_images = num_images


class ResultsReader:
    def __init__(self):
        self.folder = PROJECT_ROOT / "results"
        self.lesion_names = LESION_NAMES

    def read_one(self, title, file_name):
        path = self.folder / file_name
        try:
            with open(path) as file:
                values = json.load(file)
        except (OSError, ValueError):
            raise AppError("The results file could not be read:\n" + str(path))

        lesion_aucs = []
        for auc in values["lesion_auc"].values():
            lesion_aucs.append(auc)

        return ModelResult(
            title,
            lesion_aucs,
            values["mean_lesion_auc"],
            values["grade_kappa"],
            values["grade_accuracy"],
            values["num_images"],
        )

    def read_all(self):
        results = []
        for title, file_name in RESULT_FILES:
            results.append(self.read_one(title, file_name))
        return results

    def finding_text(self, results):
        own = results[1]
        transfer = results[2]
        text = "Finding: the CFP model does not work on UWF images. "
        text = text + "Its mean lesion AUC falls to " + format(transfer.mean_lesion_auc, ".2f")
        text = text + " and its grade kappa to " + format(transfer.kappa, ".2f")
        text = text + ", while the model trained on UWF images reaches "
        text = text + format(own.mean_lesion_auc, ".2f") + " and " + format(own.kappa, ".2f")
        text = text + ". Each image type needs its own model."
        return text


class ReportBuilder:
    def save(self, result, output_path):
        figure = Figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.add_header(figure)
        self.add_photo_card(figure, result)
        self.add_grade_card(figure, result)
        self.add_referral_card(figure, result)
        self.add_lesion_card(figure, result)
        self.add_grade_probability_card(figure, result)
        self.add_details_card(figure, result)
        try:
            figure.savefig(output_path, format="pdf")
        except OSError:
            raise AppError("The report could not be saved to:\n" + str(output_path))

    def draw_box(self, figure, x, y, width, height, fill_colour):
        box = Rectangle((x, y), width, height, facecolor=fill_colour, edgecolor=BLACK, linewidth=0.8)
        box.set_transform(figure.transFigure)
        figure.add_artist(box)

    def draw_bar(self, figure, x, y, width, value, colour):
        height = 0.007
        background = Rectangle((x, y), width, height, facecolor=LIGHT_GREY, edgecolor="none")
        background.set_transform(figure.transFigure)
        figure.add_artist(background)
        if value > 0:
            filled = Rectangle((x, y), width * value, height, facecolor=colour, edgecolor="none")
            filled.set_transform(figure.transFigure)
            figure.add_artist(filled)

    def draw_heading(self, figure, x, y, text):
        figure.text(x, y, text.upper(), fontsize=8.5, weight="bold", color=BLACK)

    def shorten(self, text):
        if len(text) <= MAX_FILE_NAME_LENGTH:
            return text
        return text[:MAX_FILE_NAME_LENGTH - 3] + "..."

    def add_header(self, figure):
        today = datetime.now().strftime("%d %b %Y, %H:%M")
        self.draw_box(figure, LEFT, 0.905, RIGHT - LEFT, 0.055, BLACK)
        figure.text(0.08, 0.932, "Diabetic Retinopathy Screening Report",
                    fontsize=16, weight="bold", color=WHITE, va="center")
        figure.text(0.92, 0.932, today, fontsize=10, color=WHITE, va="center", ha="right")

    def add_photo_card(self, figure, result):
        self.draw_box(figure, LEFT, 0.60, 0.42, 0.29, "none")
        axes = figure.add_axes([0.075, 0.61, 0.39, 0.27])
        axes.imshow(result.image)
        axes.axis("off")

    def add_grade_card(self, figure, result):
        prediction = result.prediction
        self.draw_box(figure, 0.50, 0.755, 0.44, 0.135, "none")
        self.draw_heading(figure, 0.52, 0.868, "Grade")
        figure.text(0.52, 0.815, str(prediction.grade), fontsize=44, weight="bold", va="center")
        figure.text(0.60, 0.826, prediction.grade_name(), fontsize=17, weight="bold", va="center")
        figure.text(0.60, 0.795, "most likely of five grades", fontsize=9, color=GREY, va="center")

    def add_referral_card(self, figure, result):
        decision = result.decision
        bar_x = 0.62
        bar_width = 0.30
        cutoff_x = bar_x + bar_width * decision.cutoff

        self.draw_box(figure, 0.50, 0.60, 0.44, 0.14, "none")
        self.draw_heading(figure, 0.52, 0.718, "Referral")

        self.draw_box(figure, 0.52, 0.674, 0.40, 0.032, BLACK)
        figure.text(0.72, 0.690, BADGE_TEXTS[decision.level],
                    fontsize=13, weight="bold", color=WHITE, ha="center", va="center")
        figure.text(0.52, 0.657, decision.advice, fontsize=9.5, va="center")

        figure.text(0.52, 0.634, "Score " + format(decision.score, ".2f"), fontsize=9.5, va="center")
        self.draw_bar(figure, bar_x, 0.630, bar_width, decision.score, BLACK)
        cutoff_mark = Line2D([cutoff_x, cutoff_x], [0.624, 0.644], linewidth=1.2, color=GREY)
        figure.add_artist(cutoff_mark)
        figure.text(cutoff_x, 0.611, "cutoff " + format(decision.cutoff, ".2f"),
                    fontsize=7.5, color=GREY, ha="center")

    def add_lesion_card(self, figure, result):
        self.draw_box(figure, LEFT, 0.325, RIGHT - LEFT, 0.26, "none")
        self.draw_heading(figure, 0.08, 0.560, "Lesion findings")

        y = 0.528
        for name, score, label in result.lesion_rows():
            if label == "Likely":
                weight = "bold"
            else:
                weight = "normal"
            if label == "Unlikely":
                colour = GREY
            else:
                colour = BLACK

            figure.text(0.08, y, name, fontsize=10, weight=weight, color=colour, va="center")
            self.draw_bar(figure, 0.38, y - 0.0035, 0.41, score, colour)
            figure.text(0.92, y, label, fontsize=10, weight=weight, color=colour, va="center", ha="right")
            y = y - 0.029

    def add_grade_probability_card(self, figure, result):
        self.draw_box(figure, LEFT, 0.06, 0.50, 0.25, "none")
        self.draw_heading(figure, 0.08, 0.285, "Grade probabilities")

        y = 0.250
        grade_number = 0
        for name, probability, percent_text in result.grade_rows():
            if grade_number == result.prediction.grade:
                weight = "bold"
                colour = BLACK
            else:
                weight = "normal"
                colour = GREY

            figure.text(0.08, y, name, fontsize=9, weight=weight, va="center")
            self.draw_bar(figure, 0.27, y - 0.0035, 0.20, probability, colour)
            figure.text(0.54, y, percent_text, fontsize=9, weight=weight, va="center", ha="right")
            y = y - 0.036
            grade_number = grade_number + 1

    def add_details_card(self, figure, result):
        today = datetime.now().strftime("%d %b %Y, %H:%M")
        file_name = self.shorten(Path(result.image_path).name)
        rows = [
            ["Date", today],
            ["File", file_name],
            ["Type", MODALITY_NAMES[result.modality]],
            ["Model", MODEL_NAME],
        ]

        self.draw_box(figure, 0.58, 0.06, 0.36, 0.25, "none")
        self.draw_heading(figure, 0.60, 0.285, "Details")

        y = 0.250
        for label, value in rows:
            figure.text(0.60, y, label, fontsize=9, color=GREY, va="center")
            figure.text(0.66, y, value, fontsize=9, va="center")
            y = y - 0.036
