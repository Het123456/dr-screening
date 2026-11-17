import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image

from frontend import theme
from frontend.widgets import Banner, FlatButton, GradeChart, ImageViewer, LesionList

IMAGE_FILE_TYPES = [("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
CONTROLS_WIDTH = 420

MODALITY_TEXTS = {"cfp": "Colour fundus photo (CFP)", "uwf": "Ultra-widefield photo (UWF)"}
LESION_NOTE = "Likely means a score of 70% or more, Possible means 30% to 70%."
MAP_HINT = "Click a Likely or Possible lesion to see where the model looks."
NO_MAP_TEXT = "No heat map: the model finds this lesion unlikely."


class LoadScreen(tk.Frame):
    def __init__(self, parent, analyzer, on_result):
        super().__init__(parent, background=theme.WHITE)
        self.analyzer = analyzer
        self.on_result = on_result
        self.image_path = None
        self.modality = tk.StringVar(value="cfp")
        self.build_banner()
        self.build_controls()
        self.build_viewer()

    def build_banner(self):
        self.banner = Banner(self)
        self.banner.set_texts("Screen a retina photo", "Choose the image type, choose the photo, then analyse it.")
        self.banner.pack(fill="x")

    def add_heading(self, parent, text):
        label = tk.Label(parent, text=text, font=theme.SECTION_FONT, background=theme.WHITE, foreground=theme.INK)
        label.pack(anchor="w", pady=(24, 4))
        rule = tk.Frame(parent, background=theme.INK, height=2, width=CONTROLS_WIDTH)
        rule.pack(anchor="w", pady=(0, 12))

    def add_choice(self, parent, text, value):
        choice = tk.Radiobutton(
            parent, text=text, variable=self.modality, value=value, font=theme.BODY_FONT,
            background=theme.WHITE, foreground=theme.INK, activebackground=theme.WHITE, highlightthickness=0,
        )
        choice.pack(anchor="w", pady=2)

    def build_controls(self):
        controls = tk.Frame(self, background=theme.WHITE)
        controls.pack(side="left", anchor="n", padx=(30, 40))

        self.add_heading(controls, "1. Image type")
        self.add_choice(controls, "Colour fundus photo (CFP)", "cfp")
        self.add_choice(controls, "Ultra-widefield photo (UWF)", "uwf")

        self.add_heading(controls, "2. Photo")
        choose_button = FlatButton(controls, "Choose image...", self.choose_image)
        choose_button.pack(anchor="w")
        self.file_label = tk.Label(
            controls, text="No image chosen", font=theme.SMALL_FONT, background=theme.WHITE,
            foreground=theme.MUTED, wraplength=CONTROLS_WIDTH, justify="left",
        )
        self.file_label.pack(anchor="w", pady=(8, 0))

        self.add_heading(controls, "3. Screening")
        self.analyse_button = FlatButton(controls, "Analyse", self.analyse)
        self.analyse_button.pack(anchor="w")
        self.analyse_button.disable()
        self.status_label = tk.Label(
            controls, text="", font=theme.SMALL_FONT, background=theme.WHITE,
            foreground=theme.MUTED, wraplength=CONTROLS_WIDTH, justify="left",
        )
        self.status_label.pack(anchor="w", pady=(8, 0))

    def build_viewer(self):
        self.viewer = ImageViewer(self, 520)
        self.viewer.pack(side="left", anchor="n", pady=30)
        self.viewer.show_text("The photo you choose is shown here")

    def choose_image(self):
        path = filedialog.askopenfilename(title="Choose a retina image", filetypes=IMAGE_FILE_TYPES)
        if path == "":
            return
        try:
            image = Image.open(path)
            image = image.convert("RGB")
        except (OSError, ValueError):
            messagebox.showerror("Image problem", "This file could not be opened as an image.")
            return
        self.image_path = path
        self.viewer.show_image(image)
        self.file_label.config(text=path)
        self.status_label.config(text="")
        self.analyse_button.enable()

    def analyse(self):
        self.status_label.config(text="Analysing, please wait...")
        self.analyse_button.disable()
        self.update()
        try:
            result = self.analyzer.analyze(self.image_path, self.modality.get())
        except Exception as error:
            self.status_label.config(text="")
            self.analyse_button.enable()
            messagebox.showerror("Analysis problem", str(error))
            return
        self.status_label.config(text="Done.")
        self.analyse_button.enable()
        self.on_result(result)


class ResultsScreen(tk.Frame):
    def __init__(self, parent, report_builder):
        super().__init__(parent, background=theme.WHITE)
        self.report_builder = report_builder
        self.result = None
        self.shown_lesion = None
        self.build_banner()
        self.build_footer()
        self.build_body()

    def build_banner(self):
        self.banner = Banner(self)
        self.banner.set_texts("No image analysed yet", "Go to Load image, choose a photo and analyse it.")
        self.banner.pack(fill="x")

    def build_footer(self):
        footer = tk.Frame(self, background=theme.SOFT, height=74)
        footer.pack_propagate(False)
        footer.pack(side="bottom", fill="x")

        self.report_button = FlatButton(footer, "Save PDF report...", self.save_report)
        self.report_button.pack(side="left", padx=(30, 20))
        self.report_button.disable()

        note = tk.Label(footer, text=LESION_NOTE, font=theme.SMALL_FONT, background=theme.SOFT, foreground=theme.MUTED)
        note.pack(side="left")

    def build_body(self):
        body = tk.Frame(self, background=theme.WHITE)
        body.pack(fill="both", expand=True, padx=30, pady=24)

        photo_column = tk.Frame(body, background=theme.WHITE)
        photo_column.pack(side="left", anchor="n")
        self.viewer = ImageViewer(photo_column, 380)
        self.viewer.pack()
        self.file_label = tk.Label(photo_column, text="", font=theme.BODY_BOLD_FONT,
                                   background=theme.WHITE, foreground=theme.INK)
        self.file_label.pack(anchor="w", pady=(12, 0))
        self.type_label = tk.Label(photo_column, text="", font=theme.SMALL_FONT,
                                   background=theme.WHITE, foreground=theme.MUTED)
        self.type_label.pack(anchor="w")
        self.map_label = tk.Label(photo_column, text="", font=theme.SMALL_FONT, background=theme.WHITE,
                                  foreground=theme.INK, wraplength=380, justify="left")
        self.map_label.pack(anchor="w", pady=(8, 0))

        self.lesion_list = LesionList(body, self.choose_lesion)
        self.lesion_list.pack(side="left", anchor="n", padx=(30, 40))

        self.grade_chart = GradeChart(body)
        self.grade_chart.pack(side="left", anchor="n")

    def show_result(self, result):
        self.result = result
        decision = result.decision
        colour = theme.LEVEL_COLOURS[decision.level]

        self.banner.set_colour(colour)
        self.banner.set_texts(decision.advice, result.grade_text())
        score_text = format(decision.score, ".2f")
        note_text = "referral score, refer from " + format(decision.cutoff, ".2f")
        self.banner.set_number(score_text, note_text)

        self.viewer.show_image(result.image)
        self.file_label.config(text=Path(result.image_path).name)
        self.type_label.config(text=MODALITY_TEXTS[result.modality])

        self.lesion_list.show_rows(result.lesion_rows())
        self.shown_lesion = None
        self.map_label.config(text=MAP_HINT)
        self.grade_chart.show_rows(result.grade_rows(), result.prediction.grade, colour)
        self.report_button.enable()

    def show_plain_photo(self, text):
        self.shown_lesion = None
        self.lesion_list.select(None)
        self.viewer.show_image(self.result.image)
        self.map_label.config(text=text)

    def choose_lesion(self, lesion_number):
        if self.result is None:
            return
        if lesion_number == self.shown_lesion:
            self.show_plain_photo(MAP_HINT)
            return
        heatmap = self.result.heatmaps[lesion_number]
        if heatmap is None:
            self.show_plain_photo(NO_MAP_TEXT)
            return
        name = self.result.lesion_rows()[lesion_number][0]
        self.shown_lesion = lesion_number
        self.lesion_list.select(lesion_number)
        self.viewer.show_image(heatmap)
        self.map_label.config(text="Showing " + name + ". The bright blue areas pushed the model's score up most. Click it again to hide.")

    def save_report(self):
        path = filedialog.asksaveasfilename(
            title="Save report", defaultextension=".pdf", filetypes=[("PDF file", "*.pdf")]
        )
        if path == "":
            return
        try:
            self.report_builder.save(self.result, path)
        except Exception as error:
            messagebox.showerror("Report problem", str(error))
            return
        messagebox.showinfo("Report saved", "The report was saved to:\n" + path)
