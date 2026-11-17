import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from frontend import theme
from frontend.widgets import Banner

BAR_COLOURS = [theme.INDIGO, theme.BAR_MEDIUM, theme.ORANGE]
BAR_WIDTH = 0.27
CHART_FONT = "Helvetica Neue"


class AnalysisScreen(tk.Frame):
    def __init__(self, parent, results_reader):
        super().__init__(parent, background=theme.WHITE)
        self.results_reader = results_reader
        self.build_banner()
        self.body = tk.Frame(self, background=theme.WHITE)
        self.body.pack(fill="both", expand=True, padx=30, pady=16)
        try:
            self.results = results_reader.read_all()
        except Exception as error:
            self.show_problem(str(error))
            return
        self.build_finding()
        self.build_table()
        self.build_chart()

    def build_banner(self):
        self.banner = Banner(self)
        self.banner.set_texts("Each image type needs its own model", "How the models scored on test images they had never seen.")
        self.banner.pack(fill="x")

    def show_problem(self, text):
        label = tk.Label(self.body, text=text, font=theme.BODY_FONT, background=theme.WHITE,
                         foreground=theme.RED, justify="left")
        label.pack(anchor="w")

    def build_finding(self):
        text = self.results_reader.finding_text(self.results)
        label = tk.Label(self.body, text=text, font=theme.BODY_FONT, background=theme.WHITE,
                         foreground=theme.INK, wraplength=1040, justify="left")
        label.pack(anchor="w", pady=(0, 10))

    def add_cell(self, table, row, column, text, font, colour):
        if column == 0:
            anchor = "w"
        else:
            anchor = "e"
        label = tk.Label(table, text=text, font=font, background=theme.WHITE, foreground=colour,
                         width=24, anchor=anchor)
        label.grid(row=row, column=column, sticky="ew", pady=3)

    def add_rule(self, table, row, colour, height):
        rule = tk.Frame(table, background=colour, height=height)
        rule.grid(row=row, column=0, columnspan=len(self.results) + 1, sticky="ew")

    def add_row(self, table, row, label, attribute_name):
        self.add_cell(table, row, 0, label, theme.BODY_FONT, theme.INK)
        column = 1
        for result in self.results:
            value = getattr(result, attribute_name)
            if attribute_name == "num_images":
                text = str(value)
            else:
                text = format(value, ".3f")
            self.add_cell(table, row, column, text, theme.BODY_FONT, theme.INK)
            column = column + 1
        self.add_rule(table, row + 1, theme.LINE, 1)

    def build_table(self):
        table = tk.Frame(self.body, background=theme.WHITE)
        table.pack(anchor="w", pady=(0, 6))

        self.add_cell(table, 0, 0, "Measure", theme.BODY_BOLD_FONT, theme.INK)
        column = 1
        for result in self.results:
            self.add_cell(table, 0, column, result.title, theme.BODY_BOLD_FONT, theme.INK)
            column = column + 1
        self.add_rule(table, 1, theme.INK, 2)

        self.add_row(table, 2, "Mean lesion AUC", "mean_lesion_auc")
        self.add_row(table, 4, "Grade kappa", "kappa")
        self.add_row(table, 6, "Grade accuracy", "accuracy")
        self.add_row(table, 8, "Test images", "num_images")

    def build_chart(self):
        lesion_names = self.results_reader.lesion_names
        figure = Figure(figsize=(10.4, 3.0), dpi=100, facecolor=theme.WHITE)
        axes = figure.add_subplot(111)

        result_number = 0
        for result in self.results:
            positions = []
            for lesion_number in range(len(lesion_names)):
                positions.append(lesion_number + result_number * BAR_WIDTH)
            axes.bar(positions, result.lesion_aucs, width=BAR_WIDTH, label=result.title, color=BAR_COLOURS[result_number])
            result_number = result_number + 1

        middle_positions = []
        for lesion_number in range(len(lesion_names)):
            middle_positions.append(lesion_number + BAR_WIDTH)
        axes.set_xticks(middle_positions)
        axes.set_xticklabels(lesion_names, fontsize=8, fontname=CHART_FONT, color=theme.INK)
        axes.set_ylim(0.4, 1.0)
        axes.set_ylabel("AUC on the test set", fontsize=9, fontname=CHART_FONT, color=theme.MUTED)
        axes.set_title("Lesion detection: AUC for each lesion", fontsize=11, fontname=CHART_FONT,
                       color=theme.INK, loc="left", fontweight="bold")
        axes.tick_params(axis="y", labelsize=8, colors=theme.MUTED)
        axes.tick_params(axis="x", length=0)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["left"].set_color(theme.LINE)
        axes.spines["bottom"].set_color(theme.INK)
        axes.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.62, 1.17), ncol=3, frameon=False)
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=self.body)
        canvas.draw()
        canvas.get_tk_widget().configure(background=theme.WHITE, highlightthickness=0)
        canvas.get_tk_widget().pack(anchor="w")
