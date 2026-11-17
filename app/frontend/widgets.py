import tkinter as tk

from PIL import ImageTk

from frontend import theme

LIST_WIDTH = 330
LIST_HEIGHT = 400
FIRST_ROW_Y = 48
ROW_HEIGHT = 50

CHART_WIDTH = 260
CHART_HEIGHT = 400
BASE_Y = 300
TALLEST_BAR = 210
BAR_WIDTH = 36
BAR_STEP = 52
LEGEND_LINES = ["0 No DR, 1 Mild, 2 Moderate,", "3 Severe, 4 Proliferative"]


class Banner(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, height=132)
        self.pack_propagate(False)

        self.headline = tk.Label(self, font=theme.HEADLINE_FONT, foreground=theme.WHITE)
        self.headline.place(x=28, y=20)

        self.subline = tk.Label(self, font=theme.SUBLINE_FONT, foreground=theme.WHITE)
        self.subline.place(x=28, y=82)

        self.number = tk.Label(self, font=theme.NUMBER_FONT, foreground=theme.WHITE)
        self.number.place(relx=1.0, x=-28, y=14, anchor="ne")

        self.number_note = tk.Label(self, font=theme.BODY_FONT, foreground=theme.WHITE)
        self.number_note.place(relx=1.0, x=-28, y=86, anchor="ne")

        self.set_colour(theme.INDIGO_LIGHT)

    def set_colour(self, colour):
        self.configure(background=colour)
        self.headline.configure(background=colour)
        self.subline.configure(background=colour)
        self.number.configure(background=colour)
        self.number_note.configure(background=colour)

    def set_texts(self, headline, subline):
        self.headline.configure(text=headline)
        self.subline.configure(text=subline)

    def set_number(self, number, note):
        self.number.configure(text=number)
        self.number_note.configure(text=note)


class FlatButton(tk.Label):
    def __init__(self, parent, text, command):
        super().__init__(
            parent, text=text, font=theme.BODY_BOLD_FONT, foreground=theme.WHITE,
            background=theme.INDIGO, padx=28, pady=10, cursor="hand2",
        )
        self.command = command
        self.enabled = True
        self.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if self.enabled:
            self.command()

    def enable(self):
        self.enabled = True
        self.configure(background=theme.INDIGO)

    def disable(self):
        self.enabled = False
        self.configure(background=theme.INDIGO_PALE)


class ImageViewer(tk.Frame):
    def __init__(self, parent, box_size):
        super().__init__(parent, background=theme.WHITE)
        self.box_size = box_size
        self.photo = None
        self.canvas = tk.Canvas(self, width=box_size, height=box_size, background="black", highlightthickness=0)
        self.canvas.pack()

    def fit_to_box(self, image):
        width, height = image.size
        scale = self.box_size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        return image.resize((new_width, new_height))

    def show_image(self, image):
        fitted_image = self.fit_to_box(image)
        self.photo = ImageTk.PhotoImage(fitted_image)
        middle = self.box_size / 2
        self.canvas.delete("all")
        self.canvas.create_image(middle, middle, image=self.photo)

    def show_text(self, text):
        self.photo = None
        middle = self.box_size / 2
        self.canvas.delete("all")
        self.canvas.create_text(middle, middle, text=text, font=theme.BODY_FONT, fill=theme.BAR_PALE)

    def clear(self):
        self.photo = None
        self.canvas.delete("all")


class LesionList(tk.Canvas):
    def __init__(self, parent, on_select):
        super().__init__(parent, width=LIST_WIDTH, height=LIST_HEIGHT, background=theme.WHITE, highlightthickness=0)
        self.on_select = on_select
        self.rows = []
        self.selected = None
        self.bind("<Button-1>", self.on_click)
        self.draw_all()

    def draw_title(self):
        self.create_text(0, 2, text="Lesion findings", font=theme.SECTION_FONT, fill=theme.INK, anchor="nw")
        self.create_rectangle(0, 32, LIST_WIDTH, 34, fill=theme.INK, width=0)

    def bar_colour(self, word):
        if word == "Likely":
            return theme.INDIGO
        if word == "Possible":
            return theme.BAR_MEDIUM
        return theme.BAR_PALE

    def draw_row(self, y, name, score, word, is_selected):
        if word == "Unlikely":
            text_colour = theme.MUTED
            name_font = theme.BODY_FONT
        else:
            text_colour = theme.INK
            name_font = theme.BODY_BOLD_FONT

        if is_selected:
            self.create_rectangle(0, y - 6, LIST_WIDTH, y + 42, fill=theme.HIGHLIGHT_PALE, width=0)
            self.create_rectangle(0, y - 6, 4, y + 42, fill=theme.HIGHLIGHT, width=0)

        percent_text = str(round(score * 100)) + "%  " + word
        self.create_text(10, y, text=name, font=name_font, fill=text_colour, anchor="nw")
        self.create_text(LIST_WIDTH - 6, y + 1, text=percent_text, font=theme.SMALL_FONT, fill=text_colour, anchor="ne")

        self.create_rectangle(10, y + 24, LIST_WIDTH - 6, y + 32, fill=theme.SOFT, width=0)
        bar_end = 10 + (LIST_WIDTH - 16) * score
        if bar_end >= 11:
            self.create_rectangle(10, y + 24, bar_end, y + 32, fill=self.bar_colour(word), width=0)
        self.create_rectangle(0, y + 42, LIST_WIDTH, y + 43, fill=theme.LINE, width=0)

    def draw_all(self):
        self.delete("all")
        self.draw_title()
        y = FIRST_ROW_Y
        row_number = 0
        for name, score, word in self.rows:
            self.draw_row(y, name, score, word, row_number == self.selected)
            y = y + ROW_HEIGHT
            row_number = row_number + 1

    def show_rows(self, rows):
        self.rows = rows
        self.selected = None
        self.draw_all()

    def select(self, row_number):
        self.selected = row_number
        self.draw_all()

    def on_click(self, event):
        if event.y < FIRST_ROW_Y:
            return
        row_number = (event.y - FIRST_ROW_Y) // ROW_HEIGHT
        if row_number < len(self.rows):
            self.on_select(row_number)


class GradeChart(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, width=CHART_WIDTH, height=CHART_HEIGHT, background=theme.WHITE, highlightthickness=0)
        self.draw_title()

    def draw_title(self):
        self.create_text(0, 2, text="Grade probabilities", font=theme.SECTION_FONT, fill=theme.INK, anchor="nw")
        self.create_rectangle(0, 32, CHART_WIDTH, 34, fill=theme.INK, width=0)

    def find_largest(self, rows):
        largest = 0.01
        for name, probability, percent_text in rows:
            if probability > largest:
                largest = probability
        return largest

    def draw_bar(self, grade, probability, percent_text, largest, colour, text_colour):
        height = max(3, TALLEST_BAR * probability / largest)
        left = 6 + grade * BAR_STEP
        middle = left + BAR_WIDTH / 2
        self.create_rectangle(left, BASE_Y - height, left + BAR_WIDTH, BASE_Y, fill=colour, width=0)
        self.create_text(middle, BASE_Y - height - 6, text=percent_text, font=theme.SMALL_BOLD_FONT,
                         fill=text_colour, anchor="s")
        self.create_text(middle, BASE_Y + 8, text=str(grade), font=theme.SECTION_FONT,
                         fill=text_colour, anchor="n")

    def draw_legend(self):
        y = BASE_Y + 44
        for line in LEGEND_LINES:
            self.create_text(0, y, text=line, font=theme.SMALL_FONT, fill=theme.MUTED, anchor="nw")
            y = y + 18

    def show_rows(self, rows, chosen_grade, chosen_colour):
        self.delete("all")
        self.draw_title()
        largest = self.find_largest(rows)
        grade = 0
        for name, probability, percent_text in rows:
            if grade == chosen_grade:
                self.draw_bar(grade, probability, percent_text, largest, chosen_colour, theme.INK)
            else:
                self.draw_bar(grade, probability, percent_text, largest, theme.BAR_PALE, theme.MUTED)
            grade = grade + 1
        self.create_rectangle(0, BASE_Y, CHART_WIDTH, BASE_Y + 1, fill=theme.INK, width=0)
        self.draw_legend()
