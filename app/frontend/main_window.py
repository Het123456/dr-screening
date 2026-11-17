import tkinter as tk

from frontend import theme
from frontend.analysis_screen import AnalysisScreen
from frontend.screens import LoadScreen, ResultsScreen

LOAD_TAB = "Load image"
RESULTS_TAB = "Results"
ANALYSIS_TAB = "Model analysis"


class HeaderTab(tk.Frame):
    def __init__(self, parent, name, on_select):
        super().__init__(parent, background=theme.INDIGO)
        self.name = name
        self.on_select = on_select

        self.line = tk.Frame(self, background=theme.INDIGO, height=4)
        self.line.pack(side="bottom", fill="x")

        self.label = tk.Label(
            self, text=name, font=theme.BODY_FONT, background=theme.INDIGO,
            foreground=theme.HEADER_MUTED, cursor="hand2",
        )
        self.label.pack(side="bottom", expand=True)
        self.label.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        self.on_select(self.name)

    def mark_selected(self):
        self.label.configure(font=theme.BODY_BOLD_FONT, foreground=theme.WHITE)
        self.line.configure(background=theme.WHITE)

    def mark_not_selected(self):
        self.label.configure(font=theme.BODY_FONT, foreground=theme.HEADER_MUTED)
        self.line.configure(background=theme.INDIGO)


class HeaderBar(tk.Frame):
    def __init__(self, parent, title, on_select):
        super().__init__(parent, background=theme.INDIGO, height=48)
        self.pack_propagate(False)
        self.on_select = on_select
        self.tabs = []

        title_label = tk.Label(
            self, text=title, font=theme.TITLE_FONT, background=theme.INDIGO, foreground=theme.WHITE
        )
        title_label.pack(side="left", padx=30)

        self.tab_area = tk.Frame(self, background=theme.INDIGO)
        self.tab_area.pack(side="right", padx=18, fill="y")

    def add_tab(self, name):
        tab = HeaderTab(self.tab_area, name, self.on_select)
        tab.pack(side="left", padx=12, fill="y")
        self.tabs.append(tab)

    def select(self, name):
        for tab in self.tabs:
            if tab.name == name:
                tab.mark_selected()
            else:
                tab.mark_not_selected()


class MainWindow(tk.Tk):
    def __init__(self, analyzer, results_reader, report_builder):
        super().__init__()
        self.title("DR Screening Assistant")
        self.geometry("1100x780")
        self.minsize(1100, 780)
        self.configure(background=theme.WHITE)

        self.header = HeaderBar(self, "DR Screening Assistant", self.show_screen)
        self.header.pack(fill="x")

        self.screens = {}
        self.current_screen = None

        self.load_screen = LoadScreen(self, analyzer, self.show_result)
        self.results_screen = ResultsScreen(self, report_builder)
        self.analysis_screen = AnalysisScreen(self, results_reader)

        self.add_screen(LOAD_TAB, self.load_screen)
        self.add_screen(RESULTS_TAB, self.results_screen)
        self.add_screen(ANALYSIS_TAB, self.analysis_screen)
        self.show_screen(LOAD_TAB)

    def add_screen(self, name, screen):
        self.screens[name] = screen
        self.header.add_tab(name)

    def show_screen(self, name):
        if self.current_screen is not None:
            self.current_screen.pack_forget()
        self.current_screen = self.screens[name]
        self.current_screen.pack(fill="both", expand=True)
        self.header.select(name)

    def show_result(self, result):
        self.results_screen.show_result(result)
        self.show_screen(RESULTS_TAB)
