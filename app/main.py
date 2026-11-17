import sys

from backend.analyzer import Analyzer
from backend.models import FakeModel, RETFoundModel
from backend.policies import LesionPolicy, ReferralPolicy
from backend.reports import ReportBuilder, ResultsReader
from backend.settings import ModelSettings
from frontend.main_window import MainWindow


class Application:
    def __init__(self, use_fake_models):
        self.use_fake_models = use_fake_models

    def build_real_analyzer(self):
        analyzer = Analyzer()

        cfp = ModelSettings("cfp", "cfp_retfound_512", 512, "cfp_512_referral.json", "cfp_512_lesion_calibration.json")
        self.add_real_model(analyzer, cfp)

        uwf = ModelSettings("uwf", "uwf_v2", 224, "uwf_v2_referral.json", "uwf_v2_lesion_calibration.json")
        self.add_real_model(analyzer, uwf)

        return analyzer

    def add_real_model(self, analyzer, settings):
        model = RETFoundModel(settings.checkpoint_path, settings.use_flip, settings.image_size)
        referral_policy = ReferralPolicy(settings.cutoff)
        lesion_policy = LesionPolicy(settings.lesion_slopes, settings.lesion_intercepts)
        analyzer.add_model(settings.modality, model, referral_policy, lesion_policy, settings.image_size)

    def build_fake_analyzer(self):
        analyzer = Analyzer()
        slopes = [1, 1, 1, 1, 1, 1, 1]
        intercepts = [0, 0, 0, 0, 0, 0, 0]
        analyzer.add_model("cfp", FakeModel(), ReferralPolicy(0.5), LesionPolicy(slopes, intercepts), 224)
        analyzer.add_model("uwf", FakeModel(), ReferralPolicy(0.5), LesionPolicy(slopes, intercepts), 224)
        return analyzer

    def run(self):
        if self.use_fake_models:
            analyzer = self.build_fake_analyzer()
        else:
            analyzer = self.build_real_analyzer()

        window = MainWindow(analyzer, ResultsReader(), ReportBuilder())
        window.mainloop()


if __name__ == "__main__":
    use_fake_models = "--fake" in sys.argv
    application = Application(use_fake_models)
    application.run()
