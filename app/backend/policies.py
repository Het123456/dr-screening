import math

from backend.models import REFERABLE_GRADE

URGENT_GRADE = 3

LIKELY_FROM = 0.7
POSSIBLE_FROM = 0.3
SMALLEST_SCORE = 0.000001


class ReferralDecision:
    def __init__(self, refer, level, advice, score, cutoff):
        self.refer = refer
        self.level = level
        self.advice = advice
        self.score = score
        self.cutoff = cutoff


class ReferralPolicy:
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def should_refer(self, prediction):
        if prediction.referral_score() >= self.cutoff:
            return True
        if prediction.grade >= REFERABLE_GRADE:
            return True
        return False

    def decide(self, prediction):
        score = prediction.referral_score()
        refer = self.should_refer(prediction)

        if not refer:
            level = "routine"
            advice = "No referral needed. Screen again in 12 months."
        elif prediction.grade >= URGENT_GRADE:
            level = "urgent"
            advice = "Urgent referral to an eye specialist."
        else:
            level = "refer"
            advice = "Refer to an eye specialist."

        return ReferralDecision(refer, level, advice, score, self.cutoff)


class LesionPolicy:
    def __init__(self, slopes, intercepts):
        self.slopes = slopes
        self.intercepts = intercepts

    def calibrate(self, lesion_number, raw_score):
        score = raw_score
        if score < SMALLEST_SCORE:
            score = SMALLEST_SCORE
        if score > 1 - SMALLEST_SCORE:
            score = 1 - SMALLEST_SCORE
        raw_logit = math.log(score / (1 - score))
        new_logit = self.slopes[lesion_number] * raw_logit + self.intercepts[lesion_number]
        return 1 / (1 + math.exp(-new_logit))

    def calibrate_all(self, raw_scores):
        scores = []
        for lesion_number in range(len(raw_scores)):
            scores.append(self.calibrate(lesion_number, raw_scores[lesion_number]))
        return scores

    def label(self, score):
        if score >= LIKELY_FROM:
            return "Likely"
        if score >= POSSIBLE_FROM:
            return "Possible"
        return "Unlikely"

    def label_all(self, scores):
        labels = []
        for score in scores:
            labels.append(self.label(score))
        return labels
