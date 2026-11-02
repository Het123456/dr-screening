from abc import ABC, abstractmethod


class Prediction:
    # The answer a model gives for one image
    def __init__(self, grade, lesion_present, lesion_scores):
        self.grade = grade                    # DR grade, 0 to 4
        self.lesion_present = lesion_present  # list of 7 values, 1 = present, 0 = absent
        self.lesion_scores = lesion_scores    # list of 7 confidences between 0 and 1

    def __repr__(self):
        return (f"Prediction(grade={self.grade}, lesion_present={self.lesion_present}, "
                f"lesion_scores={self.lesion_scores})")


class BaseRetinalModel(ABC):

    @abstractmethod
    def predict(self, image):
        pass


class FakeModel(BaseRetinalModel):
    def predict(self, image):
        return Prediction(
            grade=2,
            lesion_present=[1, 1, 0, 0, 0, 0, 0],
            lesion_scores=[0.91, 0.78, 0.12, 0.05, 0.03, 0.02, 0.01],
        )


if __name__ == "__main__":
    model = FakeModel()
    result = model.predict(image=None)
    print(result)
    print("grade:", result.grade)
    print("lesion present:", result.lesion_present)
    print("lesion scores:", result.lesion_scores)
    