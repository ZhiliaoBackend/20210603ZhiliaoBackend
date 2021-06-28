import random

class PredictRes(object):

    __slots__ = ['eye_abnormal','mouth_abnormal']

    def __init__(self):
        self.eye_abnormal = False  # True if eye is closed
        self.mouth_abnormal = False  # True if mouth is opened

class Detection(object):

    __slots__ = []

    def __init__(self):
        pass

    def detect(self, img):
        assert len(img.shape) == 3,"Image must have 3 dimensions"

        pred = PredictRes()
        pred.eye_abnormal = not bool(random.randint(0,10))
        pred.mouth_abnormal = not bool(random.randint(0,10))

        return pred