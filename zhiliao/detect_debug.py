import random

class PredictRes(object):

    __slots__ = ['eye_abnormal','mouth_abnormal']

    def __init__(self):
        self.eye_abnormal = None  # True if eye is closed
        self.mouth_abnormal = None  # True if mouth is opened

class Detection(object):

    __slots__ = []

    def __init__(self, weights, img_size=640):
        pass

    def detect(self, img):
        assert len(img.shape) == 3,"Image must have 3 dimensions"

        pred = PredictRes()
        pred.eye_abnormal = bool(random.randint(0,1))
        pred.mouth_abnormal = bool(random.randint(0,1))

        return pred