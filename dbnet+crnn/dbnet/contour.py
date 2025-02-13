
import numpy as np
import cv2
import pyclipper
import math


class Contour:

    def __init__(self, predict_ratio=1.5, train_radio=0.84):  # predict_ratio = 1.5, train_radio = 0.7(建议)
        self.predict_ratio = predict_ratio
        self.train_radio = train_radio

    @classmethod
    def contour_rect(cls, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        rect = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return np.array(rect, dtype=np.int32).reshape([-1, 2])

    def calc_delta(self, contour, ratio):
        delta = cv2.contourArea(contour) * ratio / cv2.arcLength(contour, True)
        return delta

    def predict_expand(self, contour):

        area = cv2.contourArea(contour) * 1.03
        length = cv2.arcLength(contour, True) * 0.97

        s1 = 0.5 * length * (1.0 / self.train_radio - 1)
        s2 = math.sqrt((0.5 * length * (1.0 / self.train_radio - 1)) ** 2 + 4 * (2.0 / self.train_radio - 1) * area)
        s3 = 4 * (2.0 / self.train_radio - 1)

        delta = (-s1 + s2) / s3

        offset = pyclipper.PyclipperOffset()
        offset.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        expand = offset.Execute(delta)
        expand = np.array(expand[0]).reshape([-1, 2])
        return expand, delta

# 支持多边形

    def threshold_expand(self, contour):
        delta = self.calc_delta(contour, self.train_radio)

        offset = pyclipper.PyclipperOffset()
        offset.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        expand = offset.Execute(delta)
        expand = np.array(expand[0]).reshape([-1, 2])
        return expand, delta

    def probably_shrink(self, contour):
        delta = self.calc_delta(contour, self.train_radio)

        padding = pyclipper.PyclipperOffset()
        padding.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrink = padding.Execute(-delta)

        if 1 == len(shrink):
            shrink = np.array(shrink[0]).reshape(-1, 2)
        else:
            shrink = np.array([])
        return shrink, delta
