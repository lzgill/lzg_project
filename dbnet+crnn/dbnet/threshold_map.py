import cv2
import numpy as np
import math
from dbnet.contour import Contour


class ThresholdMap:
    def __init__(self, thresh_min=0.3, thresh_max=0.7):
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.contour_f = Contour()

    def __call__(self, image, contours):

        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(contours)):
            self.draw_border_map(contours[i], canvas=canvas, mask=mask)

        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        thresh_map = canvas
        thresh_mask = mask
        return thresh_map, thresh_mask

    def draw_border_map(self, contour, canvas, mask):
        contour = contour.copy()
        if cv2.contourArea(contour) < 1:
            return

        padded_contour, delta = self.contour_f.threshold_expand(contour)
        # 根据扩张的多边形框架扩充得到边框坐标padded_contour
        cv2.fillPoly(mask, [padded_contour.astype(np.int32)], 1.0)
        # 得到外接矩形框
        xmin = padded_contour[:, 0].min()
        xmax = padded_contour[:, 0].max()
        ymin = padded_contour[:, 1].min()
        ymax = padded_contour[:, 1].max()
        # get the w and h
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros((contour.shape[0], height, width), dtype=np.float32)
        # 求距离
        for i in range(contour.shape[0]):
            j = (i + 1) % contour.shape[0]
            absolute_distance = self.distance(xs, ys, contour[i], contour[j])
            distance_map[i] = np.clip(absolute_distance / delta, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        # canvas[h,w]
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] =np.fmax(1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height, xmin_valid - xmin:xmax_valid - xmax + width],
                    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, pt1, pt2):

        # reference to https://blog.csdn.net/love_phoebe/article/details/81112531
        v1_x = pt2[0] - pt1[0]
        v1_y = pt2[1] - pt1[1]

        v2_xs = xs - pt1[0]
        v2_ys = ys - pt1[1]

        v3_xs = xs - pt2[0]
        v3_ys = ys - pt2[1]

        len_v1 = math.sqrt(v1_x ** 2 + v1_y ** 2)
        len_v2s = np.sqrt(np.square(v2_xs) + np.square(v2_ys))
        len_v3s = np.sqrt(np.square(v3_xs) + np.square(v3_ys))

        cross = v1_x * v2_ys - v2_xs * v1_y

        c1 = v1_x * v2_xs + v1_y * v2_ys
        c2 = v1_x * v3_xs + v1_y * v3_ys

        if len_v1 < 0.1:
            return len_v2s

        result = np.abs(cross / len_v1)
        result = (c1 < 0) * len_v2s + (c1 >= 0) * result
        result = (c2 > 0) * len_v3s + (c2 <= 0) * result

        return result
