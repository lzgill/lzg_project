import numpy as np
import cv2
import math
from dbnet.contour import Contour


class ProbablyMap:

    def __init__(self, min_text_size=4):
        self.min_text_size = min_text_size
        self.contour_f = Contour()

    def __call__(self, image, contours):

        h, w = image.shape[:2]  # hwc
        # 防止越界
        contours = self.clip_contours(contours, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        weight = np.zeros(image.shape[:2], dtype=np.float32)
        filterd = []

        for i in range(len(contours)):
            contour = contours[i]
            # 计算每个框的高和宽
            height = max(contour[:, 1]) - min(contour[:, 1])
            width = max(contour[:, 0]) - min(contour[:, 0])

            if cv2.contourArea(contour.astype(np.int32)) < 16:
                if cv2.contourArea(contour.astype(np.int32)) > 0:
                    cv2.fillPoly(mask, contour.astype(np.int32)[np.newaxis, :, :], 0)
                continue

            if min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, contour.astype(np.int32)[np.newaxis, :, :], 0)
                continue

            shrink, delta = self.contour_f.probably_shrink(contour)

            if 0 == shrink.size:
                cv2.fillPoly(mask, contour.astype(np.int32)[np.newaxis, :, :], 0)
                continue

            filterd.append(contour)
            cv2.fillPoly(gt, [shrink.astype(np.int32)], 1)  # 用1来填充
            self.draw_border_map(contour, weight)

        weight = weight * 1.0 + 1.0
        # print('add "weight = weight * 2.0 + 1.0 before" training.')

        prob_map = gt
        prob_mask = mask
        prob_weight = weight
        return prob_map, prob_mask, prob_weight, filterd

    @classmethod
    def clip_contours(cls, contours, h, w):
        for contour in contours:
            contour[:, 0] = np.clip(contour[:, 0], 0, w - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, h - 1)
        # for i in range(len(contours)):
        #     area = cv2.contourArea(contours[i])
        #     if area > 32:
        #         contours[i] = contours[i][::-1, :]
        return contours

    def draw_border_map(self, contour, canvas):
        contour = contour.copy()
        if cv2.contourArea(contour) < 1:
            return

        shrink, delta = self.contour_f.probably_shrink(contour)
        padded_contour = contour.astype(np.int32)
        contour = shrink

        xmin = padded_contour[:, 0].min()
        xmax = padded_contour[:, 0].max()
        ymin = padded_contour[:, 1].min()
        ymax = padded_contour[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
#
        distance_map = np.zeros((contour.shape[0], height, width), dtype=np.float32)

        for i in range(contour.shape[0]):
            j = (i + 1) % contour.shape[0]
            absolute_distance = self.distance(xs, ys, contour[i], contour[j])
            distance_map[i] = np.clip(absolute_distance / delta, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,xmin_valid - xmin:xmax_valid - xmax + width],
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