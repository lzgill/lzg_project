import cv2
import numpy as np
from dbnet.contour import Contour


class DbPostProcess:

    def __init__(self):
        self.max_contours = 2000
        self.perimeter_limit = 10
        self.contour_f = Contour()

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
        return np.array(rect, dtype=np.int).reshape([-1, 2])

    # def contour_score(self, bitmap, _box):
    #     h, w = bitmap.shape[:2]
    #     box = _box.copy()
    #     xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    #     xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    #     ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    #     ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)
    #
    #     mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    #     box[:, 0] = box[:, 0] - xmin
    #     box[:, 1] = box[:, 1] - ymin
    #     cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    #     return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def get_text_contours(self, prob_maps, thresh_maps, image_size, image_scale):
        # binary_maps = (prob_maps - thresh_maps) < 0.1
        # contours, _ = cv2.findContours((binary_maps * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        binary_maps = (prob_maps - thresh_maps) > 0
        contours, _ = cv2.findContours((binary_maps * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_contours)

        text_contours = []
        for i in range(num_contours):
            contour = contours[i].reshape([-1, 2])

            if cv2.arcLength(contour, True) < 16:
                continue
            if cv2.contourArea(contour) < 16:
                continue

            contour, delta = self.contour_f.predict_expand(contour)

            epsilon = 0.005 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
            contour = contour.reshape([-1, 2])

            contour = contour / image_scale
            contour = np.clip(contour, [0, 0], [image_size[0] - 1, image_size[1] - 1])
            contour = np.array(contour, dtype=np.int)

            if cv2.contourArea(contour) > 0.2 * image_size[0] * image_size[1]:
                continue

            # if cv2.contourArea(contour) < 256:
            #     continue

            text_contours.append(contour)
        return text_contours

    # 根据prob_map获取rectangle
    def get_text_rects(self, prob_maps, thresh_maps, image_size, image_scale):
        # 利用了thresh_maps 与 prob_maps
        # binary_maps = (prob_maps - thresh_maps) < 0.1
        # contours, _ = cv2.findContours((binary_maps * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        binary_maps = prob_maps > 0.5  # score = 0.5
        contours, _ = cv2.findContours((binary_maps * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_contours)
        text_rects = []
        for i in range(num_contours):
            contour = contours[i].reshape([-1, 2])

            if cv2.arcLength(contour, True) < 16:
                continue
            if cv2.contourArea(contour) < 16:
                continue
            # 扩张轮廓
            contour, delta = self.contour_f.predict_expand(contour)
            rect = self.contour_f.contour_rect(contour)
            assert rect.shape[0] == 4

            rect = rect / image_scale  #
            rect = np.clip(rect, [0, 0], [image_size[0] - 1, image_size[1] - 1])
            rect = np.array(rect, dtype=np.int32)

            if cv2.contourArea(rect) > 0.2 * image_size[0] * image_size[1]:
                continue

            # if cv2.contourArea(rect) < 256:
            #     continue

            x1 = min(rect[0, 0], rect[3, 0])
            y1 = min(rect[0, 1], rect[1, 1])
            x3 = max(rect[1, 0], rect[2, 0])
            y3 = max(rect[2, 1], rect[3, 1])

            extend = 0
            x1 = max(x1 - extend, 0)
            y1 = max(y1 - extend, 0)
            x3 = min(x3 + extend, image_size[0] - 1)
            y3 = min(y3 + extend, image_size[1] - 1)

            rect[0, 0] = x1
            rect[0, 1] = y1
            rect[1, 0] = x3
            rect[1, 1] = y1
            rect[2, 0] = x3
            rect[2, 1] = y3
            rect[3, 0] = x1
            rect[3, 1] = y3

            text_rects.append(rect)
        return text_rects
