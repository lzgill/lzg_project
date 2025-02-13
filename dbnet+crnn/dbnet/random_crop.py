
import random
import cv2
import numpy as np

# dbnet+:afm
# random crop algorithm similar to https://github.com/argman/EAST
class EastRandomCropData:
    def __init__(self, image_size=(640, 640), max_tries=50, min_crop_side_ratio=0.1, keep_ratio=True):
        self.image_size = image_size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, image, contours):

        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = self.crop_area(image, contours)
        # crop 图片 保持比例填充
        scale_w = self.image_size[0] / crop_w
        scale_h = self.image_size[1] / crop_h
        scale = min(scale_w, scale_h)
        # 保持一个为640
        h = int(crop_h * scale)
        w = int(crop_w * scale)

        image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        if self.keep_ratio:
            if len(image.shape) == 3:
                image_pad = np.zeros((self.image_size[1], self.image_size[0], image.shape[2]), image.dtype)
            else:
                image_pad = np.zeros((self.image_size[1], self.image_size[0]), image.dtype)
                # 这里相当于做了pad操作，用0填充
            image_pad[:h, :w] = cv2.resize(image, (w, h))
            image = image_pad
        else:
            image = cv2.resize(image, self.image_size)

        # crop 文本框
        crop_contours = []
        for poly in contours:
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()   # 缩小，文字框的位置缩减
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                crop_contours.append(poly)

        crop_contours = np.float32(crop_contours)
        return image, crop_contours

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, r):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys):
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h
