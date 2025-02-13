# coding=utf-8
import os
import cv2
import math
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as IDraw
import torch

from dbnet.config import Config
from dbnet.model3l import DBNet
#from dbnet.dataset0.post_process import DbPostProcess
from dbnet.post_process import DbPostProcess

# 297 	210
# 800   560
# 1188	840		/	1184	848
# 1485	1050	/	1488	1056
# 1674	1260	/	1680	1264
# 2079	1470	/	2080	1472
# 2376	1680	/	2384	1680
# LONG_SIDE = 1488
# SHORT_SIDE = 1056


class DBNetPredictor:
    def __init__(self, model_weights_file):
        cfg = Config()

        self.model = DBNet()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')

        # self.inputs_spec = 1364       # A4 120dpi 1408x992
        # self.inputs_spec = 875        # A4 96dpi  1120x800
        self.inputs_spec = 513          # A4 72dpi  864x608
        # self.inputs_spec = 234        # 576x416
        # self.inputs_spec = 180        # 480x384
        # self.inputs_spec = 140        # 448x320

        print('DBNet: Load model from "%s"' % model_weights_file)
        self.model.load_state_dict(torch.load(model_weights_file))
        self.model.to(self.device)
        self.model.eval()

        self.post_process = DbPostProcess()

        self.imgf1 = None
        self.imgf2 = None
        self.imgf3 = None
        # self.imgf4 = None

    @classmethod
    def image_auto_rotation(cls, image):
        # 灰度化
        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

        # 腐蚀、膨胀
        kernel = np.ones((5, 5), np.uint8)
        image_erode = cv2.erode(image_gray, kernel)
        ero_dil = cv2.dilate(image_erode, kernel)

        # 边缘检测
        # canny = cv2.Canny(ero_dil, 50, 200, 3)
        canny = cv2.Canny(ero_dil, 50, 200)

        try:
            # 霍夫变换得到线条
            hough_lines = []
            if len(hough_lines) < 15:
                hough_lines.extend(cv2.HoughLines(canny, 1, np.pi / 180, 300, 0, 0))
            if len(hough_lines) < 15:
                hough_lines.extend(cv2.HoughLines(canny, 1, np.pi / 180, 250, 0, 0))
            if len(hough_lines) < 15:
                hough_lines.extend(cv2.HoughLines(canny, 1, np.pi / 180, 200, 0, 0))
            if len(hough_lines) < 15:
                hough_lines.extend(cv2.HoughLines(canny, 1, np.pi / 180, 150, 0, 0))
            if len(hough_lines) < 15:
                hough_lines.extend(cv2.HoughLines(canny, 1, np.pi / 180, 100, 0, 0))

        except TypeError as e:
            return image

        thetas = []
        for line in hough_lines:
            rho, theta = line[0]

            theta = theta - np.pi / 2
            if 0.01 < abs(theta) < 0.08:
                thetas.append(theta)
        thetas.sort()

        if len(thetas) < 3:
            return image

        keep_size = len(thetas) if len(thetas) < 20 else 20

        while len(thetas) > keep_size:
            thetas.pop(0)
            thetas.pop(-1)

        theta = sum(thetas) / len(thetas)
        angle = theta * 180.0 / np.pi

        if abs(angle) < 0.7 or abs(angle) > 4.0:
            return image

        # rotation
        (w, h) = image.shape[0:2]
        center = (w // 2, h // 2)

        if abs(angle) < 2.0:
            image_wrap = cv2.getRotationMatrix2D(center, angle + 5.0, 1.0)
            image = cv2.warpAffine(image, image_wrap, (h, w), borderValue=(255, 255, 255))
            angle = -5.0

        image_wrap = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, image_wrap, (h, w), borderValue=(255, 255, 255))

        return image

    def resize_image(self, image, color='white'):
        image_width = image.size[0]
        image_height = image.size[1]

        image_scale = math.sqrt((self.inputs_spec * 1024.0) / (image_width * image_height))
        image_scale = 1.2 if image_scale > 1.2 else image_scale

        resize_width = int(image_width * image_scale)
        resize_height = int(image_height * image_scale)
        image = image.resize([resize_width, resize_height], Image.BILINEAR)

        resize_width_pack32 = (resize_width + 31) // 32 * 32 # //整除32，结果为除完后的整数
        resize_height_pack32 = (resize_height + 31) // 32 * 32
        # 将新建图片变换为32的倍数
        image_pack32 = Image.new('RGB', [resize_width_pack32, resize_height_pack32], color)

        image_pack32.paste(image, [0, 0])  # paste

        return image_pack32, image_scale

    @classmethod
    def draw_contours(cls, image, text_contours, color, width):
        img = Image.fromarray(image)
        draw = IDraw.ImageDraw(img)

        for text_contour in text_contours:
            text_contour = np.array(text_contour).reshape([-1, 2]).tolist()

            if len(text_contour) < 2:
                continue

            point_last = text_contour[0]
            for i in range(1, len(text_contour), 1):
                point_current = text_contour[i]
                draw.line([point_last[0], point_last[1], point_current[0], point_current[1]], color, width)
                point_last = point_current

            if len(text_contour) > 2:
                draw.line([point_last[0], point_last[1], text_contour[0][0], text_contour[0][1]], color, width)

        return np.array(img)

    # def draw_contours(cls, image, result, color=(255, 0, 0), width=2):
    #
    #     img_path = image.copy()
    #     for point in result:
    #         point = point.astype(int)
    #         cv2.polylines(img_path, [point], True, color, width)
    #     return img_path

    @classmethod
    def get_gt_info(cls, gt_path):
        gt_rects = []
        gt_texts = []

        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # print(gt_file)

            for line in lines:
                split_line = line.strip().split(',')
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, split_line[:8])
                text = ','.join(split_line[8:])

                gt_rects.append((x1, y1, x2, y2, x3, y3, x4, y4))
                gt_texts.append(text)

        return gt_rects, gt_texts

    def text_detect(self, image):
        image = Image.fromarray(image)
        image_size = image.size
        image, scale = self.resize_image(image)
        # Gray
        if 1 == self.model.input_channel:
            inputs = np.array(image.convert('L'))
            inputs = inputs[np.newaxis, ..., np.newaxis]
        # RGB
        elif 3 == self.model.input_channel:
            inputs = np.array(image)
            inputs = inputs[np.newaxis, ...]
        else:
            raise ValueError('input_channel must be 1 or 3. Got: %d' % self.model.input_channel)

        with torch.no_grad():
            inputs = self.model.normalize(inputs)  # 归一化
            inputs = inputs.to(self.device)

            prob_maps, thresh_maps = self.model(inputs=inputs)

            prob_maps = prob_maps[0].cpu().numpy()
            thresh_maps = thresh_maps[0].cpu().numpy()

            prob_maps = (np.squeeze(prob_maps, axis=0) * 255).astype(np.uint8)
            thresh_maps = (np.squeeze(thresh_maps, axis=0) * 255).astype(np.uint8)

            prob_maps = cv2.medianBlur(prob_maps, 5)
            thresh_maps = cv2.medianBlur(thresh_maps, 5)

            prob_maps = prob_maps.astype(np.float64) / 255.0
            thresh_maps = thresh_maps.astype(np.float64) / 255.0
            # thresh_maps = thresh_maps * 2.0

            # text_contours = self.post_process.get_text_polygons(prob_maps, thresh_maps, image_size, scale)
            text_contours = self.post_process.get_text_rects(prob_maps, thresh_maps, image_size, scale)
        # out features
        binary_maps = (prob_maps - thresh_maps) > 0

        self.imgf1 = Image.fromarray((binary_maps * 255).astype(np.uint8))
        self.imgf2 = Image.fromarray((prob_maps * 255).astype(np.uint8))
        self.imgf3 = Image.fromarray((thresh_maps * 255).astype(np.uint8))

        return text_contours


if __name__ == '__main__':
    pass
