
import cv2
import numpy as np
import glob
import os
import copy
import random
import PIL.Image as Image
import PIL.ImageDraw as IDraw

from dbnet.augment import ImageAugment
from dbnet.random_crop import EastRandomCropData
from dbnet.threshold_map import ThresholdMap
from dbnet.probably_map import ProbablyMap

from torch.utils.data import Dataset
from torchvision import transforms


class DataGenerator(Dataset):

    def __init__(self, data_paths, image_size, img_mode='RGB'):

        self.image_size = image_size
        self.img_mode = img_mode

        self.image_list = []
        self.label_list = []

        print('load dataset from : ', data_paths)

        self.transform = transforms.ToTensor()

        self.image_aug = ImageAugment()
        # 切割为640
        self.image_crop = EastRandomCropData(image_size=image_size, max_tries=50)
        # 训练阶段 概率图和阈值图是处理数据时计算而得
        # 获得图片的thresh_map、prob_map      标签之二
        self.thresh_map = ThresholdMap(thresh_min=0.3, thresh_max=0.7)
        self.prob_map = ProbablyMap()

        # self.load(data_path)
        for path in data_paths:
            self.load(data_path=path)

        print('dataset length: ', len(self.image_list))

# 实验的数据集
    def load(self, data_path):

        # self.image_list = []
        # self.label_list = []

        image_dir = os.path.join(data_path, 'img')
        label_dir = os.path.join(data_path, 'gt')

        image_files = []
        gt_files = []
        for ext in ['jpg', 'jpeg']:
            image_files.extend(glob.glob(os.path.join(image_dir, '*.{}'.format(ext))))
        image_files.sort()

        for ext in ['txt']:
            gt_files.extend(glob.glob(os.path.join(label_dir, '*.{}'.format(ext))))
        gt_files.sort()

        for gt_file in gt_files:

        #     _path, _file = os.path.split(image_file)
        #     _name, _ext = os.path.splitext(_file)
        #
        #     gt_file = os.path.join(label_dir, _name + '.txt')
        #     if not os.path.exists(gt_file):
        #         print(gt_file, 'not exist.')
        #         continue

            text_rects = []
            with open(gt_file, encoding='utf-8', mode='r') as f:
                rect_lines = f.readlines()
                for line in rect_lines:
                    line = line.strip().split(',')
                    line = [item.replace('\ufeff', '') for item in line]
                    rect = [int(v) for v in line[:8]]  # 8个坐标
                    rect = np.array(rect).reshape([-1, 2])
                    text_rects.append(rect)

            if len(text_rects) < 1:
                print('there is no suit bbox in {}'.format(gt_file))
                continue

            # self.image_list.append(image_file)
            self.label_list.append(text_rects)

        for image_file in image_files:
            self.image_list.append(image_file)

    @classmethod
    def draw_contours(cls, image, contours, color, width):
        img = Image.fromarray(image)
        draw = IDraw.ImageDraw(img)

        for contour in contours:
            contour = np.array(contour).reshape([-1, 2]).tolist()

            if len(contour) < 2:
                continue

            point_last = contour[0]
            for i in range(1, len(contour), 1):
                point_current = contour[i]
                draw.line([point_last[0], point_last[1], point_current[0], point_current[1]], color, width)
                point_last = point_current

            if len(contour) > 2:
                draw.line([point_last[0], point_last[1], contour[0][0], contour[0][1]], color, width)

        return np.array(img)

    def debug_outputs(self, index, image, contours, features):

        outpath = './test'
        image_out = os.path.join(outpath, '%d_image.jpg' % index)
        prob_map_out = os.path.join(outpath, '%d_prob_map.jpg' % index)
        prob_mask_out = os.path.join(outpath, '%d_prob_mask.jpg' % index)
        prob_weight_out = os.path.join(outpath, '%d_prob_weight.jpg' % index)
        thresh_map_out = os.path.join(outpath, '%d_thresh_map.jpg' % index)
        thresh_mask_out = os.path.join(outpath, '%d_thresh_mask.jpg' % index)

        prob_map, prob_mask, prob_weight, thresh_map, thresh_mask = features
        prob_map = prob_map * 255
        prob_mask = prob_mask * 255
        prob_weight = prob_weight * 255
        thresh_map = thresh_map * 255
        thresh_mask = thresh_mask * 255

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = self.draw_contours(image, contours, (0, 0, 255), 1)
        image = np.array(image)

        prob_map = np.array(prob_map, dtype=np.uint8)
        # prob_map = cv2.cvtColor(prob_map, cv2.COLOR_GRAY2RGB)
        # prob_map = self.draw_contours(prob_map, text_polys, (0, 0, 255), 1)
        prob_map = np.array(prob_map)

        prob_weight = np.array(prob_weight, dtype=np.uint8)
        # prob_weight = cv2.cvtColor(prob_weight, cv2.COLOR_GRAY2RGB)
        # prob_weight = self.draw_contours(prob_weight, text_polys, (0, 0, 255), 1)
        prob_weight = np.array(prob_weight)

        thresh_map = np.array(thresh_map, dtype=np.uint8)
        # thresh_map = cv2.cvtColor(thresh_map, cv2.COLOR_GRAY2RGB)
        # thresh_map = self.draw_contours(thresh_map, text_polys, (0, 0, 255), 1)
        thresh_map = np.array(thresh_map)

        thresh_mask = np.array(thresh_mask, dtype=np.uint8)
        # thresh_mask = cv2.cvtColor(thresh_mask, cv2.COLOR_GRAY2RGB)
        # thresh_mask = self.draw_contours(thresh_mask, text_polys, (0, 0, 255), 1)
        thresh_mask = np.array(thresh_mask)

        cv2.imwrite(image_out, image)
        cv2.imwrite(prob_map_out, prob_map)
        cv2.imwrite(prob_weight_out, prob_weight)
        cv2.imwrite(prob_mask_out, prob_mask)
        cv2.imwrite(thresh_map_out, thresh_map)
        cv2.imwrite(thresh_mask_out, thresh_mask)

    # 数据输出函数
    def get_image(self, index):
        # 列表存储
        image_path = self.image_list[index]
        contours = self.label_list[index]

        if self.img_mode == 'RGB':
            image = cv2.imread(image_path, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.img_mode == 'GRAY':
            image = cv2.imread(image_path, 0)
        else:
            print('error at self.img_mode')
            raise Exception('error at self.img_mode')

        image, contours = self.image_aug(image, contours)
        image, contours = self.image_crop(image, contours)

        # prob_mask, contours = self.filter_contours(image, contours)
        prob_map, prob_mask, prob_weight, contours = self.prob_map(image, contours)

        thresh_map, thresh_mask = self.thresh_map(image, contours)

        # self.debug_outputs(index, image, contours, (prob_map, prob_mask, prob_weight, thresh_map, thresh_mask))

        image = self.transform(image)

        prob_map = prob_map[np.newaxis, ...].astype(np.float32)
        prob_mask = prob_mask[np.newaxis, ...].astype(np.float32)
        prob_weight = prob_weight[np.newaxis, ...].astype(np.float32)
        thresh_map = thresh_map[np.newaxis, ...].astype(np.float32)
        thresh_mask = thresh_mask[np.newaxis, ...].astype(np.float32)

        return image, prob_map, prob_mask, prob_weight, thresh_map, thresh_mask

    def get_image_s(self, index):
        while True:
            try:
                return self.get_image(index)
            except:
                print('error at get i = ', index)
                index = random.randint(0, len(self.image_list) - 1)
                continue

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        return self.get_image_s(i)
