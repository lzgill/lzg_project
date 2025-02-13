# -*- coding:utf-8 -*-
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# crnn_chs.model8n_attention import CRNN
from crnn_chs.model8n import CRNN
#from crnn_chs.model6s import CRNN
#from crnn_chs.MCloss2 import CRNN
#from crnn_chs.MCloss import CRNN
#from crnn_chs.model_cnn_lstm import CRNN
#from crnn_chs.model_resnet_cnn import CRNN


class CrnnPredictor:
    def __init__(self, model_weights_file, character_vector_file):

        char_set_lines = open(character_vector_file, 'r', encoding='utf-8').readlines()
        self.char_set = [ch.strip(' \n') for ch in char_set_lines]
        self.folder_path = 'D:/dbnet-crnn/test_images/test_images3'
        assert 8192 == len(self.char_set)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(channel=3)

        print('CRNN: Load model from "%s"' % model_weights_file)
        self.model.load_state_dict(torch.load(model_weights_file))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def decode_text(cls, pred_ords):
        predict_text = ''
        for i, pred in enumerate(pred_ords):
            if pred > 0:
                predict_text += chr(pred)

        return predict_text

    @classmethod
    def load_rects(cls, rects_path):

        rs_rects = []
        rs_scores = []

        with open(rects_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            split_line = line.strip().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, split_line[:8])
            score = float(split_line[8])
            rs_rects.append((x1, y1, x2, y2, x3, y3, x4, y4))
            rs_scores.append(score)

        return rs_rects, rs_scores

    def resize(self, image):
        (image_width, image_height) = image.size
        input_height = self.model.input_height    # h=32
        input_width = int(input_height / float(image_height) * image_width)
        input_width = (input_width + 3) // 4 * 4
        image = image.resize([input_width, input_height], Image.BILINEAR)
        return image

    def resize_fixed_width(self, image, fixed_width=960):
        (image_width, image_height) = image.size
        input_height = self.model.input_height
        input_width = int(input_height / float(image_height) * image_width)
        image = image.resize([input_width, input_height], Image.BILINEAR)
        new_image = Image.new('RGB', (fixed_width, input_height), (255, 255, 255))
        # new_image = Image.new('L', (fixed_width, input_height), (255))
        if input_width < 960:
            new_image.paste(image, (0, 0))
        return new_image

    def predict_fixed_width(self, image):
        image = Image.fromarray(image)
        image = self.resize_fixed_width(image)
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
            inputs = self.model.normalize(inputs)
            inputs = inputs.to(self.device)
            pred_list = self.model(inputs=inputs)
            pred_list = pred_list.permute(2, 0, 1)  # [w, b, c] = [sl, bs, hs]
            pred_list = pred_list.contiguous()

        pred_vectors = pred_list.cpu().numpy().argmax(axis=2)  # 最大概率
        pred_vectors = pred_vectors.reshape(-1)

        pred_ords = []
        pred_last = 0
        for pred in pred_vectors:

            if pred > 0 and '' != self.char_set[pred]:
                pred_ord = ord(self.char_set[pred])
            else:
                pred_ord = 0

            if pred_ord > 0 and pred_ord != pred_last:
                pred_ords.append(pred_ord)
            else:
                pred_ords.append(0)

            pred_last = pred_ord

        return pred_ords

    def predict(self, image):
        image = Image.fromarray(image)
        image = self.resize(image)
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
            inputs = self.model.normalize(inputs)
            inputs = inputs.to(self.device)
            pred_list = self.model(inputs=inputs)

            pred_list = pred_list.permute(2, 0, 1)  # [w, b, c] = [sl, bs, hs]
            pred_list = pred_list.contiguous()

        pred_vectors = pred_list.cpu().numpy().argmax(axis=2)  # get the max value
        pred_vectors = pred_vectors.reshape(-1)

        pred_ords = []
        pred_last = 0
        for pred in pred_vectors:  # 挨个打印

            if pred > 0 and '' != self.char_set[pred]:
                pred_ord = ord(self.char_set[pred])
            else:
                pred_ord = 0

            if pred_ord > 0 and pred_ord != pred_last:
                pred_ords.append(pred_ord)
            else:
                pred_ords.append(0)

            pred_last = pred_ord

        return pred_ords

    def font_rec_predict(self, image):
        image = Image.fromarray(image)
        image = self.resize(image)
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
            inputs = self.model.normalize(inputs)
            inputs = inputs.to(self.device)
            pred_list = self.model(inputs=inputs)

            pred_list = pred_list.permute(2, 0, 1)  # [w, b, c] = [sl, bs, hs]
            pred_list = pred_list.contiguous()

        pred_vectors = pred_list.cpu().numpy().argmax(axis=2)  # get the max value
        pred_vectors = pred_vectors.reshape(-1)

        pred_ords = []
        pred_last = 0
        for pred in pred_vectors:  # 挨个打印

            if pred > 0 and '' != self.char_set[pred]:
                pred_ord = ord(self.char_set[pred])
            else:
                pred_ord = 0

            if pred_ord > 0 and pred_ord != pred_last:
                pred_ords.append(pred_ord)
            else:
                pred_ords.append(0)

            pred_last = pred_ord

        return pred_ords



    @classmethod
    def dump_rotated_rect(cls, image, pt1, pt2, pt3, pt4):

        radian1 = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        radian2 = math.atan2(pt3[1] - pt4[1], pt3[0] - pt4[0])
        radian = (radian1 + radian2) / 2.0

        height, width = image.shape[:2]
        new_height = width * math.fabs(math.sin(radian)) + height * math.fabs(math.cos(radian))
        new_width = height * math.fabs(math.sin(radian)) + width * math.fabs(math.cos(radian))
        new_height = int(new_height)
        new_width = int(new_width)

        rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2,), math.degrees(radian), 1)
        image_rotation = cv2.warpAffine(image, rotation_matrix, (new_width, new_height,))

        new_pt1 = np.dot(rotation_matrix, np.array([pt1[0], pt1[1], 1]))
        new_pt3 = np.dot(rotation_matrix, np.array([pt3[0], pt3[1], 1]))
        x1, y1 = int(new_pt1[0]), int(new_pt1[1])
        x3, y3 = int(new_pt3[0]), int(new_pt3[1])

        max_rows, max_cols = image_rotation.shape[:2]
        image_roi = image_rotation[max(1, y1):min(max_rows - 1, y3), max(1, x1):min(max_cols - 1, x3)]
        return image_roi

    @classmethod
    def dump_rect(cls, image, pt1, pt2, pt3, pt4):
        # 最大外接矩形
        x1, y1 = min(pt1[0], pt4[0]), min(pt1[1], pt2[1])
        x3, y3 = max(pt3[0], pt2[0]), max(pt3[1], pt4[1])
        # 在原图像取出要输入的图像
        image_roi = image[y1:y3, x1:x3]
        return image_roi

    def text_recognize(self, image, text_recs, extend):


        results = {}
        height, width = image.shape[0], image.shape[1]

        for index, rect in enumerate(text_recs):

            pt1 = (max(0, rect[0, 0] - extend), max(0, rect[0, 1] - extend))
            pt2 = (min(width, rect[1, 0] + 2 * extend), max(0, rect[1, 1] - extend))
            pt3 = (min(width, rect[2, 0] + 2 * extend), min(height, rect[2, 1] + 2 * extend))
            pt4 = (max(0, rect[3, 0] - extend), min(height, rect[3, 1] + 2 * extend))

            # 图像倾斜角度
            image_roi = self.dump_rect(image, pt1, pt2, pt3, pt4)

            # minimal roi linit
            if image_roi.shape[0] < 5 or image_roi.shape[1] < 5:
                continue

            # height < width * 2 limit
            if image_roi.shape[1] * 2 < image_roi.shape[0]:
                continue

            if self.model.fixed_width:
                text_ords = self.predict_fixed_width(image_roi)
                # raise ValueError('Unimplemented fixed_width = True')
            else:
                text_ords = self.predict(image_roi)

            results[index] = text_ords
            # text = self.decode_text(text_ords)
            # if len(text) > 0:
            #     results[index].append(text)     # 识别文字

        return results

# 存储检测到的文本截取图片
    def show_image(self, image, text_recs, extend, i):

        os.makedirs(self.folder_path, exist_ok=True)

        height, width = image.shape[0], image.shape[1]

        for index, rect in enumerate(text_recs):

            pt1 = (max(0, rect[0, 0] - extend), max(0, rect[0, 1] - extend))
            pt2 = (min(width, rect[1, 0] + 2 * extend), max(0, rect[1, 1] - extend))
            pt3 = (min(width, rect[2, 0] + 2 * extend), min(height, rect[2, 1] + 2 * extend))
            pt4 = (max(0, rect[3, 0] - extend), min(height, rect[3, 1] + 2 * extend))

            # 图像倾斜角度 截取
            image_roi = self.dump_rect(image, pt1, pt2, pt3, pt4)

            # minimal roi linit
            if image_roi.shape[0] < 5 or image_roi.shape[1] < 5:
                continue

            # height < width * 2 limit
            if image_roi.shape[1] * 2 < image_roi.shape[0]:
                continue
            image_roi = Image.fromarray(image_roi)
            image_roi = self.resize(image_roi)

            file_path = os.path.join(self.folder_path, 'image_'+str(i).zfill(3)+'_'+str(index)+'.png')
            image_roi.save(file_path, 'PNG')
            # if self.model.fixed_width:
            #     text_ords = self.predict_fixed_width(image_roi)
            #     #raise ValueError('Unimplemented fixed_width = True')
            # else:
            #     text_ords = self.predict(image_roi)
            #
            # results[index] = text_ords
            # text = self.decode_text(text_ords)
            # if len(text) > 0:
            #     results[index].append(text)     # 识别文字

            # plt.imshow(image_roi)
            # cv2.waitKey(0)
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()
