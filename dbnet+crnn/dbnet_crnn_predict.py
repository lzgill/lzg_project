# -*- coding:utf-8 -*-

import os
import random
import time
import numpy as np
import io
import glob
from PIL import Image

from dbnet_wraper import DBNetPredictor
from crnn_wraper import CrnnPredictor


def get_images(image_path):
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
        image_files.extend(glob.glob(
            os.path.join(image_path, '*.{}'.format(ext))))
    image_files.sort()
    return image_files


def main():

    image_dir = './test_images/test_images1'
    label_dir = './test_images'
    result_dir = './test_result/image_test1'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    db_model_file = './dbnet_model_weights/dbnet_mon_m3_91_0500.pth'
    # db_model_file = './train_weights/dbnet_m3_70_0200.pth'
    db_predictor = DBNetPredictor(db_model_file)
    auto_rotation = False

    # crnn_model_file = './model_weights/crnn_m8x8_020_020000.pth'
    # crnn_model_file = './model_weights/crnn_m8n_c3_od_029_020000_s3.pth'
    # crnn_model_file = './model_weights/crnn_m8xp3_015_010000.pth'
    # crnn_model_file = './crnn_train_weights/crnn_m6s_c3_od_055_040000.pth'
    crnn_model_file = './crnn_train_weights/crnn_m8n_c3_od_114_060000.pth'
    # crnn_model_file = './crnn_train_weights/crnn_m8n_c3_od_attention_090_030000.pth'
    # crnn_model_file ='./crnn_train_mcloss_weights/crnn_m8n_c3_od_036_060000.pth'
    char_set_file = './crnn_chs/character_vector_8192s.txt'
    crnn_predictor = CrnnPredictor(crnn_model_file, char_set_file)

    image_files = get_images(image_dir)

    elapsed_time = []
    i = 0
    for image_path in sorted(image_files):

        _path, _file = os.path.split(image_path)
        _name, _ext = os.path.splitext(_file)

        image_fp = open(image_path, 'rb')
        image = np.array(Image.open(io.BytesIO(image_fp.read())).convert('RGB'))

        if auto_rotation:
            image = db_predictor.image_auto_rotation(image)

        # gt file is optional
        gt_path = os.path.join(label_dir, _name + '.txt')  # label_dir
        gt_rects, gt_texts = db_predictor.get_gt_info(gt_path)

        t0 = time.time()
        # 检测结果
        rs_rects = db_predictor.text_detect(image)
        t1 = time.time()

        output_file = os.path.join(result_dir, _file)
        output_feature1_file = os.path.join(result_dir, _name + '_f1.jpg')
        output_feature2_file = os.path.join(result_dir, _name + '_f2.jpg')
        output_feature3_file = os.path.join(result_dir, _name + '_f3.jpg')

        result = crnn_predictor.text_recognize(image, rs_rects, 1)
        # 新增 保存截取的图片
        # crnn_predictor.show_image(image, rs_rects, 1, i)
        i += 1
        t2 = time.time()
        # 画线
        image = db_predictor.draw_contours(image, gt_rects, (0, 0, 255), 1)
        image = db_predictor.draw_contours(image, rs_rects, (255, 0, 0), 1)
        #
        Image.fromarray(image).save(output_file)
        db_predictor.imgf1.save(output_feature1_file)
        db_predictor.imgf2.save(output_feature2_file)
        db_predictor.imgf3.save(output_feature3_file)

        t3 = time.time()

        elapsed_time.append([t1 - t0, len(rs_rects), t2 - t1, t3 - t0])

        label_string = ''
        text_string = ''
        for index, text_ords in result.items():
            region = [str(v) for v in rs_rects[index]]
            score = 1.0
            text = crnn_predictor.decode_text(text_ords)
            # if len(text) < 1:
            #     continue
            ords = [str(ch) for ch in text_ords]
            # result_string += '%s,%1.4f,%s\n' % (','.join(region), score, text)
            label_string += '%s,%1.4f,%s\n' % (','.join(region), score, ' '.join(ords))
            text_string += '%d: %s\n' % (index, text)
            # print(text_ords)

        # if len(result_string) == 0:
        #     result_string = ' '
        print("\nRecognition Result:")
        print(text_string)
        print("%s: complete, it took %.3fs" % (_file, t3 - t0))

    print('total time: ', np.sum(np.array(elapsed_time), axis=0))


if __name__ == '__main__':

    main()
