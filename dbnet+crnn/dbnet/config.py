# -*- coding:utf-8 -*-
class Config:
    def __init__(self):
        # self.learning_rate = 1e-4
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.gpu_id = 0
        self.use_gpu = True

        self.start_epoch = 1
        self.max_epoch = 2000
        self.batch_size = 8
        self.num_workers = 0
        self.image_size = (480, 480)

        # self.save_interval = 2000
        # self.echo_interval = 100
        # self.save_interval = 300
        # self.echo_interval = 50
        self.save_interval = 200
        self.echo_interval = 5

        self.predict_model = './model_weights/dbnet_m3_36_1000.pth'

        self.save_model_dir = './train_weights/'
        self.test_save_model_dir = './dbnet_train_weights/model_fpn'
        #self.train_dir = ['D:/dbnet-crnn/dbnet_train_data/train']
        self.train_dir = ['D:/download_data/det_data_20240412/ofd_dataset_v4.0_piaoju1735']
        self.test_train_dir = ['D:/dbnet-crnn/test_images/db_train_data']

        #self.train_dir = ['E:/text_detect_dataset/ofd_ctpn_split', 'E:/text_detect_dataset/ofd_mon_split']

