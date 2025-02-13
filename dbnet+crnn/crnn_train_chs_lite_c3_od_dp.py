import os
import time
import numpy as np
import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader, ConcatDataset

from crnn_chs.dataset_lmdb import DataGenerator
from crnn_chs.dataset_lmdb import clip_image
from crnn_chs.model8n_cam import CRNN


class Config:
    def __init__(self):
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.num_workers = 8

        self.start_epoch = 1

        self.max_epoch = 120
        self.batch_size = 256

        self.image_size = (960, 32,)
        self.image_mode = 'RGB'

        self.save_interval = 5000
        self.echo_interval = 200
        # self.save_interval = 10000
        # self.echo_interval = 500

        self.character_vector_class = 8192
        self.character_vector_file = './crnn_chs/character_vector_8192s.txt'
        self.predict_model = './crnn_model_weights/crnn_m8n_c3_od_023_015000.pth'

        self.save_model_dir = './train_weights/'
        self.save_prefix = 'crnn_m8n_c3_od_'
        self.train_datasets_dir = ['D:/text_recognition/rec_test_lmdb_out']
        # self.train_datasets_dir2 = ['D:/text_recognition/rec_test_char_lmdb_out']


def main(device_ids: list):
    cfg = Config()

    # device = torch.device("cuda:0" if torch.cuda.is_available() and cfg.use_gpu else "cpu")

    channel = 3 if cfg.image_mode == 'RGB' else 1
    model = CRNN(channel=channel)
    # 更改第一个点
    # model.load_state_dict(torch.load(cfg.predict_model))
    model = torch.nn.DataParallel(model.cuda(), device_ids, output_device=device_ids[0]) # device_ids
    # model = model.to(device)
    model.train()

    if not os.path.exists(cfg.save_model_dir):
        os.mkdir(cfg.save_model_dir)

    ctc_loss = CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    ctc_loss = ctc_loss.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)

    start = time.time()
    loss_history = []

    for epoch in range(cfg.start_epoch, cfg.max_epoch, 1):

        train_generator = DataGenerator(char_vector_file=cfg.character_vector_file,
                                        dataset_dirs=cfg.train_datasets_dir,
                                        image_size=cfg.image_size,
                                        image_mode=cfg.image_mode)

        # train_generator2 = DataGenerator(char_vector_file=cfg.character_vector_file,
        #                                 dataset_dirs=cfg.train_datasets_dir2,
        #                                 image_size=cfg.image_size,
        #                                 image_mode=cfg.image_mode)

        # cat_train_generator = ConcatDataset([train_generator, train_generator2])
        #
        train_loader = DataLoader(train_generator, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        step = 0
        for i, (image, label, box, length) in enumerate(train_loader):

            # image = clip_image(image, box)

            # image = image.to(device)
            image = image.cuda()
            step = step + 1
            output = model(image)  # torch.Size([64, 8192, 120])

            output = output.permute(2, 0, 1)     # [w, b, c] = [sl, bs, hs]

            output = output.contiguous()

            output = output.log_softmax(2)

            input_lengths = torch.full([output.size(1)], output.size(0), dtype=torch.long)
            target_lengths = length

            # if ((output.size() != torch.Size([120, cfg.batch_size, 8192]) or
            #         label.size() != torch.Size([cfg.batch_size, 128]) or
            #         input_lengths.size() != torch.Size([cfg.batch_size])) or
            #         target_lengths.size() != torch.Size([cfg.batch_size])):

            #     print('error at size check.')
            #     print('output.size(): ', output.size())
            #     print('label.size(): ', label.size())
            #     print('input_lengths.size(): ', input_lengths.size())
            #     print('target_lengths.size(): ', target_lengths.size())
            #     exit(0)

            loss = ctc_loss(log_probs=output, targets=label, input_lengths=input_lengths, target_lengths=target_lengths)

            loss_out = float(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss_out)
#
            if step % cfg.echo_interval == 0:
                step_loss = float(loss_history[-1])
                avg_loss = float(np.mean(loss_history))
                # var_loss = float(np.var(loss_history))
                var_loss = float(np.sum(np.abs(np.array(loss_history) - avg_loss))) / len(loss_history)

                fps = (time.time() - start) * 1.0 / cfg.echo_interval
                print('epoch: [%d/%d], step: %d, step_loss=%1.5f, var_loss=%1.5f, avg_loss=%1.5f, time:%1.2f' %
                      (epoch, cfg.max_epoch, step, step_loss, var_loss, avg_loss, fps))
                start = time.time()
                loss_history = []

                torch.cuda.empty_cache()
                # torch.cuda.init()

            if step % cfg.save_interval == 0:
                filename = cfg.save_prefix + '%s_%s.pth' % (str(epoch).zfill(3), str(step).zfill(6))
                model_path = os.path.join(cfg.save_model_dir, filename)
                torch.save(model.module.state_dict(), model_path)
                print('Write model to: {:s}'.format(model_path))

        # reduce learn rate
        # scheduler.step(epoch)
        print('epoch:[%d/%d], scheduler set lr=%1.5f' % (epoch, cfg.max_epoch, optimizer.param_groups[0]['lr']))


def dataset_read_test(device_ids: list):

    cfg = Config()

    start = time.time()

    for epoch in range(cfg.start_epoch, cfg.max_epoch, 1):

        train_generator = DataGenerator(char_vector_file=cfg.character_vector_file,
                                        dataset_dirs=cfg.train_datasets_dir,
                                        image_size=cfg.image_size,
                                        image_mode=cfg.image_mode)

        train_loader = DataLoader(train_generator, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        step = 0
        for i, (image, label, box, length) in enumerate(train_loader):

            # image = clip_image(image, box)

            image = image.cuda()
            label = label.cuda()
            # box = box.cuda()
            length = length.cuda()
            step = step + 1

            if step % cfg.echo_interval == 0:
                step_loss = float(0)
                avg_loss = float(0)
                var_loss = float(0)

                fps = (time.time() - start) * 1.0 / cfg.echo_interval
                print('epoch: [%d/%d], step: %d, step_loss=%1.5f, var_loss=%1.5f, avg_loss=%1.5f, time:%1.2f' %
                      (epoch, cfg.max_epoch, step, step_loss, var_loss, avg_loss, fps))
                start = time.time()

                torch.cuda.empty_cache()
                # torch.cuda.init()

if __name__ == '__main__':

    device_count = torch.cuda.device_count()
    # 获取cuda可用设备列表
    # torch.cuda.is_available()
    # torch.cuda.current_device()

    cuda_device_names = []
    cuda_device_ids = []
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        cuda_device_names.append(device_name)
        cuda_device_ids.append(i)
    print('cuda_devices: ', cuda_device_names)

    main(cuda_device_ids)
    # dataset_read_test(cuda_device_ids)

