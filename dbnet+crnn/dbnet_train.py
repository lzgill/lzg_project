# coding=utf-8
import os
import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from dbnet.config import Config
from dbnet.model3l import DBNet
from dbnet.loss import DBLoss
from dbnet.generator1 import DataGenerator


cfg = Config()


def db_train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DBNet()
    criterion = DBLoss()

    model.to(device)
    criterion.to(device)

    # model.load_state_dict(torch.load(cfg.predict_model))

    if not os.path.exists(cfg.save_model_dir):
        os.mkdir(cfg.save_model_dir)

    params_to_uodate = model.parameters()
    optimizer = torch.optim.Adam(params_to_uodate, lr=cfg.learning_rate, eps=1e-7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    for epoch in range(cfg.start_epoch, cfg.max_epoch, 1):

        dataset = DataGenerator(data_paths=cfg.train_dir, image_size=cfg.image_size, img_mode='GRAY')
        # train_loader = DataLoader(dataset=_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
        #                           num_workers=cfg.num_workers)
        train_loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

        start = time.time()
        loss_history = []

        torch.cuda.empty_cache()

        for i, batch in enumerate(train_loader):

            # 数据进行转换和丢到gpu
            for k in range(len(batch)):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            image, prob_map, prob_mask, prob_weight, thresh_map, thresh_mask = batch
            features = prob_map, prob_mask, prob_weight, thresh_map, thresh_mask

            step = i + 1

            pred = model(image)
            loss = criterion([pred, features])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            if step % cfg.echo_interval == 0:
                step_loss = float(loss_history[-1])
                avg_loss = float(np.mean(loss_history))
                # var_loss = float(np.var(loss_history))
                var_loss = float(np.sum(np.abs(np.array(loss_history) - avg_loss))) / len(loss_history)

                fps = (time.time() - start) * 1.0 / cfg.echo_interval
                print('epoch:[%d/%d], step:%d, step_loss=%1.5f, var_loss=%1.5f, avg_loss=%1.5f, time:%1.2f' %
                      (epoch, cfg.max_epoch, step, step_loss, var_loss, avg_loss, fps))
                start = time.time()
                loss_history = []

            if step % cfg.save_interval == 0:
                filename = 'dbnet_m3_%s_%s.pth' % (str(epoch).zfill(2), str(step).zfill(4))
                model_path = os.path.join(cfg.save_model_dir, filename)
                torch.save(model.state_dict(), model_path)
                print('Write model to: {:s}'.format(model_path))

        print('epoch:[%d/%d], scheduler set lr=%1.6f' % (epoch, cfg.max_epoch, optimizer.param_groups[0]['lr']))


if __name__ == '__main__':

    db_train()
