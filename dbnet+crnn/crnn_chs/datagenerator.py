import os
import random
import glob
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms
import torch.utils.data


def ray(image, theta=0.001):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + theta * random.random() * random.choice([-1, 1]))
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def jitter(image, theta=0.01):
    w, h, _ = image.shape
    out = image.copy()
    if h > 10 and w > 10:
        s = int(random.random() * min(w, h) * theta)
        for i in range(s):
            out[i:, i:, :] = image[:w - i, :h - i, :]
    return out



def noise(image, alpha=1.0, mean=0.0, std=5.0):
    out = image + alpha * np.random.normal(mean, std, image.shape)
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def aug(image):
    image = ray(image, theta=0.001)
    image = jitter(image, theta=0.001)
    image = noise(image, alpha=1.5, mean=0, std=4.0)
    return image


def clip_image(image, rect):

    batch_size = image.size(0)
    batch_channel = image.size(1)
    batch_height = image.size(2)
    batch_width = image.size(3)

    rect = rect.numpy()

    clip_width = 0
    for i in range(len(rect)):
        x1 = rect[i, 0]
        x3 = rect[i, 2]
        clip_width = max(clip_width, x3 - x1)

    image_clip = np.zeros([batch_size, batch_channel, batch_height, clip_width], dtype=np.float32)

    for i in range(len(rect)):
        x1 = rect[i, 0]
        x3 = rect[i, 2]
        clip_x1 = random.randint(max(0, x3 - clip_width), min(x1, batch_width - clip_width))
        clip_x3 = clip_x1 + clip_width
        image_clip[i] = image[i, :, :, clip_x1:clip_x3]

    return torch.tensor(image_clip)


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, char_vector_file, datasets_dir, batch_size, image_size, image_mode='L'):
        self.image_mode = image_mode
        self.image_index = -1
        self.image_shuffle = []
        self.image_path = []
        self.image_name = []

        self.pack_shuffle = []
        self.pack_image = []
        # self.pack_box = []
        # self.pack_label = []
        self.pack_font = []
        # self.pack_rect = []
        # self.pack_length = []
        self.pack_crnn_out = []

        self.char_vector_dict = {}
        self.image_size = image_size
        self.batch_size = batch_size

        self.load_char_vector(char_vector_file)

        for _dataset_dir in datasets_dir:
            self.load_dataset(_dataset_dir)

        self.to_tensor = transforms.ToTensor()

        print('dataset length: ', len(self.image_name) * 512)

    def shuffle(self):
        random.shuffle(self.image_name)

    @classmethod
    def get_images(cls, image_path):
        image_files = []
        for ext in ['jpg']:
            image_files.extend(glob.glob(
                os.path.join(image_path, '*.{}'.format(ext))))
        image_files.sort()
        return image_files

    def load_char_vector(self, char_vector_file):
        self.char_vector_dict = {0: 0}
        with open(char_vector_file, 'r', encoding='utf-8') as f:
            text_lines = f.readlines()
            for i, text in enumerate(text_lines):
                if text.strip() == '':
                    continue
                self.char_vector_dict[ord(text[0])] = i

    def load_dataset(self, dataset_dir):
        image_dir = os.path.join(dataset_dir, 'image')
        # text_dir = os.path.join(dataset_dir, 'text')
        # box_dir = os.path.join(dataset_dir, 'box')
        font_dir = os.path.join(dataset_dir, 'font')

        image_files = self.get_images(image_dir)
        for image_file in image_files:
            _path, _file = os.path.split(image_file)
            _name, _ext = os.path.splitext(_file)

            # text_file = os.path.join(text_dir, _name + '.txt')
            # if not os.path.exists(text_file):
            #     print('file not exist: ', text_file)
            #     continue
            #
            # box_file = os.path.join(box_dir, _name + '.txt')
            # if not os.path.exists(box_file):
            #     print('file not exist: ', box_file)
            #     continue

            font_file = os.path.join(font_dir, _name + '.txt')
            if not os.path.exists(font_file):
                print('file not exist: ', font_file)
                continue
            # exists_unrecognized_chars = False
            # text_packs = np.loadtxt(gt_file, dtype=np.int).reshape(-1)
            #
            # for _ord in text_packs:
            #     if _ord not in self.char_vector_dict:
            #         exists_unrecognized_chars = True
            #         print('%s: char ord "%s %s", unrecognized' % (gt_file, str(_ord), chr(_ord)))
            #
            # if exists_unrecognized_chars:
            #     continue

            self.image_path.append(dataset_dir)
            self.image_name.append(_name)

        self.image_shuffle = np.arange(len(self.image_name))
        random.shuffle(self.image_shuffle)

        self.image_index = -1

        self.pack_shuffle = []
        self.pack_image = []
        # self.pack_box = []
        # self.pack_label = []
        self.pack_font = []
        # self.pack_length = []
        self.pack_crnn_out = []

    def pack_8x64_read(self, save_path, pack_name):

        pack_rows = 64
        pack_cols = 8
        pack_image = []
        # pack_label = []
        # pack_box = []
        pack_font = []
        # pack_rect = []
        # pack_length = []
        pack_crnn_out = []

        image_file = os.path.join(save_path, 'image', pack_name + '.jpg')
        text_file = os.path.join(save_path, 'text', pack_name + '.txt')
        # box_file = os.path.join(save_path, 'box', pack_name + '.txt')
        font_file = os.path.join(save_path, 'font', pack_name + '.txt')
        # rect_file = os.path.join(save_path, 'rect', pack_name + '.txt')

        size_w, size_h = self.image_size

        image = Image.open(image_file).convert(self.image_mode)

        (image_w, image_h) = image.size
        #assert image_w == size_w * pack_cols
        #assert image_h == size_h * pack_rows

        # image
        for i in range(64):
            for j in range(8):
                image_i = image.crop((j * size_w, i * size_h, (j + 1) * size_w, (i + 1) * size_h))#分割
                pack_image.append(image_i)

        pack_font_id = []
        pack_text=[]
        _text_cid=[]
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # read texts
            for i in range(pack_rows):
                texts = lines[i].split(',')
                for j in range(pack_cols):
                    text = [int(v) for v in texts[j].split(' ')]
                    pack_text.append(text)

        return pack_image, pack_text

    def __getitem__(self, index):

        image_index = index // 512
        item_index = index % 512

        image_index = self.image_shuffle[image_index]

        if image_index != self.image_index:
            self.image_index = image_index

            self.pack_shuffle = []
            self.pack_image = []
            self.pack_font = []
            self.pack_crnn_out = []

            image_path = self.image_path[image_index]
            image_name = self.image_name[image_index]

            self.pack_shuffle = np.arange(0, 512)
            random.shuffle(self.pack_shuffle)

            self.pack_image, self.pack_font = self.pack_8x64_read(image_path, image_name)

        item_index = self.pack_shuffle[item_index]

        image = self.to_tensor(self.pack_image[item_index])

        font = np.array(self.pack_font[item_index])
        return image, font

    def __len__(self):
        return len(self.image_name) * 512


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    datagen = DataGenerator(
        datasets_dir=['D:/font_recognition/dump_color/blc'],
        image_size=(960, 32),
        batch_size=4,
        char_vector_file='./character_vector_8192.txt',
    )

    train_loader = DataLoader(datagen, batch_size=4, shuffle=False, num_workers=0)
    for i, (img, label) in enumerate(train_loader):
        img = img[0]
        # print(crnn[0, :, 10])
        print(img.shape)
        # img = transforms.ToPILImage()(img)
        # print(crnn.shape)
        # print(label)
        print(label[0])
        # plt.imshow(img, cmap='gray')
        # plt.show()
        break