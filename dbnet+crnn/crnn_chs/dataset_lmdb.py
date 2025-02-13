
import io
import os.path
import random
import pickle
import lmdb
import cv2
import numpy as np
import torch.utils.data
from PIL import Image


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
    def __init__(self, char_vector_file, dataset_dirs, image_size, image_mode='L'):
        self.image_mode = image_mode
        self.image_3d = False

        self.lmdb_env = None
        self.lmdb_path = dataset_dirs[0]
        self.lmdb_keys = []
        self.lmdb_open = False

        self.char_vector_dict = {}
        self.image_size = image_size

        self.load_char_vector(char_vector_file)

        self.load_keys()
        # self.load_dataset()

    def load_char_vector(self, char_vector_file):
        self.char_vector_dict = {ord(' '): 1}
        with open(char_vector_file, 'r', encoding='utf-8') as f:
            text_lines = f.readlines()
            for i, text in enumerate(text_lines):
                if text.strip() == '':
                    continue
                self.char_vector_dict[ord(text[0])] = i

    def load_keys(self):
        keys_file = os.path.join(self.lmdb_path, 'keys.pkl')
        if os.path.exists(keys_file):
            print('load keys from file: ', keys_file)
            with open(keys_file, 'rb') as f:
                self.lmdb_keys = pickle.load(f)
        else:
            lmdb_env = lmdb.open(self.lmdb_path, readonly=True)
            print('load keys from lmdb: ', self.lmdb_path)
            with lmdb_env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    key = pickle.loads(key)
                    # value = pickle.loads(value)
                    self.lmdb_keys.append(key)
                cursor.close()
        print('dataset length: ', len(self.lmdb_keys))

    def load_dataset(self):
        if not self.lmdb_open:
            print('load dataset from ', self.lmdb_path)
            # self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True)
            self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, meminit=False, map_size=1024*1024*1024*60)
            self.lmdb_open = True

    def split_32x32(self, image, step=8):
        image_slice = []

        c, rows, cols = image.shape

        for k1 in range(0, rows, step):
            for k2 in range(0, cols, step):
                if k1 <= rows - 32 and k2 <= cols - 32:
                    t = image[:, k1:k1+32, k2:k2+32]
                    image_slice.append(t)

        image = np.array(image_slice)
        image = image.transpose([1, 0, 2, 3])
        return image

    def __getitem__(self, index):

        self.load_dataset()

        if index < len(self.lmdb_keys):
            key = self.lmdb_keys[index]
        else:
            key = self.lmdb_keys[index % len(self.lmdb_keys)]

        with self.lmdb_env.begin(write=False) as txn:
            value = txn.get(pickle.dumps(key))
            value = pickle.loads(value)
            image_io_data, text, box, font, rect, crnn = value
            image_io = io.BytesIO(image_io_data)
            image = Image.open(image_io).convert(self.image_mode)
            image = np.array(image)
            # file_bytes = np.asarray(bytearray(image_io.read()), dtype=np.uint8)
            # image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if self.image_mode == 'L':
            image = np.expand_dims(image, axis=0)
        else:
            image = image.transpose([2, 0, 1])

        if self.image_3d:
            image = self.split_32x32(image, 16)
            image = np.stack(image, axis=0)

        # normalize
        # image = self.to_tensor(image)
        image = image / 255.0
        image = image.astype(np.float32)

        _text_cid = []
        for v in text:
            if v in self.char_vector_dict:
                _text_cid.append(self.char_vector_dict[v])
            else:
                _text_cid.append(0)
        text = _text_cid

        length = len(text)
        # 修改128->120
        if length <= 128:
            for _ in range(128 - length):
                text.append(0)
        else:
            print('text length > 128')
            # 修改128->120
            text = text[:128]
            _length = 128
            # text = text[:120]
            # _length = 120

        text = np.array(text)
        # text = torch.tensor(text, dtype=torch.int64).view(120, 1)

        box = np.array(box)
        length = length

        return image, text, box, length

    def __len__(self):
        return len(self.lmdb_keys)
