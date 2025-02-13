
import numpy as np
import imgaug


class ImageAugment:
    def __init__(self):

        augment_list = [imgaug.augmenters.Fliplr(p=0.2),
                        imgaug.augmenters.Affine(rotate=[-3, 3]),
                        imgaug.augmenters.Resize(size=[0.5, 3.0]),
                        imgaug.augmenters.GaussianBlur(sigma=(0.0, 1.0)),
                        imgaug.augmenters.Rot90(k=[-1, 0, 0, 0, 1, 1, 1, 2], keep_size=False)]

        self.augmenter = imgaug.augmenters.Sequential(augment_list)

    def __call__(self, image, contours):
        shape = image.shape

        aug = self.augmenter.to_deterministic()
        aug_image = aug.augment_image(image)

        aug_contours = []
        for contour in contours:
            points = [imgaug.Keypoint(p[0], p[1]) for p in contour]
            points = aug.augment_keypoints(imgaug.KeypointsOnImage(points, shape=shape)).keypoints
            contour = [(p.x, p.y) for p in points]
            contour = np.array(contour).reshape([-1, 2])
            aug_contours.append(contour)

        return aug_image, aug_contours
