import random
import time
import numpy as np
import torchvision.transforms as T
import cv2


def window_crop(img, label=None, fg=None, window_width=(448, 448), anchor_coordinate=(0, 0)):
    h, w, c = img.shape
    # h, w, c = int(h), int(w), int(c)
    window_width_x, window_width_y = window_width
    # window_width_x, window_width_y = int(window_width_x), int(window_width_y)
    anchor_x, anchor_y = anchor_coordinate
    # anchor_x, anchor_y = int(anchor_x), int(anchor_y)
    a_start_x = max(0, -anchor_x)
    a_start_y = max(0, -anchor_y)
    b_start_x = max(0, anchor_x)
    b_start_y = max(0, anchor_y)
    a_end_x = min(window_width_x, a_start_x + h - b_start_x)
    a_end_y = min(window_width_y, a_start_y + w - b_start_y)

    new_img = np.zeros((window_width_x, window_width_y, c), dtype=np.float32)
    new_img[a_start_x:a_end_x, a_start_y:a_end_y, :] = \
        img[b_start_x:b_start_x + (a_end_x - a_start_x), b_start_y:b_start_y + (a_end_y - a_start_y), :]

    new_label = None
    if label is not None:
        new_label = np.zeros((window_width_x, window_width_y), dtype=np.uint8)
        new_label[a_start_x:a_end_x, a_start_y:a_end_y] = \
            label[b_start_x:b_start_x + (a_end_x - a_start_x), b_start_y:b_start_y + (a_end_y - a_start_y)]

    if fg is not None:
        fg_factor = (a_end_x - a_start_x) * (a_end_y - a_start_y) / (h * w)
        fg = [x * fg_factor for x in fg]

    return new_img, new_label, fg


class RandRescaleCrop(object):
    def __init__(self, target_size=448, scale_range=(0.75, 1.1)):
        self.target_size = int(target_size)
        self.min_scale, self.max_scale = scale_range
        assert self.min_scale <= self.max_scale

    def __call__(self, img, label=None, fg=None):
        h, w, c = img.shape
        window_scale = random.uniform(self.min_scale, self.max_scale)
        window_width = int(window_scale * max(h, w))
        min_anchor_y = min(w - window_width, 0)
        max_anchor_y = max(w - window_width, 0)
        min_anchor_x = min(h - window_width, 0)
        max_anchor_x = max(h - window_width, 0)
        anchor_x = int(random.uniform(min_anchor_x, max_anchor_x + 1))
        anchor_y = int(random.uniform(min_anchor_y, max_anchor_y + 1))

        img, label, fg = window_crop(img, label, fg,
                                     window_width=(window_width, window_width),
                                     anchor_coordinate=(anchor_x, anchor_y))

        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        if label is not None:
            label = cv2.resize(label, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        return img, label, fg


class MultiScaleCrop(object):
    def __init__(self, target_size=448, n_scales=3):
        self.target_size = target_size
        assert n_scales in [1, 3, 5, 7]
        self.n_scales = n_scales

    def __call__(self, img):
        h, w, c = img.shape
        edge_l = max(h, w)
        edge_s = min(h, w)
        multi_scale_list = []

        # long edge scale
        anchor_x = min((h - edge_l) // 2, 0)
        anchor_y = min((w - edge_l) // 2, 0)
        cropped_img, _, _ = window_crop(img,
                                        window_width=(edge_l, edge_l),
                                        anchor_coordinate=(anchor_x, anchor_y))
        multi_scale_list.append(
            cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))

        if self.n_scales > 2:  # short edge scale
            cropped_img, _, _ = window_crop(img,
                                            window_width=(edge_s, edge_s),
                                            anchor_coordinate=(0, 0))
            multi_scale_list.append(
                cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))
            anchor_x = max(h - edge_s, 0)
            anchor_y = max(w - edge_s, 0)
            cropped_img, _, _ = window_crop(img,
                                            window_width=(edge_s, edge_s),
                                            anchor_coordinate=(anchor_x, anchor_y))
            multi_scale_list.append(
                cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))

        if self.n_scales > 4:  # half long edge scale
            if self.n_scales == 5:  # 2 samples
                if w > h:
                    anchor_x = (h - edge_l // 2) // 2
                    anchor_y = edge_l // 2
                    cropped_img, _, _ = window_crop(img,
                                                    window_width=(edge_l // 2, edge_l // 2),
                                                    anchor_coordinate=(anchor_x, 0))
                    multi_scale_list.append(
                        cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))
                    cropped_img, _, _ = window_crop(img,
                                                    window_width=(edge_l // 2, edge_l // 2),
                                                    anchor_coordinate=(anchor_x, anchor_y))
                    multi_scale_list.append(
                        cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))
                else:
                    anchor_y = (w - edge_l // 2) // 2
                    anchor_x = edge_l // 2
                    cropped_img, _, _ = window_crop(img,
                                                    window_width=(edge_l // 2, edge_l // 2),
                                                    anchor_coordinate=(0, anchor_y))
                    multi_scale_list.append(
                        cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))
                    cropped_img, _, _ = window_crop(img,
                                                    window_width=(edge_l // 2, edge_l // 2),
                                                    anchor_coordinate=(anchor_x, anchor_y))
                    multi_scale_list.append(
                        cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))

            else:  # 4 samples
                anchor_x = h - edge_l // 2
                anchor_y = w - edge_l // 2
                cropped_img, _, _ = window_crop(img,
                                                window_width=(edge_l // 2, edge_l // 2),
                                                anchor_coordinate=(0, 0))
                multi_scale_list.append(
                    cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))
                cropped_img, _, _ = window_crop(img,
                                                window_width=(edge_l // 2, edge_l // 2),
                                                anchor_coordinate=(anchor_x, 0))
                multi_scale_list.append(
                    cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))
                cropped_img, _, _ = window_crop(img,
                                                window_width=(edge_l // 2, edge_l // 2),
                                                anchor_coordinate=(0, anchor_y))
                multi_scale_list.append(
                    cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))
                cropped_img, _, _ = window_crop(img,
                                                window_width=(edge_l // 2, edge_l // 2),
                                                anchor_coordinate=(anchor_x, anchor_y))
                multi_scale_list.append(
                    cv2.resize(cropped_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR))

        return multi_scale_list


class RandAugment(object):
    def __init__(self,
                 p_flip=0.5,
                 p_blur_sharp=(0.25, 0.25),
                 contrast_scale=(0.75, 1.25),
                 brightness_scale=(-31, 31)):
        self.p_flip = p_flip
        self.p_blur, self.p_sharp = p_blur_sharp
        assert self.p_blur + self.p_sharp <= 1
        self.contrast_low, self.contrast_high = contrast_scale
        assert self.contrast_low <= self.contrast_high
        self.brightness_low, self.brightness_high = brightness_scale
        assert self.brightness_low <= self.brightness_high

    def __call__(self, img, label=None):
        if random.random() < self.p_flip:
            img = cv2.flip(img, 1)
            if label is not None:
                np.flip(label, 0)

        if random.random() < self.p_blur:
            func_list = [lambda _img: cv2.GaussianBlur(_img, (5, 5), .5),
                         lambda _img: cv2.medianBlur(_img, 3),
                         lambda _img: cv2.blur(_img, (3, 3))]
            img = random.choice(func_list)(img)

        elif random.random() * (1 - self.p_blur) < self.p_sharp:
            blur_img = cv2.GaussianBlur(img, (0, 0), 5)
            img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

        contrast_factor = random.uniform(self.contrast_low, self.contrast_high)
        brightness = random.uniform(self.brightness_low, self.brightness_high)
        cv2.convertScaleAbs(img, img, contrast_factor, brightness)
        return img, label


class TransformToTensor(object):
    def __init__(self, target_size=(448, 448), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.target_h, self.target_w = target_size
        self.normalize_img = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        self.normalize_label = T.ToTensor()
        self.resize_img = T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_label = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)

    def __call__(self, img, label=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.normalize_img(img)
        if label is not None:
            label = self.normalize_label(label)

        if img.shape != (self.target_h, self.target_w):
            img = self.resize_img(img)
            if label is not None:
                label = self.resize_label(label)

        return img, label


def main():
    transform = RandRescaleCrop()
    aug = RandAugment()
    img = cv2.imread("color.jpg", cv2.IMREAD_COLOR)
    for i in range(10):
        new_img, _ = aug(img)
        new_img, _, _ = transform(new_img)
        cv2.imwrite("aug_img" + str(i) + ".png", new_img)


if __name__ == '__main__':
    random.seed(time.time())
    main()
