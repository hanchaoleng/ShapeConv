import logging

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.cm


def visualize_seg(label_map, color_map, num_cls, one_hot=False):
    if one_hot:
        label_map = np.argmax(label_map, axis=-1)
    if label_map.ndim == 4 and label_map.shape[3] == 1:
        label_map = label_map.squeeze(axis=-1)
    if label_map.ndim == 2:
        label_map = np.expand_dims(label_map, axis=0)
    out = np.zeros((label_map.shape[0], label_map.shape[1], label_map.shape[2], 3))

    # class 255 is rgb(0,0,0)
    for l in range(0, num_cls):
        out[label_map == l, :] = color_map[l]
    return out


def visualize_seg_with_zero(pred, label_gt, color_map, num_cls, one_hot=False):
    if one_hot:
        pred = np.argmax(pred, axis=-1)
        label_gt = np.argmax(label_gt, axis=-1)
    out = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 3))

    # class 0 is rgb(0,0,0)
    for l in range(1, num_cls):
        out[pred == l, :] = color_map[l]
    out[label_gt == 0, :] = color_map[0]
    return out


def get_color_map(color_num):
    color_map = []
    fun_tab20 = mpl.cm.get_cmap('tab20')
    for i in range(20):
        color = fun_tab20(i)
        color_map.append([color[0], color[1], color[2]])
    fun_tab20b = mpl.cm.get_cmap('tab20b')
    for i in range(20):
        color = fun_tab20b(i)
        color_map.append([color[0], color[1], color[2]])
    if color_num > len(color_map):
        logging.error("Color num too big!")
        return None
    return np.array(color_map)[:color_num]


def get_voc_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    pallete = np.reshape(pallete, (n, 3))
    return pallete


def get_custom_color_map(path):
    return np.loadtxt(path)


def show_color_map(color_map, color_names):
    color_map = (color_map * 255).astype(int)
    y_start = 50
    rect_w = 240
    rect_h = 40
    canvas = np.ones((rect_h * len(color_map)+120, rect_w * 2, 3), dtype="uint8") * 255

    for i, color in enumerate(color_map):
        print(list(color), color.dtype, color[1])
        cv2.rectangle(canvas, (0, y_start + rect_h * i), (rect_w, y_start + rect_h * (i + 1)),
                      (int(color[2]), int(color[1]), int(color[0])), -1)
        cv2.putText(canvas, color_names[i], (5, y_start + 30 + rect_h * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)


def test():
    classes = ['background',
               'wall',
               'floor',
               'cabinet',
               'bed',
               'chair',
               'sofa',
               'table',
               'door',
               'window',
               'bookshelf',
               'picture',
               'counter',
               'blinds',
               'desk',
               'shelves',
               'curtain',
               'dresser',
               'pillow',
               'mirror',
               'floor mat',
               'clothes',
               'ceiling',
               'books',
               'refridgerator',
               'television',
               'paper',
               'towel',
               'shower curtain',
               'box',
               'whiteboard',
               'person',
               'night stand',
               'toilet',
               'sink',
               'lamp',
               'bathtub',
               'bag',
               'otherstructure',
               'otherfurniture',
               'otherprop']
    show_color_map(get_color_map(len(classes)), classes)


if __name__ == '__main__':
    test()
