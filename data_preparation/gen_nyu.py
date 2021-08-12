import getopt
import sys
import numpy as np
import h5py
import os
from PIL import Image
import scipy.io as scio
from data_preparation.utils.rgbd_util import getHHA


def save_imgs(f_h5, dir_img_out):
    if not os.path.isdir(dir_img_out):
        os.makedirs(dir_img_out)

    images = np.array(f_h5["images"])
    for i, a in enumerate(images):
        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_path = os.path.join(dir_img_out, "%06d.png" % (i+1))
        img.save(img_path, optimize=True)

        print('rgb', i)
        # break


def save_depth(f_h5, dir_depth_out):
    if not os.path.isdir(dir_depth_out):
        os.makedirs(dir_depth_out)

    depths = np.array(f_h5["depths"])
    # depths_f1 = depths.flatten()
    depths *= 1000
    # depths_f2 = depths.flatten()
    # print(1, max(depths_f1))
    # print(2, max(depths_f2))
    # return
    for i, depth in enumerate(depths):
        depth = depth.transpose((1, 0))
        # depth_f1 = depth.flatten()
        # print(1, max(depth_f1))
        dep_img = Image.fromarray(np.uint32(depth))
        dep_path = os.path.join(dir_depth_out, "%06d.png" % (i+1))
        dep_img.save(dep_path, 'PNG', optimize=True)

        print('depth', i)
        # break


def save_hha(f_h5, camera_mats, dir_hha_out):
    if not os.path.isdir(dir_hha_out):
        os.makedirs(dir_hha_out)

    depths = np.array(f_h5["depths"])
    for i, depth in enumerate(depths):
        depth = depth.transpose((1, 0))
        # print('depth_min', depth.min())
        # print('depth_max', depth.max())
        # print(camera_mats[i])
        hha = getHHA(camera_mats[i], depth, depth)
        # print(hha.min(), hha.max())
        hha_img = Image.fromarray(hha)
        hha_path = os.path.join(dir_hha_out, "%06d.png" % (i+1))
        hha_img.save(hha_path, 'PNG', optimize=True)

        print('hha', i)
        # break


def save_labels(f_h5, maps, dir_label_out):
    if not os.path.isdir(dir_label_out):
        os.makedirs(dir_label_out)

    labels = np.array(f_h5["labels"])
    for i, label in enumerate(labels):
        if maps:
            for map in maps:
                label = np.vectorize(map.get)(label)
        label = label.transpose((1, 0))
        label_img = Image.fromarray(np.uint8(label))
        label_path = os.path.join(dir_label_out, "%06d.png" % (i+1))
        label_img.save(label_path, 'PNG', optimize=True)

        print('label', i)
        # break


def read_label_map(path_map):
    f_map = scio.loadmat(path_map)
    if 'classMapping13' in f_map.keys():
        map_class = np.array(f_map['classMapping13'][0][0][0])[0]
    else:
        map_class = f_map['mapClass'][0]
    print(map_class.shape)
    dict_map = {0: 0}
    for ori_id, mapped_id in enumerate(map_class):
        dict_map[ori_id + 1] = mapped_id
    return dict_map


def get_nyu_cam_mats(path_nyu_cam_mats):
    mats = []
    with open(path_nyu_cam_mats, 'r') as f_cam:
        lines = f_cam.readlines()
        for line_id in range(0, len(lines), 4):
            # this camera paras from sun_rgbd's nyu_v2
            # paras = "518.857901 0.000000 284.582449 0.000000 519.469611 208.736166 0.000000 0.000000 1.000000"
            # numbers = np.array([paras.split(' ')[:9]]).astype(float)
            # mat = np.reshape(numbers, [3, 3], 'C')
            mat = np.zeros([3, 3], dtype=np.float)
            for i in range(3):
                line = lines[line_id + i]
                eles = line.split(' ')
                for j in range(len(eles)):
                    if i == 2:
                        mat[i][j] = float(eles[j])
                    else:
                        mat[i][j] = float(eles[j]) * 1000
            mats.append(mat)
    return mats

def save_list(pth_splits, pth_train, pth_test):
    def write_txt(f_list, list_ids):
        f_list.write('\n'.join(list_ids))
        f_list.close()

    train_test = scio.loadmat(pth_splits)
    train_images = tuple([int(x) for x in train_test["trainNdxs"]])
    test_images = tuple([int(x) for x in train_test["testNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    train_ids = ["%06d" % i for i in train_images]
    test_ids = ["%06d" % i for i in test_images]

    train_list_file = open(pth_train, 'w')
    write_txt(train_list_file, train_ids)

    test_list_file = open(pth_test, 'w')
    write_txt(test_list_file, test_ids)

def main(dir_meta, dir_out):
    path_nyu_depth_v2_labeled = os.path.join(dir_meta, "nyu_depth_v2_labeled.mat")
    f = h5py.File(path_nyu_depth_v2_labeled)
    print(f.keys())

    dir_sub_out = os.path.join(dir_out, 'image')
    save_imgs(f, dir_sub_out)

    dir_sub_out = os.path.join(dir_out, 'depth')
    save_depth(f, dir_sub_out)

    pth_cam_mats = os.path.join(dir_meta, "camera_rotations_NYU.txt")
    dir_sub_out = os.path.join(dir_out, 'hha')
    save_hha(f, get_nyu_cam_mats(pth_cam_mats), dir_sub_out)

    pth_map_label40 = os.path.join(dir_meta, "classMapping40.mat")
    dir_sub_out = os.path.join(dir_out, 'label40')
    save_labels(f, [read_label_map(pth_map_label40)], dir_sub_out)

    # pth_map_label13 = os.path.join(dir_meta, "class13Mapping.mat")
    # dir_sub_out = os.path.join(dir_out, 'label13')
    # save_labels(f, [read_label_map(pth_map_label40), read_label_map(pth_map_label13)], dir_sub_out)

    pth_splits = os.path.join(dir_meta, "splits.mat")
    pth_train = os.path.join(dir_out, 'train.txt')
    pth_test = os.path.join(dir_out, 'test.txt')
    save_list(pth_splits, pth_train, pth_test)


if __name__ == '__main__':
    input_dir = ""
    output_dir = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["input_meta_dir=", "output_dir="])
    except getopt.GetoptError:
        print('gen_nyu.py -i <input_meta_dir> -o <output_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('gen_nyu.py -i <input_meta_dir> -o <output_dir>')
            sys.exit()
        elif opt in ("-i", "--input_meta_dir"):
            input_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg

    main(input_dir, output_dir)


