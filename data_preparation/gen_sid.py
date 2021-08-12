import getopt
import shutil
import sys
import numpy as np
import os
from PIL import Image
import scipy.io as scio
import cv2
import json
from multiprocessing import Process
from data_preparation.utils.rgbd_util import getHHA

train_areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
test_areas = ['area_5a', 'area_5b']
is_test = False


def save_imgs(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_img_in = os.path.join(dir_in, area, 'data', 'rgb')
        dir_img_out = os.path.join(dir_out, area, 'image')
        if not os.path.isdir(dir_img_out):
            os.makedirs(dir_img_out)

        path_img_in = os.path.join(dir_img_in, name + '_rgb.png')
        path_img_out = os.path.join(dir_img_out, name + '.png')
        shutil.copyfile(path_img_in, path_img_out)

        print('img', i, area_name)
        if is_test:
            break


def save_depth(dir_in, area_names, dir_out):
    max_depth = 0
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_dep_in = os.path.join(dir_in, area, 'data', 'depth')
        dir_dep_out = os.path.join(dir_out, area, 'depth')
        if not os.path.isdir(dir_dep_out):
            os.makedirs(dir_dep_out)

        path_dep_in = os.path.join(dir_dep_in, name + '_depth.png')
        path_dep_out = os.path.join(dir_dep_out, name + '.png')
        shutil.copyfile(path_dep_in, path_dep_out)
        # dep = cv2.imread(path_dep_in, cv2.IMREAD_UNCHANGED)
        # dep += 1    # 65535 -> 0
        # max_depth = max(max_depth, dep.flatten().max())
        # cv2.imwrite(path_dep_out, dep)
        # print('max_depth', max_depth)

        print('dep', i, area_name)
        if is_test:
            break


def save_hha(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_dep_in = os.path.join(dir_in, area, 'data', 'depth')
        dir_hha_out = os.path.join(dir_out, area, 'hha')
        if not os.path.isdir(dir_hha_out):
            os.makedirs(dir_hha_out)

        path_mat = os.path.join(dir_in, area, 'data', 'pose', name + '_pose.json')
        with open(path_mat) as f:
            json_data = f.read()
        camera_mat = json.loads(json_data)['camera_k_matrix']
        camera_mat = np.array(camera_mat)
        camera_mat[:2] = camera_mat[:2] / 2
        # print('camera_mat', camera_mat)

        path_dep_in = os.path.join(dir_dep_in, name + '_depth.png')
        depth = cv2.imread(path_dep_in, cv2.IMREAD_UNCHANGED)
        # Depth images are stored as 16-bit PNGs and
        # have a maximum depth of 128m and a sensitivity of 1/512m(65535 is maximum depth).
        depth += 1  # ignore max depth (65535 -> 0)
        depth = depth.astype(np.float)
        depth /= 512.0
        hha = getHHA(camera_mat, depth, depth)  # input depth (m)
        # print(hha.min(), hha.max())
        path_hha_out = os.path.join(dir_hha_out, name + '.png')
        cv2.imwrite(path_hha_out, hha)  # cv2.write will change hha into ahh

        print('hha', i, area_name)
        if is_test and i > 0:
            break


def save_labels(dir_in, area_names, dir_out):
    path_json_label = os.path.join(dir_in, 'assets', 'semantic_labels.json')
    with open(path_json_label) as f:
        json_labels = json.load(f)
    # print(json_labels)
    print(json_labels, len(json_labels))

    label_id_count = 0
    map_name_id = {}
    map_id = {}
    for i, label in enumerate(json_labels):
        label = label.split('_')[0]
        # print(label)
        if label not in map_name_id.keys():
            map_name_id[label] = label_id_count
            label_id_count += 1
        map_id[i] = map_name_id[label]
    print('map_name_id', map_name_id)
    print('map_id', map_id)

    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_label_in = os.path.join(dir_in, area, 'data', 'semantic')
        dir_label_out = os.path.join(dir_out, area, 'label')
        if not os.path.isdir(dir_label_out):
            os.makedirs(dir_label_out)

        path_label_in = os.path.join(dir_label_in, name + '_semantic.png')
        path_label_out = os.path.join(dir_label_out, name + '.png')
        # The semantic images have RGB images which are direct 24-bit base-256 integers
        # which contain an index into /assets/semantic_labels.json.
        label = cv2.imread(path_label_in)
        # print(label[:, :, 2])
        label = label[:, :, 0] + label[:, :, 1] * 256 + label[:, :, 2] * 256 * 256
        label = np.vectorize(map_id.get)(label)
        # print(label)
        label = np.uint8(label)
        cv2.imwrite(path_label_out, label)

        print('label', i, area_name)
        if is_test and i > 9:
            break


def get_names(dir_in, area):
    dir_img = os.path.join(dir_in, area, 'data', 'rgb')
    name_exts = os.listdir(dir_img)
    names = []
    for name_ext in name_exts:
        name = name_ext.split('.')[0].strip()
        if len(name) != 0:
            name = '_'.join(name.split('_')[:-1])
            names.append(area + ' ' + name)
    return names


def save_list(dir_in, dir_out):
    train_list = []
    for area in train_areas:
        train_list += get_names(dir_in, area)
    print('train_list', len(train_list))

    test_list = []
    for area in test_areas:
        test_list += get_names(dir_in, area)
    print('test_list', len(test_list))

    def write_txt(path_list, list_ids):
        with open(path_list, 'w') as f_list:
            f_list.write('\n'.join(list_ids))

    path_list = os.path.join(dir_out, 'train.txt')
    write_txt(path_list, train_list)

    path_list = os.path.join(dir_out, 'test.txt')
    write_txt(path_list, test_list)

    return train_list, test_list


def main(dir_in, dir_out, cpus):
    train_list, test_list = save_list(dir_in, dir_out)
    area_names = train_list + test_list
    print('area_names', len(area_names))
    save_imgs(dir_in, area_names, dir_out)
    save_depth(dir_in, area_names, dir_out)
    save_labels(dir_in, area_names, dir_out)

    len_sub = len(area_names) // cpus
    chunks_area_names = [area_names[i:i + len_sub] for i in range(0, len(area_names), len_sub)]
    processes = []
    for chunk in chunks_area_names:
        print('chunk', len(chunk))
        p = Process(target=save_hha, args=(dir_in, chunk, dir_out))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # save_hha(dir_in, area_names, dir_out)


if __name__ == '__main__':
    input_dir = ""
    output_dir = ""
    cpu_num = 24
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:c:", ["input_meta_dir=", "output_dir=", "cpus="])
    except getopt.GetoptError:
        print('gen_sid.py -i <input_meta_dir> -o <output_dir> -c <cpus>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('gen_sid.py -i <input_meta_dir> -o <output_dir>')
            sys.exit()
        elif opt in ("-i", "--input_meta_dir"):
            input_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-c", "--cpus"):
            cpu_num = int(arg)

    main(input_dir, output_dir, cpu_num)


