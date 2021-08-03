# Data Preparation

## NYU-V2
The [NYU-Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.

### Download
Images and annotations from the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), 
category mapping, etc. from the [open source project](https://github.com/ankurhanda/nyuv2-meta-data).
- [`nyu_depth_v2_labeled.mat`](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)
contains the required RGB images, depth images and semantic segmentation annotations.
- [`classMapping40.mat`](https://github.com/ankurhanda/nyuv2-meta-data/raw/master/classMapping40.mat)
contains a mapping of 849 categories to 40 categories.
- [`class13Mapping.mat`](https://github.com/ankurhanda/nyuv2-meta-data/raw/master/class13Mapping.mat)
contains a mapping of 40 categories to 13 categories.
- [`splits.mat`](https://github.com/ankurhanda/nyuv2-meta-data/raw/master/splits.mat)
contains a list of training and testing sets.
- [`camera_rotations_NYU.txt`](https://github.com/ankurhanda/nyuv2-meta-data/raw/master/camera_rotations_NYU.txt)
contains camera rotations that are useful for calculating HHA.

### Conversion
Download needed files into a folder. Then convert the files to the desired format, please run:
```shell script
python gen_nyu.py -i <input_meta_dir> -o <output_dir>
```
`input_meta_dir` is the path where the raw data is stored.
`output_dir` is the path where to save the converted data.
> Note: The converted depth values are in millimetres, and save as uint16[0,65535] png, 0~65535(mm).
> Depth pixels where the depth is missing are encoded with 0.

## SID(Stanford Indoor Dataset)

The 2D-3D-S dataset provides a variety of mutually registered modalities from 2D, 2.5D and 3D domains, with instance-level semantic and geometric annotations. It covers over 6,000 m2 collected in 6 large-scale indoor areas that originate from 3 different buildings. It contains over 70,000 RGB images, along with the corresponding depths, surface normals, semantic annotations, global XYZ images (all in forms of both regular and 360° equirectangular images) as well as camera information. It also includes registered raw and semantically annotated 3D meshes and point clouds. The dataset enables development of joint and cross-modal learning models and potentially unsupervised approaches utilizing the regularities present in large-scale indoor spaces.
For more information on the dataset, visit the [tools repo](https://github.com/alexsax/2D-3D-Semantics), the [project site](http://3dsemantics.stanford.edu/) or the [dataset wiki](https://github.com/alexsax/2D-3D-Semantics/wiki).

### Download

The link will first take you to a license agreement, and then to the data.
- [Download the full 2D-3D-S Dataset](https://goo.gl/forms/2YSPaO2UKmn5Td5m2) [checksums](https://github.com/alexsax/2D-3D-Semantics/wiki/Checksum-Values-for-Data)

The full dataset is very large at 766G. Therefore, we have split the data by area to accomodate a la carte data selection. 
The dataset also comes in two flavors: with global_xyz images (766G) and without (110G). Only need to download `noXYZ`.

### Conversion

```shell script
python gen_sid.py -i <input_meta_dir> -o <output_dir> -c <processer_nums>
```
`input_meta_dir` is the path where the raw data is stored.
`output_dir` is the path where to save the converted data.
`processer_nums` is the number of processes. Large volume of data requiring multi-process processing.
> Note: Depth images are stored as 16-bit PNGs and have a maximum depth of 128m and a sensitivity of 1/512m.
> Depth pixels where the depth is missing are encoded with 65535.

## Acknowledgments
The code to generate HHA comes from [SUN RGB-D meta data repository](https://github.com/ankurhanda/sunrgbd-meta-data). 
