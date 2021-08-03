# Model Zoo

## NYU-V2(40 categories)
| Architecture | Backbone | MS & Flip | Shape Conv | mIOU | Config | Params |
|:---:|:---:|:---:|:---:| :---:| :---:| :---:|
| DeepLabv3plus | ResNeXt-101 | True | False | 50.25% | [config](../configs/nyu/nyu40_deeplabv3plus_resnext101_baseline.py) | [Google Drive](https://drive.google.com/file/d/1v652kjDPQ9KlSO4cjazvzX5R-mK72lzP/view?usp=sharing) |
| DeepLabv3plus | ResNeXt-101 | True | True | 51.3% | [config](../configs/nyu/nyu40_deeplabv3plus_resnext101_shape.py) | [Google Drive](https://drive.google.com/file/d/1mZfZu8o4zjuKnLCWo28k0i-uYAl9BqVf/view?usp=sharing) |

## NYU-V2(13 categories)
| Architecture | Backbone | MS & Flip | Shape Conv | mIOU | Config | Params |
|:---:|:---:|:---:|:---:| :---:| :---:| :---:|
| DeepLabv3plus | ResNeXt-101 | True | False | 48.85% | [config](../configs/nyu/nyu13_deeplabv3plus_resnext101_baseline.py) | [Google Drive](https://drive.google.com/file/d/1M9ptiFP-BmLWOr9TiNMYB7YmAC4g7Qvr/view?usp=sharing) |
| DeepLabv3plus | ResNeXt-101 | True | True | 50.21% | [config](../configs/nyu/nyu13_deeplabv3plus_resnext101_shape.py) | [Google Drive](https://drive.google.com/file/d/1BCziuqhpzeLnYGQbiPOTAwU_VVnGHcwF/view?usp=sharing) |


## SUN-RGBD
| Architecture | Backbone | MS & Flip | Shape Conv | mIOU | Config |
|:---:|:---:|:---:|:---:| :---:| :---:|
| DeepLabv3plus | ResNet-101 | True | False | 47.6% | [config](../configs/sun/sun_deeplabv3plus_resnext101_baseline.py) |
| DeepLabv3plus | ResNet-101 | True | True | 48.6% | [config](../configs/sun/sun_deeplabv3plus_resnext101_shape.py) |


## SID(Stanford Indoor Dataset )
| Architecture | Backbone | MS & Flip | Shape Conv | mIOU | Config |
|:---:|:---:|:---:|:---:| :---:| :---:|
| DeepLabv3plus | ResNet-101 | True | False | 54.55% | [config](../configs/sid/sun_deeplabv3plus_resnext101_baseline.py) |
| DeepLabv3plus | ResNet-101 | True | True | 60.6% | [config](../configs/sid/sun_deeplabv3plus_resnext101_shape.py) |


