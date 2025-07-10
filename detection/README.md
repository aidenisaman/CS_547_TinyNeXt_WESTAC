# Object Detection Implementation
## Object Detection Performance Based on [SSDLite](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) for [COCO2017](https://cocodataset.org):
|  Backbone  |  AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | #Params |                      Log                      |
| :--------: | :--: | :-------------: | :-------------: | :------------: | :------------: | :------------: | :-----: | :-------------------------------------------: |
| TinyNeXt-S | 22.4 |      37.9       |      22.7       |      2.2       |      21.9      |      44.2      |  2.3M   | [log](logs/ssdlite_tinynext_s_coco/train.log) |
| TinyNeXt-M | 25.0 |      41.1       |      25.4       |      3.4       |      25.1      |      48.8      |  3.3M   | [log](logs/ssdlite_tinynext_m_coco/train.log) |

## Get Start

1.Ref to [MMDetection](https://github.com/open-mmlab/mmdetection) for the environments.

2.Configs can be found in `./configs/ssd_tinynext`

3.run :

```
./dish_train.sh configs/ssd/ssdlite_tinynext_s_coco.py 8
```

or

```
./dish_train.sh configs/ssd/ssdlite_tinynext_m_coco.py 8
```
