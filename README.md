## ClothSeg: Semantic Segmentation Network with Feature Projection for Clothing Parsing
Code for ClothSeg

## Abastract
Semantic segmentation of clothing presents a formidable challenge owing to the non-rigid geometric deformation properties inherent in garments. In this paper, we use the Transformer as the encoder to better learn global information for clothing semantic segmentation. In addition, we propose a Feature Projection Fusion (FPF) module to better utilize local information. This module facilitates the integration of deep feature maps with shallow local details, thereby enabling the network to capture both high-level abstractions and fine-grained details of features. We also design a pixel distance loss in training to emphasize the impact of edge features. This loss calculates the mean of the shortest distances between all predicted clothing edges and the true clothing edges during the training process. We perform extensive experiments and our method achieves 56.30\% and 74.97\% mIoU on the public dataset CFPD and our self-made dataset LIC, respectively, demonstrating a competitive performance when compared to the state-of-the-art.

## Preparations

### CFPD dataset

#### Download
You can obtain the CFPD dataset from the following projects ([dataset-CFPD](https://github.com/hrsma2i/dataset-CFPD)). You should convert the downloaded dataset into the VOC format. Then, the directory sctructure should thus be

``` bash
CFPD/
└──
   VOCdevkit/
   └── VOC2012
       ├── JPEGImages
       ├── SegmentationClass
       └──ImageSets
          └── Segmentation
              ├── train.txt
              ├── val.txt
              ├── test.txt
```

Alternatively, you can directly download the CFPD dataset we have prepared.[Google Drive](https://drive.google.com/file/d/177I3UKbzui1EpwiNqUObuTkNsfBbUdac/view?usp=sharing)

### Download Our weights
[Google Drive](https://drive.google.com/file/d/1KGHGNrTIa8ncLQGtJPJWJ0a_rfQoQefy/view?usp=sharing)

Afterwards, simply place the weight file into the project directory.

## Predict

Open the `test.py` and modify the following parameters.
``` bash
    parser.add_argument("--data_root", type=str, default='/Your dataset path/CFPD',
                        help="path to Dataset")
```

Predict.
``` bash
    python test.py
```

## Extract edge pixels
``` bash
    python edge.py
```

## Example of calculating pixel distance loss

We assume that the predicted results of the feature map are in the 5th and 10th rows, while the actual values of the labels are in the 12th row. So, there are 6 `(12 - 5) - 1 = 6` rows of pixels and 1 `(12 - 10) - 1 = 1` row of pixels between them and the labels, respectively. The final average distance is 3.5 `(6 + 1) / 2 = 3.5`.

``` bash
    python show_dis_loss.py
```
