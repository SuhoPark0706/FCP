# Foreground-Covering Prototype Generation and Matching for SAM-Aided Few-Shot Segmentation (FCP)
This is the official repository for the following paper:
> **Foreground-Covering Prototype Generation and Matching for SAM-Aided Few-Shot Segmentation** [[Arxiv]](https://www.arxiv.org/abs/2501.00752)
> 
> Suho Park*, SuBeen Lee*, Hyun Seok Seong, Jaejoon Yoo, Jae-Pil Heo\
> Accepted by **AAAI 2025**


## Requirements

- Python 3.10
- PyTorch 1.12
- cuda 11.6

Conda environment settings:
```bash
conda create -n fcp python=3.10
conda activate fcp

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Segment-Anything-Model setting:
```bash
cd ./segment-anything
pip install -v -e .
cd ..
```

## Preparing Few-Shot Segmentation Datasets
Download following datasets:

> #### 1. PASCAL-5<sup>i</sup>
> Download PASCAL VOC2012 devkit (train/val data):
> ```bash
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from our [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].

> #### 2. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations: 
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```
> Download COCO2014 train/val annotations from our Google Drive: [[train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing)].
> (and locate both train2014/ and val2014/ under annotations/ directory).
       

> #### 3. Image Encoder weights
> Resnet : https://drive.google.com/drive/folders/1Hrz1wOxOZm4nIIS7UMJeL79AQrdvpj6v \
> VGG : https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

Create a directory '../dataset' for the above few-shot segmentation datasets and appropriately place each dataset to have following directory structure:

    ../                         # parent directory
    ├── ./                      # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSSS dataset
    │   ├── model/              # (dir.) implementation of VRP-SAM 
    │   ├── segment-anything/   # code for SAM
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training HSNet
    │   └── SAM2Pred.py         # code for prediction module
    │    
    ├── resnet50_v2.pth
    ├── vgg16.pth
    │    
    └── dataset/
        ├── VOC2012/            # PASCAL VOC2012 devkit
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── ...
        │   └── SegmentationClassAug/
        └── COCO2014/           
            ├── annotations/
            │   ├── train2014/  # (dir.) training masks (from Google Drive) 
            │   ├── val2014/    # (dir.) validation masks (from Google Drive)
            │   └── ..some json files..
            ├── train2014/
            └── val2014/

## Training

> ```bash
>sh scripts/train_pascal.sh  
>sh scripts/train_coco.sh  
> ```


   
## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@article{park2025foreground,
  title={Foreground-Covering Prototype Generation and Matching for SAM-Aided Few-Shot Segmentation},
  author={Park, Suho and Lee, SuBeen and Seong, Hyun Seok and Yoo, Jaejoon and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2501.00752},
  year={2025}
}
````
