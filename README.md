## Introduction(V2) ##

This repository just provides a baseline for [PIC Challenge](http://www.picdataset.com/challenge/index/). **Lots of code are from [neural-motifs](https://github.com/rowanz/neural-motifs) and it is not end-to-end.**

```
@inproceedings{zellers2018scenegraphs,
  title={Neural Motifs: Scene Graph Parsing with Global Context},
  author={Zellers, Rowan and Yatskar, Mark and Thomson, Sam and Choi, Yejin},
  booktitle = "Conference on Computer Vision and Pattern Recognition",  
  year={2018}
}
```

## Usage ##

0. Compile everything. run `make` in the main directory

1. You should train a instance-segmentation model on PIC. For example, you can use [Detectron](https://github.com/facebookresearch/Detectron) to generate the boxes, masks and feature maps of train/val/test data.

2. run `./script/train_motif.sh` or `./script/train_stanford.sh` to train relation model.

3. run `./scripy/test_motif.sh` or `./scripy/test_stanford.sh` to generate and eval results.

## File Tree ##

- you can download glove.6B.100d.pt/glove.6B.200d.pt from [baiduyun pwd: at8t](https://pan.baidu.com/s/1wAWUpvmSQ9PV44uDJbUqZQ) or [official website](https://nlp.stanford.edu/data/)
- offline features

```
----------------------------------------
PIC_PATH
└───categories_list
│   └───label_categories.json 
│   └───relation_categories.json
└───image
│   └───train
│   │      xxx.jpg
│   └───val
│   │      xxx.jpg
│   └───test
│   │      xxx.jpg
└───relation
│   └───relations_train.json 
│   └───relations_val.json
└───segmentation
│   └───train
│   │      └───instance
│   │      │      xxx.png
│   │      └───semantic
│   │      │      xxx.png
│   └───val
│   │      └───instance
│   │      │      xxx.png
│   │      └───semantic
│   │      │      xxx.png
└───test_gt
----------------------------------------
MASK_RCNN_PATH
└───train
│   └───P4
│   │      xxx.npy(1,256,H,W)
└───val
│   └───P4
│   │      xxx.npy(1,256,H,W)
│   └───bboxs
│   │      xxx.npy(N,4)
│   └───labels
│   │      xxx.npy(N,1)
│   └───masks
│   │      xxx.npy(N,28,28)
└───test
│   └───P4
│   │      xxx.npy(1,256,H,W)
│   └───bboxs
│   │      xxx.npy(N,4)
│   └───labels
│   │      xxx.npy(N,1)
│   └───masks
│   │      xxx.npy(N,28,28)
----------------------------------------
DATA_PATH
└───glove.6B.100d.pt
└───glove.6B.200d.pt
----------------------------------------
```

## Help#
Feel free to push issues if you encounter trouble getting it to work!