## Introduction ##

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

2. run ./script/train.sh to train relation model.

3. run ./scripy/test.sh to generate results.


## File Tree ##


- obj_feat/xxx.npy: [\#instance, 1024+85+4+30\*30] 1024: the output feat of fc7. 85: cls_score. 4: (x1,y1,x2,y2). 30*30: mask
- roi_feat/xxx.npy: [\#instance, 256\*7\*7] the feat after RoIAlign of all instances
- you can download glove.6B.100d.pt/glove.6B.200d.pt from [baiduyun pwd: at8t](https://pan.baidu.com/s/1wAWUpvmSQ9PV44uDJbUqZQ) or [official website](https://nlp.stanford.edu/data/)

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
----------------------------------------
PIC_OFFLINE_PATH
└───train
│   └───obj_feat
│   │      xxx.npy
│   └───roi_feat
│   │      xxx.npy
└───val
│   └───obj_feat
│   │      xxx.npy
│   └───roi_feat
│   │      xxx.npy
└───test
│   └───obj_feat
│   │      xxx.npy
│   └───roi_feat
│   │      xxx.npy
----------------------------------------
DATA_PATH
└───glove.6B.100d.pt
└───glove.6B.200d.pt
----------------------------------------
```
# Help

Feel free to push issues if you encounter trouble getting it to work!