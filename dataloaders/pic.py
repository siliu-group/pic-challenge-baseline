# coding=utf-8
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from collections import defaultdict
from pycocotools.coco import COCO
from config import PIC_PATH, IM_SCALE, BOX_SCALE, PIC_OFFLINE_PATH
import lib.utils.segms as segm_utils
import json
import cv2


class PICDataset(Dataset):
    """
     Adapted from the torchvision code
     """

    def __init__(self, mode, filter_duplicate_rels=True, mask_resolution=28):
        """
        :param mode: train2014 or val2014
        """
        self.mask_resolution = mask_resolution
        self.mode = mode
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'
        tform = []
        if self.is_train:
            tform.append(RandomOrder([
                Grayscale(),
                Brightness(),
                Contrast(),
                Sharpness(),
                Hue(),
            ]))
        tform += [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)
        image_names = [name[:-4] for name in os.listdir(os.path.join(PIC_PATH, 'image/'+mode)) if name.endswith('.jpg')]
        if self.mode != 'test':
            semantic_names = [name[:-4] for name in os.listdir(os.path.join(PIC_PATH, 'segmentation/'+mode+'/semantic')) if name.endswith('.png')]
            instance_names = [name[:-4] for name in os.listdir(os.path.join(PIC_PATH, 'segmentation/'+mode+'/instance')) if name.endswith('.png')]
            image_names.sort(key=str.lower)
            semantic_names.sort(key=str.lower)
            instance_names.sort(key=str.lower)
            assert image_names == semantic_names
            assert image_names == instance_names
        # image_names = [name[:-4] for name in os.listdir(os.path.join(PIC_OFFLINE_PATH, 'val/obj_feat/')) if
        #                name.endswith('.npy')]
        # image_names.sort(key=str.lower)
        self.img_names = image_names
        rel_cats = json.load(open(os.path.join(PIC_PATH,'categories_list/relation_categories.json')))
        self.ind_to_predicates = [rel_cat['name'] for rel_cat in rel_cats]
        cls_cats = json.load(open(os.path.join(PIC_PATH, 'categories_list/label_categories.json')))
        self.ind_to_classes = [cls_cat['name'] for cls_cat in cls_cats]
        if self.mode != 'test':
            self.img2rels = dict()
            img_relations = json.load(open(os.path.join(PIC_PATH, 'relation/relations_'+self.mode+'.json')))
            for img_relation in img_relations:
                rels = []
                for index, rel in enumerate(img_relation['relations']):
                    temp = np.array([[rel['subject']-1, rel['object']-1, rel['relation']]], dtype=np.int32)
                    rels.append(temp)
                rels = np.concatenate(rels, axis=0)
                self.img2rels[img_relation['name'][:-4]] = rels
        print('====================')
        print(self.ind_to_classes)
        print(self.ind_to_predicates)

        self.id_to_ind = {ind: ind for ind, name in enumerate(self.ind_to_classes)}
        self.ind_to_id = {x: y for y, x in self.id_to_ind.items()}
        #self.create_coco_format()
    def create_coco_format(self):

        self.coco = COCO()
        dataset = dict()
        images = [{'id': index, 'file_name': img_name} for index, img_name in enumerate(self.img_names)]
        dataset['images'] = images
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        self.coco.dataset = {}


    @property
    def is_train(self):
        return self.mode.startswith('train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns: entry dict
        """

        # Optionally flip the image if we're doing training
        flipped = False #self.is_train and np.random.random() > 0.5

        img_name = self.img_names[index]
        if self.mode == 'test':
            gt_boxes = np.ones((1, 4))
            gt_classes = np.ones((1, 1))
            gt_rels = np.ones((1, 3))
            gt_masks = np.ones((1, 28, 28))
        else:
            instance = cv2.imread(os.path.join(PIC_PATH, 'segmentation/'+self.mode+'/instance/'+img_name+'.png'),
                                  flags=cv2.IMREAD_GRAYSCALE)
            semantic = cv2.imread(os.path.join(PIC_PATH, 'segmentation/' + self.mode + '/semantic/' + img_name + '.png'),
                                  flags=cv2.IMREAD_GRAYSCALE)

            max_instance_id = instance.max()
            gt_boxes = []
            gt_masks = []
            gt_classes = []
            for instance_id in range(1, max_instance_id+1):
                mask = instance == instance_id
                mask = mask.astype(np.uint8)
                # x0,y0,x1,y1
                box = segm_utils.mask_to_bbox(mask)
                gt_boxes.append(box[None, ...])
                temp = np.nonzero(mask)
                gt_classes.append(semantic[temp[0][0], temp[1][0]])
                mask = mask[int(box[1]):int(box[3])+1, int(box[0]):int(box[2])+1]
                mask = cv2.resize(mask, (self.mask_resolution, self.mask_resolution), interpolation=cv2.INTER_NEAREST)
                gt_masks.append(mask[None, ...])

            gt_boxes = np.concatenate(gt_boxes, axis=0)
            gt_masks = np.concatenate(gt_masks, axis=0)
            gt_classes = np.concatenate([gt_classes], axis=0)
            gt_rels = self.img2rels[img_name]
            if self.filter_duplicate_rels:
                # Filter out dupes!
                assert self.mode == 'train'
                old_size = gt_rels.shape[0]
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in gt_rels:
                    all_rel_sets[(o0, o1)].append(r)
                gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
                gt_rels = np.array(gt_rels)

        obj_feat = np.load(os.path.join(PIC_OFFLINE_PATH, self.mode+'/obj_feat/'+img_name+'.npy'))
        roi_feat = np.load(os.path.join(PIC_OFFLINE_PATH, self.mode+'/roi_feat/'+img_name+'.npy'))
        pred_dists = obj_feat[:, 1024:1024 + 85]
        labels = pred_dists.argmax(axis=1)
        non_bg_index = labels != 0
        pred_dists = pred_dists[non_bg_index]
        #pred_fmaps = roi_feat[:, 0:1024][non_bg_index]
        pred_fmaps = roi_feat[non_bg_index]
        pred_boxes = obj_feat[:, 1024+85:1024+85+4][non_bg_index]
        pred_masks = obj_feat[:, 1024+85+4:].reshape((-1, 30, 30))[:, 1:-1, 1:-1][non_bg_index]
        pred_masks = np.array(pred_masks >= 0.5, dtype=np.uint8)

        entry = {
            'img': img_name, #self.transform_pipeline(image_unpadded),
            'img_size': (1, 1, 1),
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'gt_relations': gt_rels,
            'index': index,
            'gt_masks': gt_masks,
            'flipped': flipped,
            'pred_fmaps': pred_fmaps,
            'pred_dists': pred_dists,
            'pred_boxes': pred_boxes,
            'pred_masks': pred_masks
        }

        assertion_checks(entry)
        return entry

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        return train, val

    def __len__(self):
        return len(self.img_names)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)

def assertion_checks(entry):
    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


# def coco_collate(data, num_gpus=3, is_train=False):
#     blob = Blob(mode='det', is_train=is_train, num_gpus=num_gpus,
#                 batch_size_per_gpu=len(data) // num_gpus)
#     for d in data:
#         blob.append(d)
#     blob.reduce()
#     return blob
def pic_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob

class PICDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    # def __iter__(self):
    #     for x in super(CocoDataLoader, self).__iter__():
    #         if isinstance(x, tuple) or isinstance(x, list):
    #             yield tuple(y.cuda(async=True) if hasattr(y, 'cuda') else y for y in x)
    #         else:
    #             yield x.cuda(async=True)

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: pic_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode == 'det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: pic_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load

    @classmethod
    def get_test_loader(cls, test_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        return cls(
            dataset=test_data,
            batch_size=batch_size * num_gpus if mode == 'det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: pic_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )