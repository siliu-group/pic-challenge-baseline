# coding=utf-8
"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

import numpy as np
import torch
import os
from dataloaders.pic import PICDataLoader, PICDataset
from config import ModelConfig, PIC_PATH
from lib.pytorch_misc import optimistic_restore
import cv2
import json
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
else:
    raise ValueError()

assert conf.dataset == 'pic'

output_root = '/home/rgh/Relation/code/baseline/data/pic_output'
relations_path = os.path.join(output_root, 'relations.json')
semantic_path = os.path.join(output_root, 'semantic')
instance_path = os.path.join(output_root, 'instance')
if not os.path.exists(semantic_path):
    os.mkdir(semantic_path)
if not os.path.exists(instance_path):
    os.mkdir(instance_path)


test = PICDataset(mode='test')
test_loader = PICDataLoader.get_test_loader(test, mode='rel', batch_size=conf.batch_size,
                                                 num_workers=conf.num_workers,
                                                 num_gpus=conf.num_gpus)

detector = RelModel(classes=test.ind_to_classes, rel_classes=test.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision,
                    dataset=conf.dataset,
                    )


detector.cuda()

ckpt = torch.load(conf.ckpt)
print("Loading EVERYTHING")
if not optimistic_restore(detector, ckpt['state_dict']):
    print('load error')

rels_info = []
def test_epoch():
    detector.eval()
    result_dict = {}
    iou_threshs = [0.25, 0.5, 0.75]
    for iou_thresh in iou_threshs:
        result_dict[iou_thresh] = {20: [], 50: [], 100: []}
    for test_b, batch in enumerate(test_loader):
        print(test_b)
        test_batch(conf.num_gpus * test_b, batch, result_dict, iou_threshs)
    with open(relations_path, 'w') as f:
        json.dump(rels_info, f)


def test_batch(batch_num, b, result_dict, iou_threshs):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, masks_i, objs_labels_i, obj_scores_i, rels_i, rel_classes_i, overal_scores_i) in enumerate(det_res):
        assert np.all(objs_labels_i[rels_i[:, 0]] > 0) and np.all(objs_labels_i[rels_i[:, 1]] > 0)
        img_name = test.img_names[batch_num + i]
        img = cv2.imread(os.path.join(PIC_PATH, 'image/' + test.mode + '/' + img_name + '.jpg'))
        im_h = img.shape[0]
        im_w = img.shape[1]
        boxes_i = boxes_i.astype(np.int32)
        pred_semantic = np.zeros((im_h, im_w), dtype=np.uint8)#np.zeros_like(gt_semantic)
        pred_instance = np.zeros((im_h, im_w), dtype=np.uint8)#np.zeros_like(gt_instance)
        obj_scores_i_sorted_index = np.argsort(obj_scores_i)
        for index in range(len(obj_scores_i_sorted_index)):
            instance_id = obj_scores_i_sorted_index[index]
            ref_box = boxes_i[instance_id, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)
            padded_mask = np.zeros((28 + 2, 28 + 2), dtype=np.float32)
            padded_mask[1:-1, 1:-1] = masks_i[instance_id, :, :]
            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask >= 0.5, dtype=np.uint8)
            mask_category = mask * int(objs_labels_i[instance_id])
            # instance_id begins from 1
            mask_instance = mask * (instance_id+1)
            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)
            pred_instance[y_0:y_1, x_0:x_1] = mask_instance[
                                                       (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                                       (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                                       ]
            pred_semantic[y_0:y_1, x_0:x_1] = mask_category[
                                                       (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                                       (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                                       ]
        cv2.imwrite(os.path.join(semantic_path, img_name + '.png'), pred_semantic)
        cv2.imwrite(os.path.join(instance_path, img_name + '.png'), pred_instance)
        instance_ids = np.unique(pred_instance)
        relations = []
        rel_num = 0
        for j in range(len(rels_i)):
            if rel_num >= 100:
                break
            subject = int(rels_i[j, 0] + 1)
            object = int(rels_i[j, 1] + 1)
            if subject not in instance_ids or object not in instance_ids:
                continue
            relations.append(
                {
                    'subject': subject,
                    'relation': int(rel_classes_i[j]),
                    'object': object,
                    'score': round(float(overal_scores_i[j]), 6)
                }
            )
            rel_num += 1
        rels_info.append({
            'relations': relations,
            'name': img_name+'.jpg',
        })


print("Test starts now!")

test_epoch()