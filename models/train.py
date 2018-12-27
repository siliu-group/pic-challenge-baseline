# coding=utf-8
"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from dataloaders.pic import PICDataLoader, PICDataset
from config import ModelConfig, BOX_SCALE, IM_SCALE, PIC_PATH
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
from lib.evaluation.pic_eval import evaluate_from_dict
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.motif_pic import RelModel
elif conf.model == 'stanford':
    from lib.stanford_pic import RelModelStanford as RelModel
else:
    raise ValueError()

# assert conf.dataset == 'pic'

train, val = PICDataset.splits()
train_loader, val_loader = PICDataLoader.splits(train, val, mode='rel', batch_size=conf.batch_size,
                                                 num_workers=conf.num_workers,
                                                 num_gpus=conf.num_gpus)
# you can use (100, 100) for high speed
size = (640, 480)
detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
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
                    )

print(print_para(detector), flush=True)

# fc6_w = torch.from_numpy(np.load('data/pretrain_models/fc6_w.npy'))
# fc6_b = torch.from_numpy(np.load('data/pretrain_models/fc6_b.npy'))
# fc7_w = torch.from_numpy(np.load('data/pretrain_models/fc7_w.npy'))
# fc7_b = torch.from_numpy(np.load('data/pretrain_models/fc7_b.npy'))

# detector.roi_fmap[0].weight.data.copy_(fc6_w)
# detector.roi_fmap[0].bias.data.copy_(fc6_b)
# detector.roi_fmap[2].weight.data.copy_(fc7_w)
# detector.roi_fmap[2].bias.data.copy_(fc7_b)

# detector.roi_fmap_obj[0].weight.data.copy_(fc6_w)
# detector.roi_fmap_obj[0].bias.data.copy_(fc6_b)
# detector.roi_fmap_obj[2].weight.data.copy_(fc7_w)
# detector.roi_fmap_obj[2].bias.data.copy_(fc7_b)

def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler

start_epoch = -1


detector.cuda()
# ckpt = torch.load('/home/rgh/Relation/code/baseline_v2/checkpoints/mask_feat/pic_rel-25.tar')
# print("Loading EVERYTHING")
# start_epoch = ckpt['epoch']
# if not optimistic_restore(detector, ckpt['state_dict']):
#     start_epoch = -1


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]

    losses = {}
    losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])
    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res


def val_epoch():
    detector.eval()
    # rel_cats are 1-30. 30: geometric_rel and 31: non_geometric_rel are added into rel_cats for convenience
    rel_cats = {
        1: 'hold', 2: 'touch', 3: 'drive', 4: 'eat', 5: 'drink', 6: 'play', 7: 'look', 8: 'throw', 9: 'ride', 10: 'talk',
        11: 'carry', 12: 'use', 13: 'pull', 14: 'push', 15: 'hit', 16: 'feed', 17: 'kick', 18: 'wear', 19: 'in-front-of', 20: 'next-to',
        21: 'on-top-of', 22: 'behind', 23: 'on', 24: 'with', 25: 'in', 26: 'sit-on', 27: 'stand-on', 28: 'lie-in', 29: 'squat', 30: 'other',
        31: 'geometric_rel', 32: 'non_geometric_rel'}
    geometric_rel_cats = {19: 'in-front-of', 20: 'next-to', 21: 'on-top-of', 22: 'behind', 23: 'on', 25: 'in'}
    iou_threshes = [0.25, 0.5, 0.75]
    # result_dict = {0.25: {'hold': [], 'touch': [], ... }, 0.5: ...}
    result_dict = {iou_thresh: {rel_cat_name: [] for rel_cat_name in rel_cats.values()} for iou_thresh in iou_threshes}

    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, result_dict, iou_threshes, rel_cats, geometric_rel_cats)
    for iou_thresh in iou_threshes:
        print('----------IoU: %.2f(R@100)----------' % iou_thresh)
        for rel_cat_id, rel_cat_name in rel_cats.items():
            recalls = result_dict[iou_thresh][rel_cat_name]
            while None in recalls:
                recalls.remove(None)
            if len(recalls) != 0:
                recall_mean = float('%.4f' % np.mean(recalls))
                result_dict[iou_thresh][rel_cat_name] = recall_mean
                print('%s: %.4f' % (rel_cat_name, recall_mean))
            # if all of recalls are None, it means that rel_cat_id does not appear in all imgs
            else:
                result_dict[iou_thresh][rel_cat_name] = None
                print('%s does not appear in gt_rels' % rel_cat_name)

    print('----------Final Result(R@100)----------')
    final_result_iou_25 = (result_dict[0.25]['geometric_rel'] + result_dict[0.25]['non_geometric_rel']) / 2
    final_result_iou_50 = (result_dict[0.5]['geometric_rel'] + result_dict[0.5]['non_geometric_rel']) / 2
    final_result_iou_75 = (result_dict[0.75]['geometric_rel'] + result_dict[0.75]['non_geometric_rel']) / 2
    final_result_iou_average = (final_result_iou_25 + final_result_iou_50 + final_result_iou_75) / 3
    print('IoU(0.25): %.4f' % final_result_iou_25)
    print('IoU(0.5): %.4f' % final_result_iou_50)
    print('IoU(0.75): %.4f' % final_result_iou_75)
    print('Average: %.4f' % final_result_iou_average)
    return final_result_iou_average



def val_batch(batch_num, b, result_dict, iou_threshes, rel_cats, geometric_rel_cats):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, masks_i, objs_labels_i, obj_scores_i, rels_i, rel_classes_i, overal_scores_i) in enumerate(det_res):
        assert np.all(objs_labels_i[rels_i[:, 0]] > 0) and np.all(objs_labels_i[rels_i[:, 1]] > 0)
        img_name = val.img_names[batch_num + i]
        gt_instance = cv2.imread(
            os.path.join(PIC_PATH, 'segmentation/' + val.mode + '/instance/' + img_name + '.png'),
            flags=cv2.IMREAD_GRAYSCALE)
        im_h = gt_instance.shape[0]
        im_w = gt_instance.shape[1]
        gt_instance = cv2.resize(gt_instance, dsize=size, interpolation=cv2.INTER_NEAREST)
        gt_semantic = cv2.imread(
            os.path.join(PIC_PATH, 'segmentation/' + val.mode + '/semantic/' + img_name + '.png'),
            flags=cv2.IMREAD_GRAYSCALE)
        gt_semantic = cv2.resize(gt_semantic, dsize=size, interpolation=cv2.INTER_NEAREST)
        # must use copy(), or your gt_relations will be modified by gt_relations[:, 0:2] += 1
        gt_relations = val.img2rels[img_name].copy()
        # because gt instance_id begins from 1
        gt_relations[:, 0:2] += 1
        gt_entry = {
            'semantic': gt_semantic.astype(np.int32),
            'instance': gt_instance.astype(np.int32),
            'relations': gt_relations
        }
        # if conf.mode == 'sgcls':
        #     pred_relations = np.concatenate((rels_i, rel_classes_i[..., None]), axis=1)
        #     pred_relations[:, 0:2] += 1
        #     pred_entry = {
        #         'semantic': gt_semantic.astype(np.int32),
        #         'instance': gt_instance.astype(np.int32),
        #         'relations': pred_relations
        #     }
        # elif conf.mode == 'sgdet':
        if True:
            boxes_i = np.round(boxes_i).astype(np.int32)
            pred_semantic = np.zeros((im_h, im_w), dtype=np.uint8)#np.zeros_like(gt_semantic)
            pred_instance = np.zeros((im_h, im_w), dtype=np.uint8)#np.zeros_like(gt_instance)
            obj_order = np.argsort(objs_labels_i)[::-1]
            for i, instance_id in enumerate(obj_order):
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
                #print(ref_box, x_0, x_1, y_0, y_1, w, h)
                nonbk = mask != 0
                pred_instance[y_0:y_1, x_0:x_1][nonbk] = mask_instance[
                                                           (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                                           (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                                           ][nonbk]
                pred_semantic[y_0:y_1, x_0:x_1][nonbk] = mask_category[
                                                           (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                                           (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                                           ][nonbk]

            pred_relations = np.concatenate((rels_i, rel_classes_i[..., None]), axis=1)
            pred_relations[:, 0:2] += 1
            pred_semantic = cv2.resize(pred_semantic, dsize=size, interpolation=cv2.INTER_NEAREST)
            pred_instance = cv2.resize(pred_instance, dsize=size, interpolation=cv2.INTER_NEAREST)
            pred_entry = {
                'semantic': pred_semantic.astype(np.int32),
                'instance': pred_instance.astype(np.int32),
                'relations': pred_relations
            }

        evaluate_from_dict(gt_entry, pred_entry, result_dict, iou_threshes=iou_threshes,
                           rel_cats=rel_cats, geometric_rel_cats=geometric_rel_cats)
        # evaluator[conf.mode].evaluate_scene_graph_entry(
        #     gt_entry,
        #     pred_entry,
        # )


print("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('pic_rel', epoch)))
    mAp = val_epoch()
    scheduler.step(mAp)
    if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
        print("exiting training early", flush=True)
        break
