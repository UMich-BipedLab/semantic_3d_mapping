#!/usr/bin/python2.7

'''
Martin Kersner, m.kersner@gmail.com
2015/11/30

Ray Zhang, rzh@umich.edu
2019/05/17

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
'''

import numpy as np
import  sys, os
from scipy import misc as misc
from collections import OrderedDict
from sets import Set as set
import matplotlib.pyplot as plt

import pdb

def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU_batch(eval_list, gt_list, class_to_ignore_set):
    assert(len(eval_list) == len(gt_list))

    Intersections = OrderedDict()
    Unions = OrderedDict()

    for i in range(len(eval_list)):
        print("Img #{}".format(i))
        iu_curr_img = mean_IU_batch_single(eval_list[i], gt_list[i], Intersections, Unions, ignore_list)
        print("On Img #{}, the mIoU is {}".format(i, iu_curr_img))

    print("\nTotal: ")
    means = 0
    num_class = 0
    for c in Intersections:        
        if c in class_to_ignore_set:
            continue
        num_class += 1
        
        iou_c = Intersections[c] / Unions[c]
        print("class #{}: IoU is {}".format(c, iou_c))
        means += iou_c
        
    print("\nMean IoU: {}".format(means / num_class))
    return Intersections, Unions
    

def mean_IU_batch_single(eval_segm, gt_segm, Intersections, Unions, ignore_list):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))

    arguments:
    eval_segm:            the numpy img from prediction
    gt_segm:              the numpy img from ground truth
    Intersections:        the OrderedDict from class label to number that gt intersections prediction
    Unions:               the OrderedDict from class label to number that gt unions prediction
    ignore_list:          the set of labels that shall be ignored
    '''
    
    check_size(eval_segm, gt_segm)

    # cl: the sorted list of unique classes in (gt union pred)
    cl, n_cl   = union_classes(eval_segm, gt_segm)
    # unique classes in gt
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = []#list([0]) * n_cl  # IoU of this iamge
    print("")


    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        if c not in Intersections:
            Intersections[c] = 0
        if c not in Unions:
            Unions[c] = 0
           
        Intersections[c] += n_ii
        Unions[c] += (t_i + n_ij - n_ii)

        if n_ii > 0 and (not c in ignore_list):
            IU.append( n_ii / (t_i + n_ij - n_ii))

            # 
            global show_mask_img
            if show_mask_img == 1:
                fig=plt.figure()
                
                ax_pred = fig.add_subplot(3, 1, 1)
                ax_pred.title.set_text('class #{} Predition'.format(c))
                ax_pred.axis('off')
                plt.imshow(curr_eval_mask)
                
                ax_gt = fig.add_subplot(3, 1, 2)
                ax_gt.title.set_text('\nclass #{} Ground Truth'.format(c))
                ax_gt.axis('off')
                plt.imshow(curr_gt_mask)

                curr_eval_mask = curr_eval_mask.astype(np.bool)
                curr_gt_mask = curr_gt_mask.astype(np.bool)
                intersect = np.logical_and(curr_eval_mask, curr_gt_mask)
                iou_mask = np.zeros(intersect.shape).astype(np.int)
                iou_mask[intersect] = 255
                gt_not_pred = np.logical_and(curr_gt_mask, ~curr_eval_mask)
                pred_not_gt = np.logical_and(~curr_gt_mask, curr_eval_mask)
                iou_mask[gt_not_pred] = 150
                iou_mask[pred_not_gt] = 50
                ax_iou = fig.add_subplot(3,1,3)
                ax_iou.title.set_text('\nclass#{} IoU'.format(c))
                plt.imshow(iou_mask)
                plt.show()
            
            print('class #{} IU: {}'.format(c, IU[-1]))
    print("=================================================")
    mean_IU_ = np.sum(IU) / len(IU)
    return mean_IU_
    #return IU


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
        if c != 255:
            print(c, IU[i])
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    '''
    check the size of the two images
    '''
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


if __name__ == "__main__":
    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]
    show_mask_img = int(sys.argv[3])
    gt_list   = []
    pred_list = []

    ignore_list = set()
    ignore_list.add(255)
    ignore_list.add(1)
    ignore_list.add(8)
    for img_name in sorted(os.listdir(gt_folder)):
        if not img_name.endswith(".png"):
            continue
        if not os.path.isfile( pred_folder + "/" + img_name ):
            continue
 
        gt_curr = misc.imread(gt_folder + "/" + img_name)
        pred_curr = misc.imread(pred_folder + "/" + img_name)
        print("Read img {}".format(img_name))
        
        mask = (pred_curr == 255)
        gt_curr[mask] = 255

        gt_list.append(gt_curr)
        pred_list.append(pred_curr)
        
    mean_IU_batch(pred_list, gt_list, ignore_list)


    
