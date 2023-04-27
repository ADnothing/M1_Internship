#!/usr/bin/env python


#Patch_Management.py

import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=False)
def fct_IoU(box1, box2):
    #box1 = np.copy(box1_o); box2 = np.copy(box2_o)
    #box1[0] += 1; box1[1] += 1; box1[2] -= 1; box1[3] -= 1
    #box2[0] += 1; box2[1] += 1; box2[2] -= 1; box2[3] -= 1
    inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    inter_2d = inter_w*inter_h
    uni_2d = (box1[2]-box1[0])*(box1[3] - box1[1]) + \
        (box2[2]-box2[0])*(box2[3] - box2[1]) - inter_2d
    enclose_w = (max(box1[2], box2[2]) - min(box1[0], box2[0]))
    enclose_h = (max(box1[3], box2[3]) - min(box1[1],box2[1]))

    cx_a = (box1[2] + box1[0])*0.5; cx_b = (box2[2] + box2[0])*0.5
    cy_a = (box1[3] + box1[1])*0.5; cy_b = (box2[3] + box2[1])*0.5
    dist_cent = np.sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b))
    diag_enclose = np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h)

    return float(inter_2d)/float(uni_2d) - float(dist_cent)/float(diag_enclose)
    #return float(inter_2d)/float(uni_2d)
    
    
@jit(nopython=True, cache=True, fastmath=False)
def fct_classical_IoU(box1, box2):
    #box1 = np.copy(box1_o); box2 = np.copy(box2_o)
    #box1[0] += 1; box1[1] += 1; box1[2] -= 1; box1[3] -= 1
    #box2[0] += 1; box2[1] += 1; box2[2] -= 1; box2[3] -= 1
    inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    inter_2d = inter_w*inter_h
    uni_2d = (box1[2]-box1[0])*(box1[3] - box1[1]) + \
        (box2[2]-box2[0])*(box2[3] - box2[1]) - inter_2d

    return float(inter_2d)/float(uni_2d)

def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.

    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.

    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]

    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)

    return subs

def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.

    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]

    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result
