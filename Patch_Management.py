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

def poolingOverlap(mat, f, stride=None, method='max', pad=False, return_max_pos=False):
	'''Overlapping pooling on 2D or 3D data.

	Args:
		mat (ndarray): input array to do pooling on the first 2 dimensions.
		f (int): pooling kernel size.
	Keyword Args:
		stride (int or None) : stride in row/column. If None, same as <f>,
					i.e. non-overlapping pooling.
		method (str) : 'max for max-pooling,
  				'mean' for average-pooling.
		pad (bool) : pad <mat> or not. If true, pad <mat> at the end in
				y-axis with (f-n%f) number of nans, if not evenly divisible,
				similar for the x-axis.
		return_max_pos (bool) : whether to return an array recording the locations
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
        
        
def NMS_1st(box_list, obj_list, thresh):
	"""
	(Author : Adrien Anthore)
	Intra-patch Non-max suppression.
	Take the result boxes and objectnesses
	from a a YOLO algorithm on a subset of a
	Mosaïc and perform an NMS among them.
	
	THIS VERSION IS NOT TESTED ON REAL DATA !
	Changes will come especially on the suppresion
	criteria (NMS threshold and objectness threshold).
	Also the box_list and obj_list might be pack on a
	single ndarray.
	
	box_list : ndarray(dtype=float)
	obj_list : array(dtype=float)
	thresh : float
	
	box_list : liste of the box found by
		YOLO method. each box contains the 2
		anchor points :
		[Xmin, Ymin, Xmax, Ymax]
		That are the bottom left and upper right
		corner of the box.
	obj_lsit : objectness of each box (0<obj<1)
	thresh : NMS Threshold compared with the
		Intersection over Union (IoU)
		from a box with the remaining (1 by 1).
		
	return :
	kept_box : ndarray(dtype=float)
	
	kept_box : The box not suppressed that can be concidered
		as detection inside a patch.
	"""
	
	n, d = box_list.shape
	
	kept_box = np.zeros((n, d+1))
	
	sort_ind = np.argsort(-obj_list)
	
	sort_box = box_list[sort_ind]
	sort_obj = obj_list[sort_ind]

	
	kept_cnt = 0
	
	while len(sort_obj) > 0:
	
		kept_box[kept_cnt,:4] = sort_box[0]
		kept_box[kept_cnt,4] = sort_obj[0]
	
		sort_box = sort_box[1:,:]
		sort_obj = sort_obj[1:]
		
		temp_box = sort_box
		temp_obj = sort_obj
		
		for i in range(len(sort_obj)):

			iou = fct_IoU(kept_box[kept_cnt,:4], temp_box[i])
			
			if(iou > thresh) or (temp_obj[i] < 0.05):
			
				ind= np.where(sort_box==temp_box[i])[0][0]
			
				sort_box = np.delete(sort_box, ind, 0)
				sort_obj = np.delete(sort_obj, ind, None)
				
				
		kept_cnt+=1
		
	
	return kept_box[:kept_cnt]
