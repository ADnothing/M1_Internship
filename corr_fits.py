#!/usr/bin/env python

#Adrien Anthore, 26 avril 2023
#corr_fits.py

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

from glob import glob
import os

from tqdm import tqdm

from Patch_Management import *


def fill_im_hole(image, mask=None, method='median', art_noise=False):
	"""
	Fill the nan value inside an image.
	The value is compute with the mean
	or the median value of the image
	(default is median). If asked, add
	a gaussian random noise to the
	filling.
	If the image is full of nan, fill
	the image with 0. only.
	The center value are value are calculated
	with respect to a mask.
	The artificial gaussian noise might not be
	significantly useful, it might even get
	worse results (to test in different circomptances)
		
	image : ndarray(dtype=float)
	mask : ndarray(dtype=bool) (optional)
	method : str (optional)
	art_noise : bool (optional)
	
	image : a 2D array of flux/beam value
	mask : a condition mask on the value.
	method : the method you want to use
		in order to compute the over
		all background signal.
		(between 'median' or mean', default
		is 'median)
	art_noise : if True, fill the hole with
		an artificial gaussian noise
		with a mean value of the center taken
		from the mean/median and with the global
		std of the image (with respect to the mask).
		
	return :
	fill_image : ndarray(dtype=float)
	
	fill_image : a 2D array of flux/beam value
		with the same dimention than image
		but with all the nan value replaced by
		the over all background signal.
	"""
	
	
	if(method=='mean'):
		center = np.nanmean(image[mask])
	elif(method=='median'):
		center = np.nanmedian(image[mask])
	else:
		print("Unknown method %s\nMedian performed instead."%method)
		center = np.nanmedian(image[mask])
	
	#Artificial noise maker :
	#We take a "background image" with a gaussian law.
	#We put 0 in the "background image" where the image has value
	#Then we change all the nan value in the image to 0.
	#Then we summ the 2 images.
	#else : uniformly replace the nan value by the center value
	#computed before.
	if(art_noise):
		std = np.std(image[mask])
		background = np.random.normal(center, std, image.shape)
		background = np.where(~np.isnan(image), 0, background)
		
		image = np.nan_to_num(image, nan=0)
		fill_image = image + background
	
	
	if(np.count_nonzero(~np.isnan(image))==0):
		fill_image = np.nan_to_num(image, nan=0)
	else:
		fill_image = np.nan_to_num(image, nan=center)
	
	return fill_image

	
#==========================================================================================	
	

def corr_fits(dir_path):
	"""
	This function is use to save corrected fits
	and save them in the path : path/../corr_fits/fitsname_corr.fits
	The correction consist in replacing the nan values by the
	median of the flux of the fits with the function
	fill_im_hole.
	
	dir_path : str
	
	dir_path : path to all the firs file considered
		format : path/
	
	return
	None.
	"""
	
	#Checking if corr_fits exists, else
	#create it.
	if not os.path.exists(dir_path+"../corr_fits"):
		os.makedirs(dir_path+"../corr_fits")
	

	#Geting the list of all the fits file in the directory
	fits_list = glob(dir_path+"*.fits")
	
	for fits_file in tqdm(fits_list):
	
		hdul = fits.open(fits_file)
		hdul[0].data = fill_im_hole(hdul[0].data, method='mean') #replacing the image with the corrected image
		
		name_file = fits_file.split("/")[-1][:-5]+"_corr.fits" #new name
		
		hdul.writeto(dir_path+"../corr_fits/"+name_file) #save the fits with the new image and the same header
		
		hdul.close()

	
#==========================================================================================	
	

def input_gen(data, patch_size=256, patch_shift=240, orig_offset=64):
	"""
	Generate input from fits data image.
	To generate the input data, we travel the data with patches of a
	determined size and shift from one patch to an other.
	
	data : ndarray(dtype=float)
	image_size : int (optional)
	patch_shift : int (optional)
	orig_offset : int (optional)
	
	data : Input ndarray containing image data.
	patch_size : size of the image patches to be generated.
	patch_shift : amount to shift each patch by.
	orig_offset : offset to the original image.
	
	return :
	input_data : ndarray(dtype=float)
	
	input_data : output ndarray containing image patches.
	
	"""
	
	#Get the map size in pixel from the
	#first dimension of the input data ndarray
	map_pixel_size = np.shape(data)[0]
	
	#Calculate the number of areas in the width and height directions
	nb_area_w = int((map_pixel_size-orig_offset)/patch_shift) + 1
	nb_area_h = int((map_pixel_size-orig_offset)/patch_shift) + 1
	
	#Calculate the total number of patches to be generated
	nb_patches = nb_area_w*nb_area_h
	
	#Initialize an ndarray for storing the image patch data
	patch = np.zeros((patch_size, patch_size), dtype="float32")
	
	#Initialize an ndarray for storing the input data
	#First dimension : patches
	#Second dimensio : each pixels of the patches
	input_data = np.zeros((nb_patches,patch_size*patch_size), dtype="float32")

	print("Input generation...")

	#Iterate over the number of patches to be generated
	for i_d in tqdm(range(0,nb_patches)):
	
		#Calculate the x and y indices for the current patch
		p_y = int(i_d/nb_area_w)
		p_x = int(i_d%nb_area_w)
		
		#Initialize the min and max x and y coordinates for the patch
		px_min = 0
		px_max = patch_size
		py_min = 0
		py_max = patch_size
		
		#Calculate the min and max x and y coordinates for the patch based on the current x and y indices
		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + patch_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + patch_size - orig_offset
		
		# Initialize a flag for whether any of the patch coordinates are out of bounds
		set_zero = 0
	
		# If any of the patch coordinates are out of bounds, 
		#set the corresponding min and/or max values and set the flag to 1
		if(xmin < 0):
			px_min = -xmin
			xmin = 0
			set_zero = 1
		if(ymin < 0):
			py_min = -ymin
			ymin = 0
			set_zero = 1
		if(xmax > map_pixel_size):
			px_max = patch_size - (xmax-map_pixel_size)
			xmax = map_pixel_size
			set_zero = 1
		if(ymax > map_pixel_size):
			py_max = patch_size - (ymax-map_pixel_size)
			ymax = map_pixel_size
			set_zero = 1
		
		#If any of the patch coordinates are out of bounds, set the patch data to 0	
		if(set_zero):
			patch[:,:] = 0.0
	
		#extract the data for the current patch, and fill in any holes with the mean value of the data
		#filled with 0. if the patch is full of nan.

		sub_data = np.flip(data[xmin:xmax,ymin:ymax],axis=0)
		patch[px_min:px_max,py_min:py_max] = fill_im_hole(sub_data, method='mean')
				
		#flattening the data
		input_data[i_d,:] = patch.flatten("C")

	print("Input generated !")
		
	return input_data
		
