#!/usr/bin/env python

#Adrien Anthore, 05 mai 2023
#make_cat.py

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../')
from corr_fits import *

from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy import wcs as WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.nddata import Cutout2D

from astrodendro import Dendrogram, pp_catalog

from scipy.optimize import curve_fit



def patch_gen(image, wcs, patch_size=512, patch_shift=480, orig_offset=60):
	"""
	Divide an image from a fits file in patches with respect to its
	wcs.
	
	image : ndarray(dtype=float)
	wcs : astropy object wcs
	patch_size : int (optional, default=512)
	patch_shift : int (optional, default=480)
	orig_offset : int (optional, default=60)
	
	image : the image from the fits file
	wcs : the wcs corresponding to the fits header
	patch_size : size of the image patches to be generated
	patch_shift : amount to shift each patch by
	orig_offset : offset to the original image
	
	return :
	patches : list[astropy object Cutout2D]
	
	patches : list of the patches, each ellement is an
		astropy object from which you can get attributes
		such as data (patches[i].data)
		or wcs (patches[i].wcs)
	"""
	
	#Get the map size in pixel from the
	#first dimension of the input data ndarray
	map_pixel_size = np.shape(image)[0]
	
	#Calculate the number of areas in the width and height directions
	nb_area_w = int((map_pixel_size-orig_offset)/patch_shift) + 1
	nb_area_h = int((map_pixel_size-orig_offset)/patch_shift) + 1
	
	#Calculate the total number of patches to be generated
	nb_patches = nb_area_w*nb_area_h
	
	#initialisation of the list of patches
	patches = []
	
	print("Patches generation...")
	
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

		# If any of the patch coordinates are out of bounds, 
		#set the corresponding min and/or max values
		if(xmin < 0):
			px_min = -xmin
			xmin = 0

		if(ymin < 0):
			py_min = -ymin
			ymin = 0

		if(xmax > map_pixel_size):
			px_max = patch_size - (xmax-map_pixel_size)
			xmax = map_pixel_size

		if(ymax > map_pixel_size):
			py_max = patch_size - (ymax-map_pixel_size)
			ymax = map_pixel_size


		#making the cutout and append to the list
		cutout = Cutout2D(image, (xmin+patch_size/2, ymin+patch_size/2), patch_size*u.pix, wcs=wcs)
		patches.append(cutout)
		
	print("Done.")
	
	return patches
			
#===============================================================================================================

def get_params(image):
	"""
	Compute the parameters min_val and
	delta use for the function compute in
	astrodendro.Dendrogram (see online doc)
	Those parameters are caught by fitting a gaussian
	in the distribution of the number of pixels
	as a function of their values.
	
	Note : We have to use not corrected image, if the image
	with filled holes is used, the distribution might not
	look like a gaussian and the function curve_fit might not
	converge. (resulting a safty error to occure)
	
	image : ndarray(dtype=float)
	
	image : the image array.
	
	return
	min_val : float
	delta : float
	
	min_val : see compute in astrodendro doc
	delta : see compute in astrodendro doc
	"""

	flat_image = image[~np.isnan(image)] #this function gets rid of nan values and make the ndarray flat
	n = len(flat_image)
	
	#Here we create a window that will most likely contain
	#the gaussian shaped distribution of the pixel value
	median = np.median(flat_image)
	std = np.std(flat_image)
	
	inf = median-0.5*std
	sup = median+0.5*std
	
	mask = np.logical_and(flat_image>inf, flat_image<sup) #This mask will be applied to the flat_image to create the window
	
	#Here we create the histogram which will be fit
	bins = np.linspace(flat_image[mask].min(), flat_image[mask].max(), int(n*(300/5e7)))
	hist, bins_edges = np.histogram(flat_image, bins=bins)
	
	#It appears that applying a mask according to the hist value may change a bit
	#the fit. For instance taking only the values that are superior to 10% of
	#the hist max value.
	#After few testing I didn't find the result to be significant but it 
	#may be interresting to go trhough.
	Y = hist
	#As bins_edges correrspond to the edges, we take the mean between
	#the left and right edges of the bin to get the center points
	X = (bins_edges[1:] + bins_edges[:-1])/2
	#the K factor in the gaussian model is simply a scale factor and it is
	#not important to be saved.
	gauss_model = lambda x, mean, std, K: (K/std)*np.exp(-((x-mean)**2/std**2))
	
	#Note : fitting a function that has 3 or more parameters is
	#rather difficult for curve_fit. It is important to have 
	#enough data points (X) and also a smooth function to fit.
	#Otherwise it may not converge or worse it may return
	#incoherant values.
	popt, _ = curve_fit(gauss_model, X, Y)
	
	min_val = popt[0]
	delta = popt[1]

	return min_val, delta
				
#===============================================================================================================

def Crea_dendrogram(fits_file):
	"""
	Generate dendrograms and catalogs from a fits file
	using the library astrodendro.
	See the doc : https://dendrograms.readthedocs.io/en/stable/
	
	fits_file : str
	
	fits_file : Path to the fits file
	
	return :
	None.
	"""

	#Get the image withits wcs
	hdul = fits.open(fits_file)
	image = hdul[0].data
	hdr = hdul[0].header
	hdul.close()
	wcs = WCS.WCS(hdr)
	
	#Definition of the meta data that will be use to generate the
	#pp_catalog.
	#data_unit correspond to the data unit of the pixels
	#Spatial_scale correspond to the reolution of the instrument.
	#beam_major/beam_minor correspond to the beam size along
	#its 2 axis.
	#see the doc for more details.
	metadata = {}
	metadata['data_unit'] = u.Jy/u.beam
	metadata['spatial_scale'] =  hdr["CDELT2"]*3600 * u.arcsec
	metadata['beam_major'] =  hdr["BMAJ"]*3600 * u.arcsec
	metadata['beam_minor'] =  hdr["BMIN"]*3600 * u.arcsec

	#see the parameters from the compute function
	#of astrodendro and see the algorithm in the doc.
	min_val, delta = get_params(image)
	
	print("min_value :",min_val)
	print("delta :", delta)
	
	patches = patch_gen(image, wcs)
	nb_patch = len(patches)

	print("Catalogs generation...")
	#The catalog will be stored in a single file
	#you may change the path of the destination here.
	name_file = "./dendrocat/"+fits_file.split("/")[-1][:-5]+"_DendCat.txt"
	f = open(name_file, 'w')
	f.write("_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\n")
	f.close()
	
	#min number of pixel to be taken into account
	#to compute a dendrogram.
	#see compute function parameters.
	min_pix_nb = 2*int(6/(hdr["CDELT2"]*3600))

	for i in tqdm(range(nb_patch)):
	
		patch = patches[i].data
		patch_wcs = patches[i].wcs
	
		#Checking if the current patch is empty
		#if it's not, compute the dendrogram
		#and generate the catalog.
		#Note : here I put a minimum number of pixel
		#at 16. This is purely arbitrary and might be
		#more relevant to find a criteria based on the
		#resolution, pixel size, and number of pixels per patches.
		if (np.count_nonzero(~np.isnan(patch))>16):
			#Note : filling the holes (nan values) helps
			#the compute function to run faster and also
			#helps to run properly.
			patch = fill_im_hole(patch)
			
			#The 
			d = Dendrogram.compute(patch, min_value=min_val, min_delta=5*delta, min_npix=min_pix_nb, wcs=patch_wcs, verbose=True)
		

			if len(d.leaves)>0:
				#Note : computing the leaves instead of the full dendrograms
				#prevent the catalog to put sereal points in the same coordinates.
				#See the dendrograms structure.
				cat = pp_catalog(d.leaves, metadata)
				
				#cat.pprint(show_unit=True, max_lines=10)

				#The information we keep in the catalog
				RA, DEC = patch_wcs.wcs_pix2world(cat["x_cen"][:], cat["y_cen"][:], 0)
				flux = cat["flux"][:]*1e3
				Maj = cat["major_sigma"][:]
				Min = cat["minor_sigma"][:]

				#We use 3 masks to avoid some problems :
				#mask1 : remove any source with no spatial extension.
				#mask2 : remove sources that are too much elongated
				#	this mask avoid some artefacts to be taken into account
				#	(this criteria is not sufficient and must be changed)
				#mask3 : remove sources with too low flux
				mask1 = Maj != 0

				RA = RA[mask1]
				DEC = DEC[mask1]
				flux = flux[mask1]
				Min = Min[mask1]
				Maj = Maj[mask1]

				mask2 = Min/Maj >= 5e-2

				RA = RA[mask2]
				DEC = DEC[mask2]
				flux = flux[mask2]
				Min = Min[mask2]
				Maj = Maj[mask2]
				
				mask3 = flux > 1e-1

				RA = RA[mask3]
				DEC = DEC[mask3]
				flux = flux[mask3]
				Min = Min[mask3]
				Maj = Maj[mask3]
				

				#append to the end of the file.
				f = open(name_file, 'a')
				np.savetxt(f, np.array([RA,DEC,flux,Maj,Min]).T,fmt=['%.3f', '%.3f', '%.6e', '%.3f', '%.3f'] , delimiter='\t', comments='')
				f.close()
			else:
				print("No objects found in patch")

		
	print("Catalogs generated !")

"""	
if __name__=="__main__":
	Crea_dendrogram("../P7Hetdex11.fits")
"""
	
