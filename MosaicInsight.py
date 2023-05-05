#!/usr/bin/env python

#Adrien Anthore, 17 avril 2023
#MosaicInsight.py

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy import wcs as WCS
from astropy.coordinates import SkyCoord

from corr_fits import *



def show_fits(fits_file, frame=None, Title='', cmap='hot', vmin=1e-4, vmax=4e-4, fill=False, save=False):
	"""
	Display the image of a fits file, entirely or within
	a determine frame, with a colormap to chose (including
	the min and max value). You also can fill the hole
	created by nan values. Can be save as Title.pdf.
	Note : if the title contains blank spaces,
	they will be replaced by "_".
	
	fits_file : str
	frame : tuple(float, float, float) (optional, default=None)
	Title : str (optional, default='')
	cmap : str (optional, default='hot')
	vmin : float (optional, default=1e-4)
	vmax : float (optional, default=4e-4)
	fill : bool (optional, defualt=False)
	save : bool (optional, default=Fault)
	
	fits_file : the path to the fits file
		that contains the image.
	frame : the frame you want to display,
		format : (RA, DEC, side)
		where RA and DEC are ICRS angles
		and side the side of the frame in
		pixels.
		note : the side is such that
		(RA,DEC) si the center of the frame.
	Title : Title of the figure
	cmap : matplotlib colormap
	vmin : min value in the colormap
	vmax : max value in the colormap
	fill : if True, fill the hole with the function
		fill_im_hole.
	save : if True save the image in pdf format
		
	return
	None.
	"""

	#Extracting image + header of the fits
	hdul = fits.open(fits_file)
	image = hdul[0].data #The mosaic
	hdr = hdul[0].header #The header of the file
	hdul.close()
	wcs = WCS.WCS(hdr) #We get the ICRS coordinates here as well as the convestion ICRS<->Pixels
	
	
	#Redefinition of the image
	
	#Changing the frame
	if(frame!=None):
		RA, DEC, side = frame
		Px, Py = wcs.wcs_world2pix(RA, DEC, 0)
		image = image[int(Py - side/2):int(Py + side/2) , int(Px - side/2):int(Px + side/2)]
		wcs.wcs.crval = [RA, DEC]
		wcs.wcs.crpix = [side/2, side/2]
	#Filling nan values
	if(fill):
		image = fill_im_hole(image)
	
	#plot
	plt.figure(figsize=(20,22))
	plt.subplot(projection=wcs)
	plt.imshow(image, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
	plt.xlabel("RA (ICRS) [h m s]", size='xx-large')
	plt.ylabel("DEC (ICRS) [deg ' \"]", size='xx-large')
	plt.title(Title, size='xx-large')
	clb=plt.colorbar()
	clb.ax.set_title(label="Flux [mJy/Beam]",fontsize=15)
	if(save):
		plt.savefig(Title.replace(" ","_")+".pdf")
	plt.show()


#==========================================================================================


def show_image(image, wcs=None, Title='', cmap='hot', vmin=1e-4, vmax=4e-4, save=False):
	"""
	Display an image of the sky (2D array from fits file).
	Can take into account wcs.
		
	image : ndarray(dtype=float)
	wcs : astropy wcs object (optionnal, default=None)
	Title : str (optional, default='')
	cmap : str (optional, default='hot')
	vmin : float (optional, default=1e-4)
	vmax : float (optional, default=4e-4)
	save : bool (optional, default=False)
	
	image : a 2D array of flux/beam value
	wcs : The world coordinate system of
		the image from the header
		of the fits file.
	Title : Title of the figure
	cmap : matplotlib colormap
	vmin : min value in the colormap
	vmax : max value in the colormap
	save : if True save the image in pdf format
	
	
	return
	None.
	"""
	
	plt.figure(figsize=(20,22))
	plt.subplot(projection=wcs)
	plt.imshow(image, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
	plt.xlabel("RA (ICRS) [h m s]", size='xx-large')
	plt.ylabel("DEC (ICRS) [deg ' \"]", size='xx-large')
	plt.title(Title, size='xx-large')
	clb=plt.colorbar()
	clb.ax.set_title(label="Flux [mJy/Beam]",fontsize=15)
	if(save):
		plt.savefig(Title.replace(" ","_")+".pdf")
	plt.show()


#==========================================================================================


def calc_axis(coord, list_fits, method='mean'):
	"""
	Compute the major and minor axis of a target
	from the coordinates and a list of mosaic.
	Maj and Min are computed as the full width at
	half maximum of a single gaussian with no close
	neighbour an low background signal
	within a box of 30x30 pixels.
	If you use multiple fits, the result can change
	if you use a median or a mean. The default here is
	a mean for which all the fits have the same weights.
	
	This method has a bias : the target are considered
	invariant by rotation. Therefore the target will
	most likely found to be circular (Min==Maj).
	
	coord : tuple.
	list_fits : list[str]
	method : str (optional, default='mean')
	
	coord : (RA, DEC) in degrees.
	list_fits : list of path to the mosaics.
	method : the method you want to use
		(between 'median' and 'mean', default
		is 'mean')
	
	return :
	Maj : float
	Min : float
	
	Maj : Major axis in arcsec
	Min : Minor axis in arcsec
	"""
	
	#Arrays of Major and Minor axis that will be flatten with a mean/median
	array_Maj = np.zeros(len(list_fits))
	array_Min = np.zeros(len(list_fits))
	
	
	for i in range(len(list_fits)): #Incrementing on the index of list_fits 
					#(which are the same index for array_Maj and array_Min)
	
		fits_file = list_fits[i]
	
		hdul = fits.open(fits_file)
		image = hdul[0].data #The mosaic
		hdr = hdul[0].header #The header of the file
		hdul.close()
		
		wcs = WCS.WCS(hdr) #We get the ICRS coordinates here as well as the convestion ICRS<->Pixels
		
		RA, DEC = coord
		x, y = wcs.wcs_world2pix(RA, DEC, 0) #RA & DEC (ICRS) -> X & Y (pixels)
		
		Xs = np.linspace(-15,15,30)
		F1 = image[int(y)-15:int(y)+15,int(x)] #flux on the first axis
		F2 = image[int(y),int(x)-15:int(x)+15] #flux on the second axis
		
		#Here the discretisation from np.where induce a small error
		#The order of magnitude of the error change with respect
		#to the position on the mosaic.
		xpeak = x + Xs[np.where(F2==F2.max())[0][0]] #x pixel of the maximum
		ypeak = y + Xs[np.where(F1==F1.max())[0][0]] #y pixel of the maximum
		
		
		
		#The method to get the FWHM is :
		#We substract half of the maximum to the flux,
		#then we make a spline interpolation, and from the
		#interpolation we catch the roots of the F-max(F)/2.
		#The distance between the roots is the FWHM.
		#Error that can occures : if they is too much signal
		#close by or the SNR is too week, they might be
		#more or less than 2 roots. Therfore the error message
		#will be on the expected number of value to unpack.
		
		#Precisions :
		#You will find numbers in the variable names.
		#The first number is related to the axis (1 or 2)
		#the second number is related to order of the positions.
		#(1 -> before the peak ; 2 -> after the peak)
		
		#1st axis
		spline1 = UnivariateSpline(Xs, F1-np.max(F1)/2, s=0)
		r11, r12 = spline1.roots()
		FWHM1 = r12-r11

		#Pixels coordinates of the bounds
		Px11 = xpeak
		Py11 = ypeak-FWHM1/2
		Px12 = xpeak
		Py12 = ypeak+FWHM1/2
		
		#Positions in ICRS of the bounds
		C11 = WCS.utils.pixel_to_skycoord(Px11, Py11, wcs, 0)
		C12 = WCS.utils.pixel_to_skycoord(Px12, Py12, wcs, 0)

		#separation returns the angular separation between the
		#two points.
		ax1 = C11.separation(C12).arcsecond
		
		#2nd Axis
		spline2 = UnivariateSpline(Xs, F2-np.max(F2)/2, s=0)
		r21, r22 = spline2.roots()
		FWHM2 = r22-r21
		
		#Pixels coordinates of the bounds
		Px21 = xpeak-FWHM2/2
		Py21 = ypeak
		Px22 = xpeak+FWHM2/2
		Py22 = ypeak
		
		#Positions in ICRS of the bounds
		C21 = WCS.utils.pixel_to_skycoord(Px21, Py21, wcs, 0)
		C22 = WCS.utils.pixel_to_skycoord(Px22, Py22, wcs, 0)
		
		#separation returns the angular separation between the
		#two points.
		ax2 = C21.separation(C22).arcsecond



		#Sorting between Maj and Min
		if(ax1>=ax2):
			array_Maj[i] = ax1
			array_Min[i] = ax2
		else:
			array_Maj[i] = ax2
			array_Min[i] = ax1
			
			
			
			
	if (method=='median'):
		Maj = np.median(array_Maj)
		Min = np.median(array_Min)
	elif (method=='mean'):
		Maj = np.mean(array_Maj)
		Min = np.mean(array_Min)
	else:
		print("Unknown method : %s\nMean performed instead."%method)
		Maj = np.mean(array_Maj)
		Min = np.mean(array_Min)
	
	
	
	return Maj, Min	


#==========================================================================================	


def save_subfits(fits_file, frame, display=False):
	"""
	Save a subset fits image and display it
	if asked.
	
	Note : in astropy module, there is a way to do
	it more efficiently using the cutout function.
	At this moment I have still not figure out how
	it works, but this function works so I wont try
	to change it as long as it does what I want.
	
	fits_file : str
	frame : tuple(float, float, float)
	display : bool (optional, default=False)
	
	fits_file : the path to the fits file
	frame : the frame you want to keep,
		format : (RA, DEC, side)
		where RA and DEC are ICRS angles
		and side the side of the frame in
		pixels.
		note : the side is such that
		(RA,DEC) si the center of the frame.
	display : if True display the subset.
	
	return
	None.
	"""
	
	#Open the file
	hdul = fits.open(fits_file)
	hdr = hdul[0].header
	wcs = WCS.WCS(hdr)
	
	#Definition of the frame in pixel
	RA, DEC, side = frame
	Px, Py = wcs.wcs_world2pix(RA, DEC, 0)
	
	#reshape of the image
	hdul[0].data = hdul[0].data[int(Py - side/2):int(Py + side/2) , int(Px - side/2):int(Px + side/2)]
	
	#changing the wcs
	wcs.wcs.crval = [RA, DEC]
	wcs.wcs.crpix = [side/2, side/2]
	
	#Update the header with the new WCS information
	hdr.update(wcs.to_header())
	
	#display
	if display:
		show_image(hdul[0].data, wcs=wcs, Title='Subset of the fits %s'%fits_file)
	
	#saving the result
	name_file = "sub_"+fits_file.split("/")[-1]
	hdul.writeto(name_file, overwrite=True)
	
	hdul.close()
	
