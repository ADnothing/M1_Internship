#!/usr/bin/env python

#Adrien Anthore, 19 mai 2023
#corr_cat.py

import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

import os


def Rexcl(flux, P1=0, R1=6, P2=5, R2=600):
	"""
	Rejection radius function.
	"""

	c = (R2-R1*np.exp(P2-P1))/(1-np.exp(P2-P1))
	k = (R1-c)*np.exp(-P1)
	
	return k*np.exp(np.log10(flux)) + c
	
def update_progress(progress):
    """
    Function to update the progress bar in the console.
    """
    bar_length = 100  # Number of characters in the progress bar
    filled_length = int(bar_length * progress)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    percentage = int(progress * 100)
    print(f'\rProgress of the cleaning : |{bar}| {percentage}% ', end='')
	
	
def clean_cat(name_file, leaf=False, leafdir=""):
	"""
	Clean the catalog generated with crea_dendrogram
	to suppress multiple detections as well as
	supposedly false detection arround bright sources
	(artefacts radio)
	
	name_file : str
	leaf : bool (optional, default = False)
	leafdir : str (optional, default = "")
	
	name_file : path to the catalog
	leaf : if True take into account the
		leaves saved in fits file by 
		crea_dendrogram
	leafdir : directory where the leaves fits
		are stored
		
	return :
	None.
	"""
	excl = 0 #Number of exclusion (is displayed for the user)
	
	#The data are loaded and sorted by decreasing flux
	#We consider the flux as some kind of a confidence factor.
	data = np.loadtxt(name_file, skiprows=1)

	if data.shape != (7,) and data.shape != (0,):
		data = data[np.argsort(-data[:,2])]
	
		i = 0 #progress indice
		while i<data.shape[0]-1:

			flux = data[i,2]
			R = Rexcl(flux)
			
			c1 = SkyCoord(data[i,0]*u.deg, data[i,1]*u.deg, frame='icrs')
			c2 = SkyCoord(data[i+1:,0]*u.deg, data[i+1:,1]*u.deg, frame='icrs')
			sep_array = c1.separation(c2)
			
			#Here is the exclusion condition,
			#A source to be excluded must :
			#Be closer than 6 arcsecond to the current source
			#or be inside the rejection radius with a flux lower than 10 mJy
			excl_ind = np.where(np.logical_or(np.logical_and(data[i+1:,2]<1e1,sep_array.arcsecond < R), sep_array.arcsecond<2))[0] + i + 1
			
			for ind in sorted(excl_ind, reverse=True):
				
				data = np.delete(data, ind, 0)
				excl+=1
		
			i+=1
		
			progress = (i + 1) / (data.shape[0]-1)
			update_progress(progress)
	
	
		#Deal with the leaves if asked to
		if leaf:	
			path = leafdir
		
			indexes = data[:,-1]
			for fitsfile in os.listdir(path):
				try:
					n = int(fitsfile.split("_")[1].split(".")[0])
					if n not in indexes:
						file2remove = os.path.join(path, fitsfile)
						os.remove(file2remove)
				except (IndexError, ValueError):
					continue
					
				
		print("\nNumber of exclusions : %d"%excl)
		f = open(name_file, 'w')
		np.savetxt(f, data,fmt=['%.6f', '%.6f', '%.6e', '%.3f', '%.3f', '%.1f', '%d'], header="_RAJ2000\t_DECJ2000\tSpeakTot\tMaj\tMin\tPA\tleaf\n" , delimiter='\t', comments='')
		f.close()

