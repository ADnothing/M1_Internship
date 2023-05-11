#!/usr/bin/env python

#Adrien Anthore, 11 mai 2023
#cross_test.py

#The purpose of this code is to generate random positions of source to test
#fortunate cross match with known catalog and then estimate the quality of real detetion
#that will then get cross matched with those catalogs.

import numpy as np

from astropy.io import fits
from astropy import wcs as WCS
from astropy.coordinates import SkyCoord

from tqdm import tqdm

from numba import jit

#=== Global variables ===
#Those density are estimated around my test
#area, they are not necesserally true
#but it is sufficient in that purpose
DESI_density = 52550 #target/degree^2
ALLWISE_density = 14663 #target/degree^2

test_surface = 2.14 #degree^2

hdul = fits.open("im_168_47.fits")
image = hdul[0].data #The mosaic
hdr = hdul[0].header #The header of the file
hdul.close()
wcs = WCS.WCS(hdr)

l, c = image.shape

RA0, DEC0 = wcs.wcs_pix2world(0, 0, 0)
RA1, DEC1 = wcs.wcs_pix2world(c, l, 0)

if RA0>RA1:
	maxRA = RA0
	minRA = RA1
else:
	maxRA = RA1
	minRA = RA0
	
if DEC0>DEC1:
	maxDEC = DEC0
	minDEC = DEC1
else:
	maxDEC = DEC1
	minDEC = DEC0
#========================

@jit(nopython=True)
def gen_pos(nb_sou):
	"""
	Generate a determine number of random position
	distributed uniformly in the tested area.
	
	nb_sou : int
	
	nb_sou : number of coordinates to be generated
	
	return :
	coord_arry : float
	
	coord_array : an array of shape (nb_sou, 2)
		that contains a all the RA DEC
		of the source uniformly distributed.
	"""

	coord_array = np.zeros((nb_sou, 2))
	
	for i in range(nb_sou):
	
		RA = np.random.random() * (maxRA - minRA) + minRA
		DEC = np.random.random() * (maxDEC - minDEC) + minDEC

		
		coord_array[i,:] = np.array([RA, DEC])
		
	return coord_array
	
		
def gen_test(dens, surface, n=1, name="Xmatch_test"):
	"""
	Generate multiples test set of random coordinates and
	save them in txt file.
	The number of coordinates are determined from the density
	of source in the area with a modulation following a normal
	distribution.
	
	Note : the generated file will overwrite if a file of the
	same name already exist. The format is : an header with the title
	of the column, the first column is the RA and the second is the DEC.
	This file can be read with Aladin software and then be cross_matched with
	real catalog.
	
	dens : float
	surface : float
	n : int (optionnal, default=1)
	name : str (optionnal default = "Xmatch_test")
	
	dens : the density of source in the local area
	surface : the tested surface area
	n : the number of test set to be generated
	name : the name of the file that will be saved
		int txt format.
	
	return
	None.
	"""

	mean = dens*surface
	nb_sou = abs(np.round(np.random.normal(mean, 0.05*mean)).astype(int))
	
	for i in range(n):
	
		file_name = name+"_%d.txt"%(i+1)
		print("Generating %s ..."%(file_name))
	
		coords = gen_pos(nb_sou)
		np.savetxt(file_name, coords, fmt="%.3f", delimiter='\t', header="_RAJ2000\t_DECJ2000", comments='')
		print("Done")
	
	


if __name__ == "__main__":
	
	gen_test(DESI_density, test_surface, n=10, name="./xmatch_test/DESI_Xmatch_test")
	gen_test(ALLWISE_density, test_surface, n=10, name="./xmatch_test/ALLWISE_Xmatch_test")


#Feel free to modify the main function as well as the global variables
#to adapt to a different contexts/test sets.
