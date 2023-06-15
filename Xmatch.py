#!/usr/bin/env python

#Adrien Anthore, 19 mai 2023
#Xmatch.py

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy import wcs as WCS
from astropy.coordinates import SkyCoord

from numba import jit

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

import time
import os.path

from make_cat import delta_test, min_test, deltamin_test
from corr_cat import clean_cat

@jit(nopython=True, cache=True, fastmath=False)
def match_coord(cat2cross, refcoord, sep):
	"""
	This function is an attempt to reproduce a cross matching
	function where sources of the input catalog (cat2cross)
	can be associated with a maximum of 1 source of the reference
	catalog. Once the a target of the reference catalog got associated,
	it can't be associated with an other source.
	
	This function got a major problem : it doesn't take into account the
	best association and just go through the input catalog sources by sources.
	It can cause sources to be associated with a wrong target and thus
	have sources left with no counterpart.
	Note : this function has been written in order to be master the way
	the sources can be associated and to avoid recall > 100%.
	
	cat2cross : ndarray (dtype=float)
	refcoord : ndarray (dtype=float)
	sep : float
	
	cat2cross : input catalog
	refcoord : reference catalog
	sep : maximum sepration between source and target
	
	Note : for both cat2cross and refcoord, the 2 first columns
	must be RA and DEC in this order.
	
	return : 
	crosscat : ndarray (dtype=float)
	counterpart : ndarray (dtype=float)
	
	crosscat : cross matched catalog
	counterpart : targets that correspond to the
		cross matched sources (in the same order)
	
	"""
	
	refcoord = np.ascontiguousarray(refcoord)
	
	crosscat = np.zeros(cat2cross.shape)
	counterpart = np.zeros(refcoord.shape)
	
	i = 0
	for j in range(crosscat.shape[0]):
		if refcoord.shape[0] > 0:
			source = cat2cross[j,:]
			
			lon1 = source[0]*np.pi/180.
			lat1 = source[1]*np.pi/180.
			lon2 = refcoord[:,0]*np.pi/180.
			lat2 = refcoord[:,1]*np.pi/180.
			
			sdlon = np.sin(lon2 - lon1)
			cdlon = np.cos(lon2 - lon1)
			slat1 = np.sin(lat1)
			slat2 = np.sin(lat2)
			clat1 = np.cos(lat1)
			clat2 = np.cos(lat2)
			
			num1 = clat2 * sdlon
			num2 = clat1 * slat2 - slat1 * clat2 * cdlon
			denominator = slat1 * slat2 + clat1 * clat2 * cdlon
			
			separation = np.arctan2(np.hypot(num1, num2), denominator)*(180.*3600.)/np.pi

			select = refcoord[separation <= sep]
			fine_sep = separation[separation <= sep]
			
			if len(select)>0:
				crosscat[i] = source
				min_index = np.argmin(fine_sep)
				counterpart[i] = select[min_index]
				
				new_refcoord = np.zeros((refcoord.shape[0]-1, refcoord.shape[1]))
				m = 0
				for n in range(refcoord.shape[0]):
					if not np.array_equal(refcoord[n, :], counterpart[i]):
					
						new_refcoord[m,:] = refcoord[n,:]
						m+=1

				refcoord = new_refcoord
				
				i+=1
		else:
			break

		
	return crosscat[:i,:], counterpart[:i,:]
			
	
	
	
	
	

def get_refcat(center, R=1.65):
	"""
	Get DESI legacy and allwise
	catalog for a determined region of the sky.
	
	Note : 1.65 deg is typically the size
		of a mosaic from lofar
		
	center : tuple(float, float)
	R : float (optionnal, default=1.65)
	
	center : coordinate of the center
		of the region in degrees
	R : radius of the region in degrees
	
	return
	DESI : astropy table object
	ALLWISE : astropy table object
	
	DESI : the catalog table of DESI
		legacy soruces
	ALLWISE : the catalog table of ALLWISE
		sources
	"""


	RA, DEC = center	
	
	center_point = SkyCoord(ra=RA, dec=DEC, unit='deg', frame='icrs')

	DESI = Vizier.query_region(center_point, radius=R*u.deg, catalog="VII/292/north")[0]
	ALLWISE = Vizier.query_region(center_point, radius=R*u.deg, catalog="II/328/allwise")[0]
	
	return DESI, ALLWISE
	
def get_LoTSS(center, R=1.65):
	"""
	Get LoTSS DR2 PyBDSF catalog 
	for a determined region of the sky.
	
	Note : 1.65 deg is typically the size
		of a mosaic from lofar
		
	center : tuple(float, float)
	R : float (optionnal, default=1.65)
	
	center : coordinate of the center
		of the region in degrees
	R : radius of the region in degrees
	
	return
	LoTSS : astropy table object
	
	LoTSS : the catalog table of LoTSS
		soruces
	"""
	
	RA, DEC = center	
	
	center_point = SkyCoord(ra=RA, dec=DEC, unit='deg', frame='icrs')
	
	LoTSS = Vizier.query_region(center_point, radius=R*u.deg, catalog="J/A+A/659/A1")[0]
	
	return LoTSS
	
	
def test_cat(cat, hdr, sep=12):
	"""
	Test the matching between LoTSS catalog and the proposed catalog
	require the header of the fits where the tested catalog derives.
	"""
	
	n = cat.shape[0]
	
	center = (hdr["CRVAL1"], hdr["CRVAL2"])
	LoTSS = get_LoTSS(center)
	
	RA_LoTSS = LoTSS["RAJ2000"]
	DEC_LoTSS = LoTSS["DEJ2000"]
	flux = LoTSS["SpeakTot"]
	maj = LoTSS["Maj"]
	min = LoTSS["Min"]
	
	n_LoTSS = len(RA_LoTSS)
	
	Xmatch, ctp  = match_coord(cat, np.array([RA_LoTSS, DEC_LoTSS, flux, maj, min]).T, sep)
	
	nb_Xmatched = Xmatch.shape[0]
	
	recall = 100*nb_Xmatched/n
	prec = 100*nb_Xmatched/n_LoTSS
	fluxreld = np.mean((Xmatch[:,2]-ctp[:,2])/ctp[:,2])
	majreld = np.mean((Xmatch[:,3]-ctp[:,3])/ctp[:,3])
	minreld = np.mean((Xmatch[:,4]-ctp[:,4])/ctp[:,4])
	
	return recall, prec, fluxreld, majreld, minreld
	

def purity_test(fits_file, min_values=10, delta_values=10):
	"""
	Performs a purity test on a FITS file by varying 
	min_value and min_delta factors independently.
	plot recall %, purity %, # of sources and computation time
	as a function min_value/min_delta.
	"""
	
	#Determine the space for min_values
	if type(min_values)==int:
		min_space = np.arange(min_values)
	elif type(min_values)==type([]):
		min_space = min_values
	elif type(min_values)==type(np.array([])):
		min_space = min_values
		
	#Determine the space for delta_values
	if type(delta_values)==int:
		delta_space = np.arange(delta_values)
	elif type(delta_values)==type([]):
		delta_space = delta_values
	elif type(delta_values)==type(np.array([])):
		delta_space = delta_values
	
	hdul = fits.open(fits_file)
	image = hdul[0].data
	hdr = hdul[0].header
	hdul.close()
	wcs = WCS.WCS(hdr)
	
	center = (hdr["CRVAL1"], hdr["CRVAL2"])

	#Get LoTSS catalog in the mosaic
	LoTSS = get_LoTSS(center)
	RA_LoTSS = LoTSS["RAJ2000"]
	DEC_LoTSS = LoTSS["DEJ2000"]
	
	recall_delta = np.zeros(len(delta_space))
	purity_delta = np.zeros(len(delta_space))
	compu_time_delta = np.zeros(len(delta_space))
	nb_sou_delta = np.zeros(len(delta_space))
	min_deltas = np.zeros(len(delta_space))
	
	recall_min = np.zeros(len(min_space))
	purity_min = np.zeros(len(min_space))
	compu_time_min = np.zeros(len(min_space))
	nb_sou_min = np.zeros(len(min_space))
	min_values = np.zeros(len(min_space))
	
	#Loop over min_space values
	for i in range(len(min_space)):
	
		min_factor = min_space[i]
	
		start = time.time()
		val = min_test(fits_file, min_factor)
		name_file = "./dendrocat/"+fits_file.split("/")[-1][:-5]+"_test_%d_min.txt"%min_factor
		clean_cat(name_file)
		dt = time.time() - start
		
		dendrocat = np.loadtxt(name_file, skiprows=1, usecols=[0,1])
		n = dendrocat.shape[0]
		
		Xmatch, _ = match_coord(dendrocat, np.array([RA_LoTSS, DEC_LoTSS]).T, 12)
		
		nb_Xmatched = Xmatch.shape[0]
		
		purity_min[i] = 100*nb_Xmatched/n
		recall_min[i] = 100*nb_Xmatched/len(RA_LoTSS)
		compu_time_min[i] = dt
		min_values[i] = val
		nb_sou_min[i] = n
	
	#Loop over min_delta values
	for i in range(len(delta_space)):
	
		delta_factor = delta_space[i]
	
		start = time.time()
		delta = delta_test(fits_file, delta_factor)
		name_file = "./dendrocat/"+fits_file.split("/")[-1][:-5]+"_test_%d_delta.txt"%delta_factor
		clean_cat(name_file)
		dt = time.time() - start
		
		dendrocat = np.loadtxt(name_file, skiprows=1, usecols=[0,1])
		n = dendrocat.shape[0]
		
		Xmatch, _ = match_coord(dendrocat, np.array([RA_LoTSS, DEC_LoTSS]).T, 12)
		
		nb_Xmatched = Xmatch.shape[0]
		
		purity_delta[i] = 100*nb_Xmatched/n
		recall_delta[i] = 100*nb_Xmatched/len(RA_LoTSS)
		compu_time_delta[i] = dt
		min_deltas[i] = delta
		nb_sou_delta[i] = n
		
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8,6), dpi=300)
	fig.subplots_adjust(hspace=0)
	fig.suptitle('Variation of min_value factor', fontsize='x-large')
	
	ax1.plot(min_values, recall_min, '--r')
	ax1.grid()
	ax1.set_ylabel("recall %", size='medium')
	
	ax2.plot(min_values, purity_min, '--g')
	ax2.grid()
	ax2.set_ylabel("purity %", size='medium')
	
	ax3.plot(min_values, nb_sou_min, '--k')
	ax3.grid()
	ax3.set_ylabel("# sources", size='medium')

	ax4.plot(min_values, compu_time_min, '--b')
	ax4.grid()
	ax4.set_xlabel("min_value", size='x-large')
	ax4.set_ylabel("computation time\n[s]", size='medium')
	
	plt.savefig('./fig/purity_minvar.pdf')
	plt.show()
		
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8,6), dpi=300)
	fig.subplots_adjust(hspace=0)
	fig.suptitle('Variation of min_delta factor', fontsize='x-large')
	
	ax1.plot(min_deltas, recall_delta, '--r')
	ax1.grid()
	ax1.set_ylabel("recall %", size='medium')
	
	ax2.plot(min_deltas, purity_delta, '--g')
	ax2.grid()
	ax2.set_ylabel("purity %", size='medium')
	
	ax3.plot(min_deltas, nb_sou_delta, '--k')
	ax3.grid()
	ax3.set_ylabel("# sources", size='medium')

	ax4.plot(min_deltas, compu_time_delta, '--b')
	ax4.grid()
	ax4.set_xlabel("min_delta", size='x-large')
	ax4.set_ylabel("computation time\n[s]", size='medium')
	
	plt.savefig('./fig/purity_sigmavar.pdf')
	plt.show()
	
def purity_test2(fits_file, min_values=10, delta_values=10):
	"""
	Performs a purity test on a FITS file by varying 
	min_value and min_delta factors simultaneously.
	"""
	
	if type(min_values)==int:
		min_space = np.arange(min_values)
	elif type(min_values)==type([]):
		min_space = min_values
	elif type(min_values)==type(np.array([])):
		min_space = min_values
		
	if type(delta_values)==int:
		delta_space = np.arange(delta_values)
	elif type(delta_values)==type([]):
		delta_space = delta_values
	elif type(delta_values)==type(np.array([])):
		delta_space = delta_values
	
	hdul = fits.open(fits_file)
	image = hdul[0].data
	hdr = hdul[0].header
	hdul.close()
	wcs = WCS.WCS(hdr)
	
	center = (hdr["CRVAL1"], hdr["CRVAL2"])

	LoTSS = get_LoTSS(center)
	RA_LoTSS = LoTSS["RAJ2000"]
	DEC_LoTSS = LoTSS["DEJ2000"]
	
	recall = np.zeros((len(delta_space),len(min_space)))
	purity = np.zeros((len(delta_space),len(min_space)))
	
	for i in range(len(delta_space)):
		for j in range(len(min_space)):
		
			delta_factor = delta_space[i]
			min_factor = min_space[j]
		
			name_file = "./dendrocat/old_test/"+fits_file.split("/")[-1][:-5]+"_test_%d_delta_%d_min.txt"%(delta_factor, min_factor)
		
			if(not(os.path.isfile(name_file))):

				deltamin_test(fits_file, delta_factor, min_factor)

				clean_cat(name_file)
				
				dendrocat = np.loadtxt(name_file, skiprows=1, usecols=[0,1])

			else:
				dendrocat = np.loadtxt(name_file, skiprows=1, usecols=[0,1])

			n = dendrocat.shape[0]
			
				
			Xmatch, _ = match_coord(dendrocat, np.array([RA_LoTSS, DEC_LoTSS]).T, 12)
		
			nb_Xmatched = Xmatch.shape[0]
				
			purity[i,j] = 100*nb_Xmatched/n
			recall[i,j] = 100*nb_Xmatched/len(RA_LoTSS)
				
	np.savez("purity_test_res.npz", Purity=purity, Recall=recall, minval=min_space, mindelta=delta_space)
	
"""	
if __name__ == "__main__":


	hdul = fits.open("../P7Hetdex11.fits")
	image = hdul[0].data
	hdr = hdul[0].header
	hdul.close()
	wcs = WCS.WCS(hdr)
	
	center = (hdr["CRVAL1"], hdr["CRVAL2"])
	
	cat_LoTSS = get_LoTSS(center)
	
	leaves = np.loadtxt("./dendrocat/P7Hetdex11_DendCat_leaves.txt", skiprows=1)
	trunk = np.loadtxt("./dendrocat/P7Hetdex11_DendCat_trunk.txt", skiprows=1)
	
	n_leaves = leaves.shape[0]
	n_trunk = leaves.shape[0]
	
	n_LoTSS  = len(cat_LoTSS["RAJ2000"].value)
	
	LoTSS = np.array([cat_LoTSS["RAJ2000"].value, cat_LoTSS["DEJ2000"].value, cat_LoTSS["SpeakTot"].value, cat_LoTSS["Maj"].value, cat_LoTSS["Min"].value]).T
	
	leaves_xm, leaves_ctp = match_coord(leaves, LoTSS, 12)
	trunk_xm, trunk_ctp = match_coord(trunk, LoTSS, 12)
	
	print("purity :")
	print("leaves :",100*len(leaves_xm)/n_leaves)
	print("trunk :",100*len(trunk_xm)/n_trunk)
	
	print("\nrecall :")
	print("leaves :",100*len(leaves_xm)/n_LoTSS)
	print("trunk :",100*len(trunk_xm)/n_LoTSS)
	
	print("\nflux rel err :")
	print("leaves :",np.mean(100*abs(leaves_xm[:,2]-leaves_ctp[:,2])/leaves_ctp[:,2]), "max :",np.max(100*abs(leaves_xm[:,2]-leaves_ctp[:,2])/leaves_ctp[:,2]), "min :", np.min(100*abs(leaves_xm[:,2]-leaves_ctp[:,2])/leaves_ctp[:,2]))
	print("trunk :",np.mean(100*abs(trunk_xm[:,2]-trunk_ctp[:,2])/trunk_ctp[:,2]), "max :", np.max(100*abs(trunk_xm[:,2]-trunk_ctp[:,2])/trunk_ctp[:,2]), "min :",np.min(100*abs(trunk_xm[:,2]-trunk_ctp[:,2])/trunk_ctp[:,2]))

	
	print("\nMaj rel err :")
	print("leaves :",np.mean(100*abs(leaves_xm[:,3]-leaves_ctp[:,3])/leaves_ctp[:,3]), "max :",np.max(100*abs(leaves_xm[:,3]-leaves_ctp[:,3])/leaves_ctp[:,3]), "min :", np.min(100*abs(leaves_xm[:,3]-leaves_ctp[:,3])/leaves_ctp[:,3]))
	print("trunk :",np.mean(100*abs(trunk_xm[:,3]-trunk_ctp[:,3])/trunk_ctp[:,3]), "max :", np.max(100*abs(trunk_xm[:,3]-trunk_ctp[:,3])/trunk_ctp[:,3]), "min :",np.min(100*abs(trunk_xm[:,3]-trunk_ctp[:,3])/trunk_ctp[:,3]))
	
	print("\nMin rel err :")
	print("leaves :",np.mean(100*abs(leaves_xm[:,4]-leaves_ctp[:,4])/leaves_ctp[:,4]), "max :",np.max(100*abs(leaves_xm[:,4]-leaves_ctp[:,4])/leaves_ctp[:,4]), "min :", np.min(100*abs(leaves_xm[:,4]-leaves_ctp[:,4])/leaves_ctp[:,4]))
	print("trunk :",np.mean(100*abs(trunk_xm[:,4]-trunk_ctp[:,4])/trunk_ctp[:,4]), "max :", np.max(100*abs(trunk_xm[:,4]-trunk_ctp[:,4])/trunk_ctp[:,4]), "min :",np.min(100*abs(trunk_xm[:,4]-trunk_ctp[:,4])/trunk_ctp[:,4]))"""
