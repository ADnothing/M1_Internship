#!/usr/bin/env python

#Adrien Anthore, 06 juin 2023
#main.py
#main python file


import numpy as np

from multiprocessing import Pool
import os
from tqdm import tqdm

from astropy.io import fits

from make_cat import crea_dendrogram
from corr_cat import clean_cat
from Xmatch import test_cat

if not os.path.exists("./dendrocat"):
	os.makedirs("./dendrocat")
	
if not os.path.exists("done.txt"):
	with open("done.txt", "w"):
		pass

with open("res.txt", 'w') as file:
	file.write("cat\trecall\tprec\tfluxreld\tmajreld\þminreld\n")	


path = "/minerva/stageM2/SDC1/LOTSS/fits"

def run(fits_file):

        flag=1
        cat_name = "./dendrocat/"+fits_file[:-5]+"_DendCat.txt"
        fits_file = path+"/"+fits_file

        with open("done.txt", "r") as done_file:
                if fits_file in done_file.read():
                        flag=0

        if fits_file[-5:] != ".fits":
                flag=0

        if flag:
                crea_dendrogram(fits_file, promt=False)
                clean_cat(cat_name)

                with open("done.txt", "a") as done_file:
                        done_file.write(fits_file + "\n")



if __name__ == "__main__":

	list_fits = os.listdir(path)
	
	with Pool(processes=8) as pool:
		pool.map(run, list_fits)
		
		
	list_cat = os.listdir("./dendrocat/")

        recall_arr = np.zeros(len(list_cat))
        prec_arr = np.zeros(len(list_cat))
        flux_reld = np.zeros(len(list_cat))
        maj_reld = np.zeros(len(list_cat))
        min_reld = np.zeros(len(list_cat))

        for i in tqdm(range(len(list_cat))):

                cat = list_cat[i]
                fits_file = path +"/"+ cat[:-12]+".fits"
                catalog = np.loadtxt("./dendrocat/"+cat, skiprows=1)
                if catalog.shape == (0,):
                        maj_reld[i] = 1
                        min_reld[i] = 1
                        flux_reld[i] = 1
                        continue
                if catalog.shape == (7,):
                        catalog = catalog.reshape(1,7)

                hdul = fits.open(fits_file)
                hdr = hdul[0].header
                hdul.close()

                recall, prec, fluxreld, majreld, minreld = test_cat(catalog, hdr)
                recall_arr[i] = recall
                prec_arr[i] = prec
                flux_reld[i] = fluxreld
                maj_reld[i] = majreld
                min_reld[i] = minreld

                file = open("res.txt", "a")

                line = '%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f'%(cat, recall, prec, fluxreld, majreld, minreld)
                file.write(line + '\n')

                file.close()




        print(recall_arr.mean(), prec_arr.mean(), flux_reld.mean(), maj_reld.mean(), min_reld.mean())

