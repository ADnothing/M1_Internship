#!/usr/bin/env python

#Adrien Anthore, 29 mars 2023
#data_prep.py

import numpy as np
import pandas as pd

def data_prep():
	"""
	From the compressed csv file of the catalog, create npz files
	"""

	colname = ["Source_Name", "RA", "E_RA", "DEC", "E_DEC", "Peak_flux", "EPeakflux", "Total_flux", "ETotalflux", "Maj", "E_Maj", "Min", "E_Min", "DC_Maj", "EDCMaj", "DC_Min", "EDCMin", "PA", "E_PA", "DC_PA", "EDCPA", "Isl_rms", "S_Code", "Mosaic_ID", "Number_Pointings", "Masked_Fraction"]
	print("Reading the catalog...")
	df = pd.read_csv("catalog.csv.gz", compression='gzip', header=None, sep=',', names=colname)
	print("Insight :\n", df)
	print("Saving...")
	np.savez("npzdata/names.npz", Source_Name=np.char.strip(df["Source_Name"].values.astype(str)))
	np.savez("npzdata/positions.npz", RA=df["RA"].values, E_RA=df["E_RA"].values, DEC=df["DEC"].values, E_DEC=df["E_DEC"].values)
	np.savez("npzdata/flux.npz", Peak_flux=df["Peak_flux"].values, EPeakflux=df["EPeakflux"].values, Total_flux=df["Total_flux"].values, ETotalflux=df["ETotalflux"].values)
	np.savez("npzdata/dim.npz", Maj=df["Maj"].values, E_Maj=df["E_Maj"].values, Min=df["Min"].values, E_Min=df["E_Min"].values, DC_Maj=df["DC_Maj"].values, EDCMaj=df["EDCMaj"].values, DC_Min=df["DC_Min"].values, EDCMin=df["EDCMin"].values)
	np.savez("npzdata/PA.npz", PA=df["PA"].values, E_PA=df["E_PA"].values, DC_PA=df["DC_PA"].values, EDCPA=df["EDCPA"].values)
	np.savez("npzdata/mosaic.npz", Isl_rms=df["Isl_rms"].values, S_Code=np.char.strip(df["S_Code"].values.astype(str)), Mosaic_ID=df["Mosaic_ID"].values, Number_Pointings=df["Number_Pointings"].values, Masked_Fraction=df["Masked_Fraction"].values)
	print("Done.")
