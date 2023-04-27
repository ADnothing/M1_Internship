# M1 Internship
Intership performed at the LERMA and supervised by David Cornu and Philippe Salomé.

This repository contains the code I used during my internship.

# Introduction

...

# LoTSS DR2

You can dowload the catalog in : https://cdsarc.cds.unistra.fr/ftp/J/A+A/659/A1/

catalog.dat 
dim : 26
size : 4396228


All information about the LoTSS DR2 can be found in : https://lofar-surveys.org/dr2_release.html
This is the documentation fo the data found in https://vo.astron.nl/lotss_dr2/q/src_cone/info :

Byte-by-byte Description of file: catalog.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 22  A22   ---       Source    The radio name of the source, automatically
                                     generated from RA and DEC (Source_Name)
  24- 32  F9.5  deg       RAdeg     Right ascension (J2000) (RA) (1)
  34- 39  F6.2  arcsec  e_RAdeg     rms uncertainty on RA (E_RA) (2)
  41- 49  F9.5  deg       DEdeg     Declination (J2000) (DEC) (1)
  51- 56  F6.2  arcsec  e_DEdeg     rms uncertainty on DE (E_DEC) (2)
  58- 66  F9.3 mJy/beam   Speak     The peak Stokes I flux density per beam of
                                     the source (Peak_flux)
  68- 73  F6.3 mJy/beam e_Speak     The 1-sigma error on the peak flux density
                                     per beam of the source (EPeakflux)
  75- 83  F9.3  mJy       SpeakTot  The total, integrated Stokes I flux density
                                     of the source at the reference frequency
                                     (Total_flux)
  85- 91  F7.3  mJy     e_SpeakTot  The 1-sigma error on the total flux density
                                     of the source (ETotalflux)
  93- 98  F6.2  arcsec    Maj       FWHM of the major axis of the source,
                                     INCLUDING convolution with the 6-arcsec
                                     LOFAR beam (Maj)
 100-105  F6.2  arcsec  e_Maj       The 1-sigma error on the FWHM of the major
                                     axis of the source (E_Maj)
 107-112  F6.2  arcsec    Min       FWHM of the minor axis of the source,
                                     INCLUDING convolution with the 6-arcsec
                                     LOFAR beam (Min)
 114-119  F6.2  arcsec  e_Min       The 1-sigma error on the FWHM of the minor
                                     axis of the source (E_Min)
 121-126  F6.2  arcsec    DCMaj     The FWHM of the major axis of the source,
                                     after de-convolution with the 6-arcsec
                                     LOFAR beam (DC_Maj)
 128-133  F6.2  arcsec  e_DCMaj     The 1-sigma error on the FWHM of the
                                     deconvolved major axis of the source
                                     (EDCMaj)
 135-140  F6.2  arcsec    DCMin     The FWHM of the minor axis of the source,
                                     after de-convolution with the 6-arcsec
                                     LOFAR beam (DC_Min)
 142-147  F6.2  arcsec  e_DCMin     The 1-sigma error on the FWHM of the
                                     deconvolved minor axis of the source
                                     (EDCMin)
 149-154  F6.2  deg       PA        The position angle of the major axis of the
                                     source measured east of north, after
                                     de-convolution with the 6-arcsec LOFAR
                                     beam (PA)
 156-161  F6.2  deg     e_PA        The 1-sigma error on the position angle of
                                     the deconvolved major axis of the source
                                     (E_PA)
 163-168  F6.2  deg       DCPA      The position angle of the major axis of the
                                     source measured east of north, after
                                     de-convolution with the 6-arcsec LOFAR
                                     beam (DC_PA)
 170-175  F6.2  deg     e_DCPA      The 1-sigma error on the position angle of
                                     the deconvolved major axis of the source
                                     (EDCPA)
 177-182  F6.3 mJy/beam   Islrms    The average background rms value of the
                                     island  (Isl_rms)
     184  A1    ---       SCode     A code that defines the source structure in
                                     terms of the fitted Gaussian components
                                     (S_Code) (3)
 186-196  A11   ---       Mosaic    The name mosaic in which the FITS image can
                                     be found (Mosaic_ID)
     198  I1    ---       Npoint    The number of pointings that are mosaiced at
                                     the position of the source
                                     (Number_Pointings)
 200-204  F5.3  ---       MaskFract The fraction of the source that is in the
                                     CLEAN mask (Masked_Fraction)
--------------------------------------------------------------------------------
Note (1): Positions from PyBDSF or combination of PyBDSF components.
Note (2): Posiiton errors estimated by PyBDSF. Only fitting uncertainties are
  included.
Note (3): Code as follows:
  S = indicates an isolated source that is fit with a single Gaussian
  C = represents sources that are fit by a single Gaussian but are within an
       island of emission that also contains other sources
  M = is used for sources with are extended and fit with multiple Gaussians
  

I converted the catalog.dat to a catalog.csv, then I save the data in npz files
that I can load separately the files and avoid memory issue.

names.npz : Source_Name
positions.npz : RA, E_RA, DEC, E_DEC
flux.npz : Peak_flux, EPeakflux, Total_flux, ETotalflux
dim.npz : Maj, E_Maj, Min, E_Min, DC_Maj, EDCMaj, DC_Min, EDCMin 
PA.npz : PA, E_PA, DC_PA, EDCPA
mosaic.npz : Isl_rms, S_Code, Mosaic_ID, Number_Pointings, Masked_Fraction

No need to dowload the catalog, you can find directly the npz files in this repository.
