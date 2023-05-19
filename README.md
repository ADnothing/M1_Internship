# M1 Internship 2023
Intership performed at the LERMA and supervised by David Cornu and Philippe Salomé.

This repository contains the code I used during my internship.

## Introduction

...

## LoTSS DR2


Parameters of the catalog are explored in the notebook "Catalog_analysis.ipynb".

You can find the catalog in : https://cdsarc.cds.unistra.fr/ftp/J/A+A/659/A1/

All the explanations are in the file : Info

The catalog is too heavy for GitHub, if you want to run the Notebook "Catalog_analysis.ipynb" You should download the catalog on your own and maybe modify the way the data are stored and called.

## Files and content

| File name            | Content                                                 |
| -------------------- | ------------------------------------------------------- |
| MosaicInsigh.py      | show_fits ; show_image ; calc_axis ; save_subfits ; dump_fits         |
| corr_fits.py         | fill_im_hole ; corr_fits ; input_gen ; input_gen_MD                   |
| Patch_Management.py (1)| fct_IoU ; fct_classical_IoU ; asStride ; poolingOverlap ; NMS_1st|
| cross_test.py | gen_pos ; gen_test |
| make_cat.py | patch_gen ; get_params ; Crea_dendrogram |

Note :

(1) : The functions present in this file are (mostlty) written by David Cornu.
