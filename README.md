# M1 Internship 2023
Intership performed at the LERMA and supervised by David Cornu and Philippe Salomé.

This repository contains the code I used during my internship.

## Introduction

Radio-astronomy is experiencing a rebirth in its low-frequency domain, particularly with the development of giant interferometers such as LOFAR, ALMA, NenuFAR, and the upcoming Square Kilometer Array (SKA).
These instruments produce large and highly dimensional datasets, presenting challenges for traditional methods of source detection and classification.
In parallel, Machine Learning methods have undergone algorithmic developments that bring them to a high level of maturity for these tasks.
The MINERVA project (Machine Learning for Radioastronomy at the Observatoire de Paris) is at the forefront of applying Machine Learning to radio-astronomical datasets.
The project has won 2 years ago the second SKA data challenge 2 (Hartley, et al., 2023) that was focused on source detection and characterization in a large 3D synthetic cube.
For this purpose, the team developed a specialized Deep Learning detection method.
This method not only demonstrated state-of-the-art performance for data challenge 2 but also showed great performance in data challenge 1 (Bonaldi, et al., 2020),
which focused on source detection and classification in synthetic 2D images.
Now the team looks forward to using this methodology for real observed datasets in order to construct new source catalogs and explore new ways of combining information from multiple surveys.


The goal of the internship was to construct a complementary training sample specifically for the LoTSS (the LOFAR Two-metre Sky Survey) survey by pruning source catalogs obtained with other detection methods.
The LoTSS survey consists of a high-resolution 120-168 MHz survey covering 27% northern sky.
From this data, a catalog of approximately 4 million radio sources has been derived by the tool Python Blob Detector and Source Finder (PyBDSF) (Shimwell, et al., 2022).
For that purpose, I followed David Cornu’s class on machine learning.
Then, I extensively explored the LoTSS DR2 (Shimwell, et al., 2022) to understand its features and potential challenges.
This informed my approach to constructing a highly reliable and comprehensive training set, which is crucial for effective machine learning applications.
This internship builds upon the work done in a previous M2 internship where a network trained on SKA data challenge 1 data was applied to the LoTSS dataset, yielding satisfactory results.
The objective of my internship is to further improve these results by the creation of this training set that will refine the network’s training specifically for LoTSS.

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
