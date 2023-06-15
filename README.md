# M1 Internship 2023
Intership performed at the LERMA and supervised by David Cornu and Philippe Salomé.

This repository contains the code I used during my internship.

## Abstract

The objective of the MINERVA team (Machine Learning for Radioastronomy at the Observatoire de Paris) is to enhance their previous results of sources detection and characterization on
the LoTSS survey. This internship report focuses on constructing high-confidence catalogs for
the LoTSS fields in order to construct later a high-quality training dataset to better train the
team’s deep-learning method to take into account specific properties and artifacts of the LOFAR radio-telescope. The catalogs derived from the entire LoTSS field reached satisfactory
performances with an average recall of 59.9 ± 15.5% and an average precision of 61.9 ± 7.5%.

## Introduction

Radio-astronomy is experiencing a rebirth in its low-frequency domain, particularly with the
development of giant interferometers such as LOFAR, ALMA, NenuFAR, and the upcoming Square Kilometer Array (SKA). These instruments produce large and highly dimensional
datasets, presenting challenges for traditional methods of source detection and characterization.
In parallel, Machine Learning methods have undergone algorithmic developments that bring
them to a high level of maturity for these tasks.
The MINERVA project (Machine Learning for Radioastronomy at the Observatoire de Paris) is
at the forefront of applying Machine Learning to radio-astronomical datasets. The project led
a team that won the second edition of the SKA Science Data Challenges held in 2021(Hartley,
et al., 2023), and that was focused on source detection and characterization in a large 3D synthetic cube of HI emission. For this purpose, the team developed a specialized Deep Learning
detection method. Their approach not only demonstrated state-of-the-art performance for data
challenge 2 but also showed great performance in data challenge 1 (Bonaldi, et al., 2020), which
focused on source detection and characterization in synthetic continuum 2D images, as seen in
figure 1. Now the team looks forward to using this methodology for real observed datasets in
order to construct new source catalogs and explore new ways of combining information from
multiple surveys. In particular, the team is interested in the survey LoTSS (the LOFAR Two-
metre Sky Survey). The LoTSS DR2 survey consists of a high-resolution low-frequency survey
accompanied by a catalog of approximately 4 million radio sources that has been derived by
using classical detection tools (Shimwell, et al., 2022). This internship builds upon the work
done in a previous M2 internship where a network trained on the SDC1 dataset was applied to
the LoTSS survey, yielding satisfactory results.
The goal of the internship was to construct a complementary high-confidence training sample,
specifically for the LoTSS by refining and combining source catalogs obtained with other detection methods. My internship builds on the results of the previous internship by focusing on
the creation of a training set that will refine the network’s training specifically for LoTSS. For
that purpose, I had to train in machine learning. Then, I extensively explored the LoTSS DR2
to understand its features and potential challenges. This informed my approach to constructing
a highly reliable and comprehensive training set, which is crucial for effective machine-learning
training. The method used to build the training set is a classical method, different from the one
used for LoTSS DR2 in order to challenge their performance.
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
| corr_cat.py | Rexcl ; clean_cat |
| Xmatch.py | match_coord ; get_refcat ; get_LoTSS ; test_cat |

Note :

(1) : The functions present in this file are (mostlty) written by David Cornu.
