# Topk documentation

## Description
Topk is a cosmological likelihood code designed to be interfaced with the framework Cobaya. It is created to perform MCMC 
analysis of 21cm and 21cm-galaxy cross correlation power spectra. It offers a wide variety of options which thoroughly
described in the official documentation which can be found in the topk github repository in PDF form.

> [!NOTE]
> If you use topk, please cite:
> - M. Berti, M. Spinelli, M. Viel, *Multipole expansion for 21cm Intensity Mapping power spectrum: forecasted
> cosmological parameters estimation for the SKA Observatory,* Monthly Notices of the Royal Astronomical Society, [521 (2023) 3](https://academic.oup.com/mnras/article/521/3/3221/7070719), [arXiv:2209.07595](https://arxiv.org/abs/2209.07595)

## Prerequisites 
Topk is written in python 3.9 and requires an installation of Cobaya. Further informations on prerequisites for Cobaya and more
info on Cobaya's installation can be found in [Cobaya's documentation](https://cobaya.readthedocs.io/en/latest/index.html). 
The following python modules are required:
* numpy
* scipy
* os

## Installation
To use topk, download the latest release, all necessary files are already in place and ready to be used. Topk files don't need to 
be located inside Cobaya's folder.

## How to run 
Topk is designed to be run with Cobaya. When running Cobaya, insert topk's full path in the likelihood block of the input .yaml file.
Here, all the options can be specified.

## Examples
Examples of MCMC runs using topk can be found in the example section of the documentation.
