# Topk documentation

## Description
Topk is a cosmological likelihood code designed to be interfaced with the framework Cobaya. It is created to perform MCMC 
analysis of 21cm and 21cm-galaxy cross correlation power spectra. It offers a wide variety of options which thoroughly
described in the official documentation which can be found in the topk github repository in PDF form.

> [!NOTE]
> If you use topk, please cite the following papers:
> 1. M. Berti, M. Spinelli, B.S. Haridasu, M. Viel, A. Silvestri, *Constraining beyond* $\Lambda$*CDM models with 21cm 
> intensity mapping forecast observations combined with latest CMB data.* [arXiv:2109.03256](https://arxiv.org/abs/2109.03256)
> 2. M. Berti, M. Spinelli, M. Viel, *Multipole expansion for 21cm Intensity Mapping power spectrum: forecasted
> cosmological parameters estimation for the SKA Observatory.* [arXiv:2209.07595](https://arxiv.org/abs/2209.07595)
> 3. M. Berti, M. Spinelli, M. Viel, *21 cm intensity mapping cross-correlation with galaxy surveys: Current and
> forecasted cosmological parameters estimation for the SKAO.* [arXiv:2309.00710](https://arxiv.org/abs/2309.00710)

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
