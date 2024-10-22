# topk - a likelihood code for the 21cm power spectrum
Authors: Gabriele Autieri, Maria Berti, Marta Spinelli, and Matteo Viel

## Description
topk is a cosmology code designed to compute the theoretical predictions and the likelihood for several 21cm intensity mapping observables. It can work with multiple redshift-bin observations of the 21cm auto-power spectrum multipoles (monopole, quadrupole, and hexadecapole) and of cross-correlation power spectra with galaxy clustering. topk offers a wide variety of options, thoroughly described in the documentation that can be found in the repository.

> [!NOTE]
> topk is based on the following publications: [arXiv:2109.03256](https://arxiv.org/abs/2109.03256), [arXiv:2209.07595](https://arxiv.org/abs/2209.07595), [arXiv:2309.00710](https://arxiv.org/abs/2309.00710).
> 
> If you use topk, please cite at least [arXiv:2209.07595](https://arxiv.org/abs/2209.07595).

## Pre-requisites 
topk is designed to be interfaced with the code [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html). The only pre-requisites are Python (version ≥ 3.9) and basic Python packages (numpy, scipy, os).

## How to run 
In general, to use topk with Cobaya only the `topk.py` module is needed. The full path to `topk.py` must be specified in the likelihood block of the input yaml file for a Cobaya run.

In the folder `examples`, we provide mock data sets and yaml files to reproduce the results from the papers and examples discussed in the documentation.
