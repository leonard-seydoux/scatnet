# Scatnet

Learnable scattering network in TensorFlow.

## Deprecation notice

We would like to notify you of a deprecation notice for the Scatnet repository written in 2020, which accompanied the paper _Clustering earthquake signals and background noises in continuous seismic data with unsupervised deep learning_ by Seydoux et al. ([2020](https://www.nature.com/articles/s41467-020-17841-x)). While the package was built on TensorFlow 1.13, it is now outdated and not compatible with most modern GPUs. To reproduce the results presented in the paper, we suggest installing a TensorFlow version that is equal to or older than 1.13.

Please note that some of the experimental packages used in the paper have since been removed in TensorFlow 2.x, and we cannot guarantee the same behavior if you attempt to convert the code to TensorFlow 2.x.

## New version

We are pleased to announce the development of a new version of a scattering transform for seismic time series analysis available at https://github.com/scatseisnet/scatseisnet. The new project features a non-learnable scattering transform and has been utilized in several more recent papers, all of which are documented in the associated documentation at https://scatseisnet.readthedocs.io/en/latest/. We welcome any and all feedback and contributions, and invite you to install the package and leave a comment on the GitHub repository.

Best regards,

Leonard Seydoux, on behalf of the co-authors.
