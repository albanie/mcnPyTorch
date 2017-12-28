PyTorch model converter
---

A tool for converting [PyTorch](https://github.com/pytorch/pytorch) models into 
[MatConvNet](https://github.com/vlfeat/matconvnet). 

### Imported pretrained models

Some of the useful pretrained models available in the `torchvision.models` module 
have been converted into MatConvNet and are available for download at the link
below: 

[Torchvision models](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#pytorch-models)

The *ResNeXt* family of models have also been imported and are available for download:

[ResNeXt models](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#resnext-models)

### Converting your own models

The conversion script requires Python (with PyTorch installed) and MATLAB. 
Converting models between frameworks tends to be a non-trivial task, so it is 
likely that modifications will be needed for unusual models.  To get started, 
see the `importer.sh` script (this can be modified to import new models).

### Installation

The easiest way to use this module is to install it with the `vl_contrib` 
package manager. `mcnPyTorch` can be installed with the following three commands from 
the root directory of your MatConvNet installation:

```
vl_contrib('install', 'mcnPyTorch') ;
vl_contrib('setup', 'mcnPyTorch') ;
```

**Dependencies**: 

* `Python3` 
* `PyTorch`
* standard numerical Python modules (which should be easy to install with `conda`) 
* The [Cadene repo](https://github.com/Cadene/pretrained-models.pytorch/tree/master/pretrainedmodels) of pre-trained pytorch models (adds support for additional networks which are not included in the main torchvision module)

To run the imported networks, the following matconvnet modules are also required:

* [autonn](https://github.com/vlfeat/autonn) - automatic differenation
* [mcnExtraLayers](https://github.com/albanie/mcnExtraLayers) - extra MatConvNet layers

Both of these can be setup directly with `vl_contrib` (i.e. run `vl_contrib install <module-name>` then `vl_contrib setup <module-name>`).



### Notes

* The normalisation used by the pretrained PyTorch models differs significantly from the typical matconvnet approach (see [here](https://github.com/albanie/mcnPyTorch/blob/master/benchmarks/cnn_imagenet_pt_mcn.m#L95) for an example).
* The weights in the converted model are modified slightly from the originals to compensate for differences in certain computational blocks.  For instance, PyTorch adds an `espilon` term to batch norm denominator during both training and inference.
