# DID: "Measuring dissimilarity with diffeomorphism invariance"

Author: [Théophile Cantelobre](https://theophilec.github.io)
## What is DID? 

In a nutshell, DID is a generic dissimilarity that can be used to compare 
two points (images, time series, ...) in such a way that is invariant to 
diffeomorphisms between the points. For example: `DID(x, rotate(x))` gives a small
value.

This repository provides an implementation of DID for images. We hope providing our
code will help spur enthusiasm around the method, and allow researchers to 
reproduce the experiments in our paper.


For more complete motivation (and all the details), see the [paper](https://arxiv.org/pdf/2202.05614.pdf):

> _Measuring dissimilarity with diffeomorphism invariance_, Théophile Cantelobre, Carlo Ciliberto,
> Benjamin Guedj, Alessandro Rudi (2022). ICML.

**Wait... why is the library called `did` and not `diffy` like the repo?** We have
not decided on a name yet... but you can open an issue if you have an opinion.

### Documentation
This README presents installation and getting started information.
See the [documentation](https://diffy-ml.readthedocs.io/en/latest/) for more details.

To make the documentation locally, run `make html` in the `docs` directory.

## Getting started with DID 

```bash
$ git clone
```
### Install dependencies
With conda:
```bash
$ conda create -n did_env -f environment.yml
$ conda activate did_env
```

With pip:
```bash
$ python3 -m venv did_env
$ source did_env/bin/activate
(did_env) $ pip install -r requirements.txt
(did_env) $ pip install -r docs_requirements.txt  # optional
```
### Using DID (without installing)

```bash
(did_env) $ python appendix_peppers_match.py
```

### Installing DID (optional)

```bash
(did_env) $ python setup.py install
```

### Hardware acceleration (optional)
DID can be hardware accelerated thanks to `torch` and CUDA. If you do not have a GPU,
set `device` to `cpu`.

## Playing with DID

The best place to start is in the `demo_... .py` files in this repository. They contain
simple examples you can play around with... think of them as minimal working examples. In
fact, they reproduce Figure 1 in the paper.

Once you have an understanding of the way DID works, you can experiment on your own, or use
the `exp_... .py` files. These are for experiments that concern the other figures in the paper.

### Feeding DID (data)

The `demo_... .py` scripts depend on data in the `perspective` and `scenes` directories. Other 
provided scripts require Imagenette (ImageNet will work too). You can set a path to 
Imagenette/Imagenet in `imagenet.py`.

## Citing this work

If you use DID, we'd be happy to read about it if you cite the pre-print:
```
 @article{cantelobre2022measuring,
        title={Measuring dissimilarity with diffeomorphism invariance}, 
        author={Théophile Cantelobre and Carlo Ciliberto and Benjamin Guedj and Alessandro Rudi},
        year={2022}, 
        journal={arXiv 2202.05614}, 
        volume={stat.ML},
        url={https://arxiv.org/abs/2202.05614}
 }
```
