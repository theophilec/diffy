Getting started
===============

`did` is a library which should be used for computing the DID dissimilarity between
images.

This tutorial explains the installation of `did` as well as a first example.

**Wait... why is the library called `did` and not `diffy` like the repo?** We have
not decided on a name yet... but you can open an issue if you have an opinion.




Installation
------------

Dependencies
____________

`did` has the following major dependencies. The versions given are those used to develop and test it, however they should be flexible. Please add an issue on Github if you notice dependency issues.

- `numpy` and `scipy` 
- `torch` and `torchvision`
- `pillow`
- `matplotlib` 

You can install these by hand with your choice of method (for example `pip` or `conda`), 
or use the `environment.yml` (`conda create -n did_env -f environment.yml`) and `requirements.txt`
(`pip install -r requirements.txt`) available in the repository.

.. code-block:: console
   
   (did_env) $ pip install -r requirements.txt  # method with pip
   (base) $ conda create -n did_env -f environment.yml  # method with conda



Locally installing the library (optional)
________________________________________

If you plan on using `did` in your own projects (and you should!), the easiest method is to locally
install the package with `python setup.py install` in your environment.

This is not necessary if you simply want to reproduce the experiments in the paper 
(for instance, with the scripts in the repository).

First examples
--------------


