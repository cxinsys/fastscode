<p align="center"><img src="assets/logo.png" alt="Drawing" width="600px"/></p>


## Indroduction
- FastSCODE: an accelerated implementation of SCODE based on manycore computing 

## Installation
- :snake: [Anaconda](https://www.anaconda.com) is recommended to use and develop FastSCODE.
- :penguin: Linux distros are tested and recommended to use and develop FastSCODE.

### Anaconda virtual environment

After installing anaconda, create a conda virtual environment for FastSCODE.
In the following command, you can change the Python version
(e.g. `python=3.12`).

```
conda create -n fastscode python=3.12
```

Now, we can activate our virtual environment for FastSCODE as follows.

```
conda activate fastscode
```
<br>

<!--
### Install from PyPi

```
pip install fastscode
```
- **Default backend framework of the FastSCODE is PyTorch.**- **You need to install other backend frameworks such as CuPy, Jax, and TensorFlow**

<br>
-->

### Install from GitHub repository


[//]: # (**You must install [MATE]&#40;https://github.com/cxinsys/mate&#41; before installing FastSCODE**)

First, clone the recent version of this repository.

```
git clone https://github.com/cxinsys/fastscode.git
```


Now, we need to install FastSCODE as a module.

```
cd fastscode
pip install -e .
```
