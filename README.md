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


### Install from PyPi

```
pip install fastscode
```
- **Default backend framework of the FastSCODE is PyTorch.**- **You need to install other backend frameworks such as CuPy, Jax, and TensorFlow**

<br>

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


## FastSCODE tutorial

### Create FastSCODE instance

The FastSCODE class requires loaded files such as expression data arrays and pseudo time arrays, 
as well as several parameters for linear ODE optimization.

#### parameters
- **exp_data**: expression data array (Gene (G) x Cell (C)), required
- **pseudotime**: pseudotime data vector (C), required
- **node_name**: vector for name of genes (G), required
- **droot**: root directory for storing score matrix and RSS arrays, optional, default value is None, which means that the results are not saved 
- **num_tf**: number of genes to use, optional, default value is None, and all genes are used
- **num_cell**: number of cells to use, optional, default value is None, and all cells are used
- **num_z**: length of vector z for optimization, optional, default: 4
- **max_iter**: number of iterations for optimization, optional, default: 100
- **max_b**: maximum initialization value for parameter b, optional, default: 2.0
- **min_b**: minimum initialization value for parameter b, optional, default: -10.0
- **dtype**: data type, optional, default: float32

```angular2html
import fastscode as fs

exp_data = np.loadtxt(dpath_exp_data, delimiter=",", dtype=str)
node_name = exp_data[0, 1:]
exp_data = exp_data[1:, 1:].astype(np.float64).T  # gene x cell

pseudotime = np.loadtxt(dpath_trj_data, delimiter="\t")

worker = fs.FastSCODE(exp_data=exp_data,
                      pseudotime=pseudotime,
                      node_name=node_name,
                      droot=spath_droot_r,
                      num_tf=None,
                      num_cell=None,
                      num_z=num_z,
                      max_iter=max_iter,
                      dtype=np.float32)
```

<br>
<br>

### Run FastSCODE

#### parameters
- **backend**: optional, default: 'cpu'
- **device_ids**: list or number of devcies to use, optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **batch_size_b**: batch size of optimization parameter B, optional, default: 1
- **batch_size**: gene batch size of expression data, optional, default: compute all gene data at once


```angular2html
rss, score_matrix = worker.run(backend='gpu',
                               device_ids=8,
                               sampling_batch=100,
                               batch_size=1024)
```

<br>
<br>

### Run FastSCODE with tutorial_notf.py

- **Before run run_scode.py, batch_size_b and batch_size parameter must be modified to fit your gpu memory size**

#### Usage
```angular2html
python run_scode.py --droot [root directory]
                    --fp_exp [expression file path]
                    --fp_trj [trajectory (pseudotime) file path] 
                    --fp_branch [cell select file path] 
                    --num_z [number of vector z]
                    --max_iter [number of optimization step]
                    --backend [name of backend framework]
                    --num_devices [number of devices]
                    --batch_size_b [number of parameter b]
                    --batch_size [number of batch size]
                    --sp_droot [droot directory for saving results]
                    --num_repeat [total number of computation iterations]
```

#### Example
```angular2html
python run_scode.py --droot .
                    --fp_exp expression_dataTuck_sub.csv
                    --fp_trj pseudotimeTuck.txt
                    --fp_branch cell_selectTuck.txt
                    --num_z 10
                    --max_iter 100
                    --backend gpu
                    --num_devices 8
                    --batch_size_b 10
                    --batch_size 4000
                    --sp_droot out
                    --num_repeat 6
```

#### Output
```angular2html
RSS.txt

ex)
3367844277.01837


score_matrix.txt                            

ex)
Score	GENE_1	GENE_2	GENE_3	...	GENE_M
GENE_1	0	0.05	0.02	...	0.004
GENE_2	0.01	0	0.04	...	0.12
GENE_3	0.003	0.003	0	...	0.001
.
.
.
GENE_M	0.34	0.012	0.032	...	0
```

<br>
<br>


## Downstream analysis tutorial

### Create NetWeaver instance



#### parameters

- **result_matrix**: result TE matrix of FastTENET, required
- **gene_names**: gene names from result matrix, required
- **tfs**: tf list, optional
- **fdr**: specifying fdr, optional, default: 0.01
- **links**: specifying number of outdegrees, optional, default: 0
- **is_trimming**: if set True, trimming operation is applied on grn, optional, default: True
- **trim_threshold**: trimming threshold, optional, default: 0

```angular2html
result_matrix = np.loadtxt(fpath_result_matrix, delimiter='\t', dtype=str)
gene_name = result_matrix[0][1:]
result_matrix = result_matrix[1:, 1:].astype(np.float32)

tf = np.loadtxt(fpath_tf, dtype=str)

weaver = fte.NetWeaver(result_matrix=result_matrix,
                       gene_names=gene_name,
                       tfs=tf,
                       fdr=fdr,
                       links=links,
                       is_trimming=True,
                       trim_threshold=trim_threshold,
                       dtype=np.float32
                       )
```

### Run weaver
- **backend**: optional, default: 'cpu'
- **device_ids**: list or number of devices to use, optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **batch_size**: if set to 0, batch size will automatically calculated, optional, default: 0

```angular2html
grn, trimmed_grn = weaver.run(backend=backend,
                              device_ids=device_ids,
                              batch_size=batch_size)
```

### Count outdegree
- **grn**: required

```angular2html
outdegrees = weaver.count_outdegree(grn)
trimmed_ods = weaver.count_outdegree(trimmed_grn)
```

<br>
<br>

### Downstream analysis with reconstruct_grn.py

reconstruct_grn.py is a tutorial script for the output of grn and outdegree files.

#### Usage
When specifying an fdr
```angular2html
python reconstruct_grn.py --fp_rm [result matrix path] --fp_tf [tf file path] --fdr [fdr] --backend [backend] --device_ids [number of device]
```

#### Example
```angular2html
python reconstruct_grn.py --fp_rm TE_result_matrix.txt --fp_tf mouse_tf.txt --fdr 0.01 --backend gpu --device_ids 1
```

#### Output
```angular2html
score_matrix.fdr0.01.sif, score_matrix.fdr0.01.sif.outdegrees.txt
score_matrix.fdr0.01.trimIndirect0.sif, score_matrix.fdr0.01.trimIndirect0.sif.outdegrees.txt
```

<br>

#### Usage
When specifying the links
```angular2html
python reconstruct_grn.py --fp_rm [result matrix path] --fp_tf [tf file path] --links [links] --backend [backend] --device_ids [number of device]
```

#### Example
```angular2html
python reconstruct_grn.py --fp_rm TE_result_matrix.txt--fp_tf mouse_tf.txt --links 1000 --backend gpu --device_ids 1
```

#### Output
```angular2html
score_matrix.links1000.sif, score_matrix.links1000.sif.outdegrees.txt
score_matrix.links1000.trimIndirect0.sif, score_matrix.links1000.trimIndirect0.sif.outdegrees.txt
```



## TODO

- [x] Upload to PyPi
