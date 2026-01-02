<p align="center"><img src="assets/logo.png" alt="Drawing" width="600px"/></p>


## Introduction
- FastSCODE is an accelerated implementation of [SCODE](https://github.com/hmatsu1226/SCODE) based on manycore computing.

## Installation
- :snake: [Anaconda](https://www.anaconda.com) is recommended to use and develop FastSCODE.
- :penguin: Linux distros are tested and recommended to use and develop FastSCODE.

### Create a virtual environment

After installing Anaconda, create a conda virtual environment for FastSCODE.
We can also specify the Python version (e.g. `python=3.12`).

```bash
conda create -n fastscode python=3.12
```

Now, we can activate our conda virtual environment for FastSCODE as follows.

```bash
conda activate fastscode
```
<br>


### Install from PyPi

```bash
pip install fastscode
```
- 🔥**The default backend framework of FastSCODE is [PyTorch](https://pytorch.org/).**
- 📱**You can install another backend framework such as [CuPy](https://cupy.dev/), [Jax](https://github.com/jax-ml/jax), and [TensorFlow](https://www.tensorflow.org/)**

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

### Create `FastSCODE` instance

The FastSCODE class requires input files such as expression data arrays and pseudo time arrays, 
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
- **use_binary**: save result matrix as binary file, optional, default: True

```python
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
                      dtype=np.float32,
                      use_binary=True)
```

<br>
<br>

### Run FastSCODE

#### parameters
- **backend**: optional, default: 'cpu'
- **device_ids**: list or number of devcies to use, optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **batch_size_b**: batch size of optimization parameter B, optional, default: 1
- **batch_size**: gene batch size of expression data, optional, default: None (compute all gene data at once, recommended)
- **chunk_size**: gene chunk size of expression data in inner loop of algorithm, optional, default: None (auto calculated)

```python
rss, score_matrix = worker.run(backend='gpu',
                               device_ids=8,
                               sampling_batch=100,
                               batch_size=1024)
```

<br>
<br>

### Run FastSCODE with `run_scode.py`

- Before run `run_scode.py`, `batch_size_b` and `batch_size` must be adjusted to fit your gpu memory size.

#### Usage
```bash
python run_scode.py --droot [root directory]
                    --fp_exp [expression file path]
                    --fp_trj [trajectory (pseudotime) file path] 
                    --fp_branch [cell select file path] 
                    --num_z [number of vector z]
                    --max_iter [number of optimization step]
                    --backend [name of backend framework]
                    --num_devices [number of devices]
                    --batch_size_b [number of parameter b]
                    --sp_droot [droot directory for saving results]
                    --num_repeat [total number of computation iterations]
```

#### Example
```bash
python run_scode.py --droot .
                    --fp_exp expression_dataTuck_sub.csv
                    --fp_trj pseudotimeTuck.txt
                    --fp_branch cell_selectTuck.txt
                    --num_z 10
                    --max_iter 100
                    --backend gpu
                    --num_devices 8
                    --batch_size_b 10
                    --sp_droot out
                    --num_repeat 6
```

#### Output
The average matrix for the repeatedly computed score matrix is saved as a binary file in `--sp_droot`. 

```
avg_score_matrix.npy
ex)
0	0.05	0.02	...	0.004
0.01	0	0.04	...	0.12
0.003	0.003	0	...	0.001
0.34	0.012	0.032	...	0


node_name.txt
ex)
GENE_1
GENE_2
GENE_3
.
.
.
GENE_M
```

The result file for each iteration is saved in the `[Number of repetation]` folder under `--sp_droot`. 

When `use_binary` is True, we can obtain the following result.

```
RSS.txt
ex)
3367844277.01837


score_matrix.npy
ex)
0	0.05	0.02	...	0.004
0.01	0	0.04	...	0.12
0.003	0.003	0	...	0.001
0.34	0.012	0.032	...	0


node_name.txt
ex)
GENE_1
GENE_2
GENE_3
.
.
.
GENE_M
```

When `use_binary` is False, we can obtain the following result.

```
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


## A tutorial for downstream analysis

### Create `NetWeaver` instance

`NetWeaver` infers the network links based on the results of running FastSCODE.

#### parameters

- **result_matrix**: result score matrix of fastscode, required
- **gene_names**: gene names from result matrix, required
- **tfs**: tf list, optional
- **fdr**: specifying fdr, optional, default: 0.01
- **links**: specifying number of outdegrees, optional, default: 0
- **is_trimming**: if set True, trimming operation is applied on grn, optional, default: True
- **trim_threshold**: trimming threshold, optional, default: 0

```python
result_matrix = np.loadtxt(fpath_result_matrix, delimiter='\t', dtype=str)
gene_name = result_matrix[0][1:]
result_matrix = result_matrix[1:, 1:].astype(np.float32)

tf = np.loadtxt(fpath_tf, dtype=str)

weaver = fs.NetWeaver(result_matrix=result_matrix,
                      gene_names=gene_name,
                      tfs=tf,
                      fdr=fdr,
                      links=links,
                      is_trimming=True,
                      trim_threshold=trim_threshold,
                      dtype=np.float32)
```

### Run NetWeaver
- **backend**: optional, default: 'cpu'
- **device_ids**: list or number of devices to use, optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **batch_size**: if set to 0, batch size will automatically calculated, optional, default: 0

```python
grn, trimmed_grn = weaver.run(backend=backend,
                              device_ids=device_ids,
                              batch_size=batch_size)
```

### Count outdegree
- **grn**: required

```python
outdegrees = weaver.count_outdegree(grn)
trimmed_ods = weaver.count_outdegree(trimmed_grn)
```

<br>
<br>

### Network reconstruction

`reconstruct_grn.py` shows an example of reconstructing network structures from the output of `grn` and `outdegree` files. <br>
When using a binary file, we must pass the path to the `node_name.txt` file to the `--fp_gn parameter`. <br>
If it is not a binary file, the `--fp_gn` parameter is optional.


#### Usage
We can specify fdr as follows.

```bash
python reconstruct_grn.py --fp_rm [result matrix path]  --fp_gn [gene name file path] --fp_tf [tf file path] --fdr [fdr] --backend [backend] --device_ids [number of device]
```

#### Example
```bash
python reconstruct_grn.py --fp_rm avg_score_matrix.txt --fp_gn node_name.txt --fp_tf mouse_tf.txt --fdr 0.01 --backend gpu --device_ids 1
```

#### Output
```csv
avg_score_matrix.fdr0.01.sif, avg_score_matrix.fdr0.01.sif.outdegrees.txt
avg_score_matrix.fdr0.01.trimIndirect0.sif, avg_score_matrix.fdr0.01.trimIndirect0.sif.outdegrees.txt
```

<br>

#### Usage
We can also specify the links.

```bash
python reconstruct_grn.py --fp_rm [result matrix path] --fp_gn [gene name file path]  --fp_tf [tf file path] --links [links] --backend [backend] --device_ids [number of device]
```

#### Example
```bash
python reconstruct_grn.py --fp_rm avg_score_matrix.txt --fp_gn node_name.txt --fp_tf mouse_tf.txt --links 1000 --backend gpu --device_ids 1
```

#### Output
```csv
avg_score_matrix.links1000.sif, avg_score_matrix.links1000.sif.outdegrees.txt
avg_score_matrix.links1000.trimIndirect0.sif, avg_score_matrix.links1000.trimIndirect0.sif.outdegrees.txt
```



## TODO

- [x] Upload to PyPi
