# Generalizing Fair Top-k Selection: An Integrative Approach

## Overview
This repository hosts code for *Generalizing Fair Top-k Selection: An Integrative Approach* (paper comming soon).

The `main` branch is all you need for reproducing experimental results and the `preprocessing` branch contains code for data preprocessing.

## Build
### Containerization (recommended)
This project can be built using either Apptainer or Docker containerization. 

#### Apptainer
1. [Install Apptainer](https://apptainer.org/docs/admin/main/installation.html) (see [this](https://github.com/apptainer/apptainer/blob/main/INSTALL.md#apparmor-profile-ubuntu-2310) for installation on Ubuntu 23.10+)
2. Build Apptainer image
   ```
   apptainer build ./container/fair_topk_container.sif ./container/fair_topk_container.def
   ```
3. Launch the container
   ```
   apptainer run ./container/fair_topk_container.sif
   ```
4. Compilation
   ```
   cmake . && make -j
   ```
#### Docker
1. [Install Docker](https://docs.docker.com/engine/install)
2. Pull Docker image
   ```
   docker pull caiguangya/fair-topk:latest
   ```
3. Launch the container
   ```
   docker run -it -v $(pwd):/fair-topk -w /fair-topk caiguangya/fair-topk:latest
   ```
4. Compilation
   ```
   cmake . && make -j
   ```
### Local
1. Install dependencies
   - g++ (>= 13.3), [CMake](https://cmake.org) (>= 3.12)
   - [Eigen](https://libeigen.gitlab.io) (>= 3.4)
   - [Gurobi](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer) (>= 11.0.3)
   - [SCIP](https://github.com/scipopt/scip) (>= 9.2.2), [PaPILO](https://github.com/scipopt/papilo) (>= 2.4.2), [SoPlex](https://github.com/scipopt/soplex) (>= 7.1.4)
   - [CLI11](https://github.com/CLIUtils/CLI11) (>= 2.5.0)
   - [libcds](https://github.com/khizmax/libcds)
   - [Boost](https://www.boost.org)
   - [xtensor](https://github.com/xtensor-stack/xtensor)
  
   See ```container/fair_topk_container.def``` for installation commands on Debian or Ubuntu systems.
  
2. Compilation
   ```
   cmake . && make -j
   ```
   
Output programs: **klevel_based_method**, **klevel_based_method_2d**, **mip_based_method**, **baseline** and **baseline_2d**

## Reproducibility
1. [Download preprocessed datasets](https://www.dropbox.com/scl/fo/cilj9qgefmsuwi6ofa5ww/AEZMo3AbbaZ3ItqoqBAlk5U?rlkey=7pq6ih63y708um5ynqao3k24j&st=f9r2h7ix&dl=0)
2. Launch the container (skip this step if locally built)
3. Run programs (inside the container)
    ```
    program [-t] [-q] [-opt objective] [-f <PREPROCESSED DATASET PATH>] [-k k_value] \
        [-plb lower_bounds] [-pub upper_bounds] [-eps epsilon] \ 
        [-ns num_samples [-us]] [-nt num_threads] [-sol milp_solver]
    ```
    * program: klevel_based_method, klevel_based_method_2d, mip_based_method, baseline or baseline_2d
    * -t: Runtime experiment
    * -q: Quality validation experiment
    * -opt: Optimization objective
        * wd/wtsdiff/weightsdifference: w difference
        * u/util/utility: Utility loss
        * su/stbutil/stableutility: Utility loss with stabilization
    * -plb: Proportional lower bounds of the protected groups (round-down)
    * -pub: Proportional upper bounds of the protected groups (round-up)
    * -ns: Number of weight vectors
    * -us: Uniform weight vector sampling method
    * -nt: Number of threads (optional for klevel_based_method and mip_based_method with Gurobi solver)
    * -sol: gurobi or scip (default: gurobi)

    See below for examples of commands and their outputs.

For executing mip_based_method with Gurobi solver inside the container, you might need to apply a new [Gurobi license](https://www.gurobi.com/features/web-license-service/). Before executing mip_based_method, run the following command
```
export GRB_LICENSE_FILE=\path\to\gurobi\license
```
Make sure that the license file is accessible inside the container.

## Output samples
### Runtime experiments
#### Multiple protected groups
Command: ``` klevel_based_method -t -opt wd -f compas-50.csv -k 50 -plb 0.4 0.7 0.3 -pub 0.6 0.9 0.55 -eps 0.05 -ns 20 -nt 64 ```

Output:
``` 
k: 50 | Protected Group Proportion Bounds: [0.4, 0.6] [0.7, 0.9] [0.3, 0.55] | Epsilon: 0.05 | Optimization Goal: Weights Difference | Number of Threads: 64
3/20 fair weight vectors are found
Average run time: 1.094942e+02
```
#### Single protected group
Command:  ```mip_based_method -t -f compas-50.csv -k 50 -plb 0.4 -pub 0.6 -eps 0.05 -ns 20 -sol scip ```

Output:
``` 
k: 50 | Protected Group Proportion Bounds: [0.4, 0.6] | Epsilon: 0.05 | Optimization Goal: None
2/20 fair weight vectors are found
Average run time: 4.696330e+01
```

### Validation experiments
#### Multiple protected groups
Command:  ``` klevel_based_method -q -opt stbutil -f compas-50.csv -k 50 -plb 0.4 0.7 0.3 -pub 0.6 0.9 0.55 -eps 0.05 -ns 50 -us -nt 128 ```

Output:
``` 
k: 50 | Protected Group Proportion Bounds: [0.4, 0.6] [0.7, 0.9] [0.3, 0.55] | Epsilon: 0.05 | Optimization Goal: Stable Utility | Number of Threads: 128
5/50 input weight vectors are fair
7/45 fair weight vectors are found
Average weight vector difference: 1.812736e-01
Average utility loss: 7.991146e-03
```

#### Single protected group
Command:  ``` mip_based_method -q -opt weightsdifference -f compas-50.csv -k 50 -plb 0.4 -pub 0.6 -eps 0.05 -ns 50 -us -nt 64 -sol gurobi ```

Output:
``` 
k: 50 | Protected Group Proportion Bounds: [0.4, 0.6] | Epsilon: 0.05 | Optimization Goal: Weights Difference | Number of Threads: 64
6/50 input weight vectors are fair
8/44 fair weight vectors are found
Average weight vector difference: 7.333302e-02
Average protected group proportion: 6.000000e-01
Average utility loss: 4.912341e-03
``` 