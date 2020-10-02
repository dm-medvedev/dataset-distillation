# Dataset Distillation
The code was forked from the [initial project](https://github.com/SsnL/dataset-distillation) and changed by [Dmitry Medvedev](https://github.com/dm-medvedev).  
This project contains code of experiments for [coursework](https://github.com/dm-medvedev/dataset-distillation/blob/master/docs/presentation.pdf)

# Fast Experiments Restart
Download the data from [google drive](https://drive.google.com/file/d/1zvu7ywHsaG8Ek2G_atzmSy_rycptDeYd/view?usp=sharing) and unzip it into the project directory. You can now use the following commands to restart experiments with already distilled data:  
``python experiment_whole_data.py --results_dir ./Results/experiment_whole_data``  
``python experiment_general_distillation.py --results_dir ./Results/experiment_general_distillation``  
``python experiment_strategies.py``  

**Note:** if you cannot find any experiments, it is because the code is under refactoring and all experiments will be added soon.

## Prerequisites

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies
- ``torch >= 1.0.0``
- ``torchvision >= 0.2.1``
- ``numpy``
- ``matplotlib``
- ``pyyaml``
- ``tqdm``
