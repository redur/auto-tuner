# Code to Master Thesis "Quantum Dot Charge State Auto Tuner"

# Content
- Data processing, augmentation and labeling routine
- auto-tuner
- code used for generating clusters of the charge induced current offset


# Structure

```

│
├── README.md                <- The top-level README for developers using this project
│
├── environment.yml          <- The requirements file for reproducing the analysis environment, e.g.
│                                generated with `pip freeze > requirements.txt`
│
│
├── data                     <- Folder where all data is located (not on GitHub due to large size)
│   ├── fine                 <- Folder for fine data
│   ├── coarse                 <- Folder for coarse data
│
├── report                   <- Report
│
│
├── src                      <- Source code of this project.
│   ├── __init__.py              <- Makes src a Python module
│   ├── clusterer                <- Scripts to download or generate data
│   │   └── make_clusters.py
│   │
│   ├── data_generation      <- Scripts to turn raw data into features for modeling
│   │   └── augmenter.py        <- class for augmentation       
│   │   └── labeler.py          <- class for automated labeling of fine frames
│   │   └── marker.py           <- class for marking of charge transition lines
│   │   └── occupation_labeler.py <- class for automated labeling of coarse frames
│   │   └── measurement_series.py <- functions for repeated measurements
│   │
│   │
│   └── utils                <- Scripts to create exploratory and results oriented visualizations / functions to evaluate the models
│       └── exploration.py      <- model exploration
│       └── evaluation.py       <- model evaluation
│       └── funcs.py            <- general functions, preprocessing of data
│       └── measurement_funcs.py <- functions used to measure with Labber
│       └── visualization.py    <- functions for visualization

```
# Instruction
In order to run the code, the following python environment needs to be installed.

## create a python env based on a list of packages from environment.yml
```conda env create -f environment.yml -n env_auto_tuner```

## update a python env based on a list of packages from environment.yml
```conda env update -f environment.yml -n env_auto_tuner```

## activate the env  
  ```activate env_auto_tuner```
  
## in case of an issue clean all the cache in conda
   ```conda clean -a -y```

## delete the env to recreate it when too many changes are done  
  ```conda env remove -n env_auto_tuner```

  
