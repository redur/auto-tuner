# Code to Master Thesis "Quantum Dot Charge State Auto Tuner"

TO BE EDITED!

Bla bla project description


# Structure

```

├── LICENSE
│
│
├── README.md                <- The top-level README for developers using this project
│
├── environment.yml          <- The requirements file for reproducing the analysis environment, e.g.
│                                generated with `pip freeze > requirements.txt`
│
├── additional_models        <- Folder for additional models, e.g. Language models from Spacy, nltk,...
│
│
├── bin                      <- Stuff to be deleted
│
│
├── data
│   ├── external             <- Data from third party sources.
│   ├── processed            <- The final, canonical data sets for modeling.
│   └── raw                  <- The original, immutable data dump.
│
│
├── documents                <- Documentation of data etc.
│   ├── docs
│   ├── images
│   └── references           <- Data dictionaries, manuals, and all other explanatory materials.
│
│
├── misc                     <- miscellaneous
│
│
├── notebooks                <- Jupyter notebooks. Every developper has its own folder for exploratory
│   ├── name                    notebooks. Usually every model has its own notebook where models are
│   │   └── exploration.ipynb   tested and optimized.
│   └── model
│       └── model_exploration.ipynb <- different optimized models can be compared here if preferred    
│
│
├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures               <- Generated graphics and figures to be used in reporting
│
│
├── results
│   ├── outputs
│   └── models               <- Trained and serialized models, model predictions, or model summaries
│
│
├── scores                   <- Cross validation scores are saved here. (Automatically generated)
│   └── model_name           <- every model has its own folder. 
│
├── src                      <- Source code of this project.
│   ├── __init__.py          <- Makes src a Python module
│   ├── programs.py          <- main calls a function from programs. These functions represent different steps
│   │                           within the workflow.
│   ├── data                 <- Scripts to download or generate data
│   │   └── data_extraction.py
│   │
│   ├── process              <- Scripts to turn raw data into features for modeling
│   │   └── processing.py
│   │
│   │
│   └── utils                <- Scripts to create exploratory and results oriented visualizations
│       └── exploration.py      / functions to evaluate models
│       └── evaluation.py

```
- create a python env based on a list of packages from environment.yml    
  ```conda env create -f environment.yml -n env_auto_tuner```
 

# Instruction
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

  
