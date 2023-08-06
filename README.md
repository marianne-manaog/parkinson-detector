# Cassandra and Machine Learning-driven Parkinson's Disease Detector
This repository hosts Python codes to detect Parkinson's Disease leveraging speech data, Apache Cassandra's vector search similarity, and ensembles of Decision Trees.

## Setup

* Updating conda

Please update conda by running:

`conda update -n base -c defaults conda`

* Configuration of the environment

Please create a conda virtual environment and install all required dependencies of this application by
running: 

`conda env create -f environment.yml`

* Activating and deactivating the conda environment

To activate this environment, please run:

`conda activate pd_detector`

To install and build the codes as a Python package in editable mode from the top-level directory:
`pip install -e .`

To deactivate an active environment, please run the following command:

`conda deactivate`

## Datasets
Five speech-related datasets are used in this project. These files are taken from the University California Irvine
Machine Learning repository (Little, 2008, 2009; Naranjo et al., 2016; Sakar et al., 2013; Sakar et al., 2018) and 
the references are provided in the corresponding sub-section below.

## Data analysis and creation of train and test datasets
- The five speech datasets of interest are analysed via descriptive statistics as per the 
module `src/analyse_data/analyse_speech_datasets.py`.
- Thereafter, the datasets are standardised via the module `src/process_data/prepare_data.py`, which 
ensures the target column `status` is named consistently (`1` for patients with Parkinson's Disease, 
`0` for healthy subjects), that only the relevant columns are retained and that are renamed consistently too.
- Eventually, the data are combined into two sets (train and test) via the module 
`src/create_train_and_test_data/merge_speech_data.py`.

## References

- Little, M. (2008) Parkinsons data set. UCI Machine Learning Repository.
- Little, M. (2009) Parkinsons Telemonitoring data set. UCI Machine Learning Repository.
- Naranjo, L., Perez, C. J., Campos-Roca, Y., & Martin, J. (2016) Addressing voice recording replications for 
Parkinson’s disease detection. Expert Systems with Applications 46: 286-292.
- Sakar, B. E., Isenkul, M. E., Sakar, C. O., Sertbas, A., Gurgen, F., Delil, S., ... & Kursun, O. (2013) 
Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings. 
IEEE Journal of Biomedical and Health Informatics 17(4): 828-834.
- Sakar, C., Serbes, G., Gunduz, A., Nizam, H., and Sakar, B. (2018) 
Parkinson's Disease Classification. UCI Machine Learning Repository.
