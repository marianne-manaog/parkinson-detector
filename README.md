# Cassandra and Machine Learning-driven Parkinson's Disease Detector
This repository hosts Python codes to detect Parkinson's Disease leveraging speech data, Apache Cassandra's vector search similarity, and ensembles of Decision Trees.

## Datasets
Five speech-related datasets are used in this project. These files are taken from the University California Irvine
Machine Learning repository (Little, 2008, 2009; Naranjo et al., 2016; Sakar et al., 2013; Sakar et al., 2018) and 
the references are provided in the corresponding sub-section below.

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

To deactivate an active environment, please run the following command:

`conda deactivate`

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
