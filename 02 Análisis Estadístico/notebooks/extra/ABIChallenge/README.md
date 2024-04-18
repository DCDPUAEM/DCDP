# ABIChallenge_MauricioRosales
This repo is related to AB InBev MLOps Challenge v7

The repository contains the following:

* flowchart.md - It includes a basic diagram for deploying a pre-trained machine learning model.
* model.py - This script is used for training various tree-based algorithms, searching for the best hyperparameters using Hyperopt, and obtaining the best model, which is saved in JSON format.
* prediction.py - This script is used to test the model obtained from the 'model.py' script. It encodes the input parameters and provides an example of using the XGBRegressor model, resulting in the predicted weight of a fish species based on its size and species characteristics.

## Additional Notes
For a limited period of time, the user can test the model in this [page](https://rosalesrm-abichallenge-mauriciorosales-main-zajvqp.streamlit.app/)
