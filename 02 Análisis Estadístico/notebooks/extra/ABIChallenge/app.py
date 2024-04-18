# The necessary libraries are imported
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

# Load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# The XGBoost regressor model is loaded from the "best_model.json" file using xgb.XGBRegressor and load_model.
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

def evauation_model(inp_species, input_Length1, input_Length2, input_Length3, input_Height, input_Width):
  input_species = encoder.transform(np.expand_dims(inp_species, -1))
  inputs = np.expand_dims(
      [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
  prediction = best_xgboost_model.predict(inputs)
  print("final pred", np.squeeze(prediction, -1))

