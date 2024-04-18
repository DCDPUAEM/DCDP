# The necessary libraries are imported
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

#Dummy test
# A header is displayed using st.header to indicate the purpose of the app.
st.header("Fish Weight Prediction App")

# A text input field is created using st.text_input to allow the user to 
# enter their name. The key parameter is set to "name" to uniquely identify 
# this input field.
st.text_input("Enter your Name: ", key="name")

# This dataset is used for the fish weight prediction.
# <<<<<<< HEAD
# # Check issues with tokens in private repos... later
# #data = pd.read_csv("https://raw.githubusercontent.com/RosalesRM/ABIChallenge_MauricioRosales/Master/data/Fish.csv?token=GHSAT0AAAAAACD52C62NDIBWHUWYJDGKG2GZEL3JNQ")
# =======

# data = pd.read_csv("https://raw.githubusercontent.com/RosalesRM/ABIChallenge_MauricioRosales/Master/data/Fish.csv?token=GHSAT0AAAAAACD52C63AIOCZIU2Q3WDEI4KZEL3S3Q")
# >>>>>>> d8dba969838a0c98d142e9112062af5dd7e62d2c


# Load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# The XGBoost regressor model is loaded from the "best_model.json" file using xgb.XGBRegressor and load_model.
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

# if st.checkbox('Show Training Dataframe'):
#     data
    
# The user is prompted to select the relevant features of the fish using sliders 
# and a radio button. The sliders allow the user to select values for vertical 
# length, diagonal length, cross length, height, and diagonal width. 
# The radio button allows the user to select the name of the fish species.
st.subheader("Please select relevant features of your fish!")
left_column, right_column = st.columns(2)
with left_column:
    inp_species = st.radio(
        'Name of the fish:',
        ['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish'])

input_Length1 = st.slider('Vertical length(cm)', 0.0, 59.0, 1.0)
input_Length2 = st.slider('Diagonal length(cm)', 0.0, 63.5, 1.0)
input_Length3 = st.slider('Cross length(cm)', 0.0, 68.0, 1.0)
input_Height = st.slider('Height(cm)', 0.0, 19.0, 1.0)
input_Width = st.slider('Diagonal width(cm)', 0.0, 8.5, 1.0)

# When the "Make Prediction" button is clicked, the selected fish species is 
# transformed using the label encoder, and the input features are combined 
# into an array. The XGBoost model is used to predict the fish weight based 
# on the inputs.
if st.button('Make Prediction'):
    input_species = encoder.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")
