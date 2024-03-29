import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to evaluate model performance
def evauation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score

# Load the dataset
data = pd.read_csv("data/Fish.csv")
data_cleaned = data.drop("Weight", axis=1)
y = data['Weight']

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data_cleaned, y, test_size=0.2, random_state=42)

# Label encode the 'Species' column
label_encoder = LabelEncoder()
x_train['Species'] = label_encoder.fit_transform(x_train['Species'].values)
x_test['Species'] = label_encoder.transform(x_test['Species'].values)

# Save label encoder classes
np.save('classes.npy', label_encoder.classes_)

# Load the trained model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

# Make predictions on the test set
pred = best_xgboost_model.predict(x_test)

# Evaluate the model
score_MSE, score_MAE, score_r2score = evauation_model(pred, y_test)
print(score_MSE, score_MAE, score_r2score)
#%%
# Load the label encoder classes
loaded_encoder = LabelEncoder()
loaded_encoder.classes_ = np.load('classes.npy',allow_pickle=True)

#%%
# Example usage: Predicting on new inputs
print(x_test.shape)
input_species = loaded_encoder.transform(np.expand_dims("Parkki",-1))
print(int(input_species))
inputs = np.expand_dims([int(input_species),15,20,10,4,5],0)
print(inputs.shape)
prediction = best_xgboost_model.predict(inputs)
print("final pred", np.squeeze(prediction,-1))

