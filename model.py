import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('data/studs.csv')

# Function to assign study and sleep hours
def assign_hours(cgpa, max_cgpa, min_cgpa):
    normalized_cgpa = (cgpa - min_cgpa) / (max_cgpa - min_cgpa)
    hours_req_per_unit = round((1 - normalized_cgpa) * 11 + 1)  # Scale to 1-12
    sleep_hours = round(normalized_cgpa * 7 + 1)  # Scale to 1-8
    return hours_req_per_unit, sleep_hours

# Determine max and min CGPA for normalization
max_cgpa = data['cgpa'].max()
min_cgpa = data['cgpa'].min()

# Apply the function to create new columns
data[['hours_req_per_unit', 'hours_of_sleep']] = data['cgpa'].apply(lambda x: assign_hours(x, max_cgpa, min_cgpa)).apply(pd.Series)

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'parental_level_of_education', 'lunch', 'test_preparation']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data[['gender', 'parental_level_of_education', 'lunch', 'test_preparation', 'hours_req_per_unit', 'hours_of_sleep', 'lastsemcgpa', 'cgpabefore']]
y = data['cgpa']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models and store results
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostRegressor()
}

# Train and evaluate models
trained_models = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[model_name] = model

def predict_cgpa(input_data):
    input_df = pd.DataFrame(input_data)
    predictions = {}
    for model_name, model in trained_models.items():
        pred = model.predict(input_df)
        predictions[model_name] = pred[0]  # Get the first prediction
    return predictions
