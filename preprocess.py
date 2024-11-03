import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def load_and_preprocess_data(filename):
    # Load the data
    df = pd.read_csv(filename)

    # Remove the timestamp column (assuming its name is 'Timestamp' or similar)
    # Replace 'Timestamp' with the actual name of the timestamp column in your dataset
    if 'Timestamp' in df.columns:
        df.drop(columns=['Timestamp'], inplace=True)

    # Specify the target column
    target_column = 'What was your CGPA in the previous academic semester/term?  '
    
    # Step 1: Clean and convert the target column to numeric
    df[target_column] = pd.to_numeric(
        df[target_column].replace({'No CGPA ': None, 'Nill': None, '84%': None, '1st year': None, '3.4 / 4': None}),
        errors='coerce'
    )
    
    # Step 2: Fill NaN values in the target column with the median
    df[target_column].fillna(df[target_column].median(), inplace=True)

    # Step 3: Handle missing values in other columns if necessary
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Step 4: Encode categorical columns
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Step 5: Split data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Step 6: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 7: Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

def preprocess_input(data, label_encoders, scaler):
    # Convert input to DataFrame for easy manipulation
    df = pd.DataFrame([data])

    # Encode categorical variables
    for column, le in label_encoders.items():
        if column in df:
            # Check if the value exists in the encoder's classes
            if df[column].values[0] in le.classes_:
                df[column] = le.transform(df[column])
            else:
                # If unseen label, handle it gracefully
                df[column] = np.nan  # Set to NaN or use a default value if needed

    # Scale the data
    scaled_data = scaler.transform(df.fillna(0))  # Fill NaN with 0 or appropriate default
    return scaled_data
