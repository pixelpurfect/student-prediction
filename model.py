import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from preprocess import load_and_preprocess_data

def train_model(filename):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, label_encoders = load_and_preprocess_data(filename)
    
    # Initialize the model
    model = RandomForestRegressor(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save the model and preprocessing objects
    joblib.dump(model, "trained_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

# Call the train_model function with your dataset
train_model("Real Time Dataset - Form responses 1.csv")
