import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import anfis_core

def load_data(input_features, decision_feature, rows=0, path="data.csv"):
    """Load dataset"""
    df = pd.read_csv(path)
    if rows == 0:
        rows = len(df)
    df = df.iloc[:rows]
    X = df[input_features].values.astype(float)
    y = df[decision_feature].values.astype(float)
    
    # Normalize features to [0, 1]
    X_scaler = MinMaxScaler()
    X_normalized = X_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2)

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler

def run_cli_interface(trained_mfs, trained_consequents, X_scaler):
    """Listens for user input in console"""
    while True:
        try:
            user_input = input("\nEnter person age, person income,person employent experience, person credit history length (comma-separated), or 'exit' to quit:\n> ")
            if user_input.lower() == 'exit':
                break
            values = np.array([float(x.strip()) for x in user_input.split(',')])
            if len(values) != 4:
                print("Please enter exactly 4 values.")
                continue

            # Normalize the input using the same scaler
            normalized_values = X_scaler.transform([values])

            pred = anfis_core.predict(normalized_values, trained_mfs, trained_consequents)
            pred_val = pred[0][0] if pred.ndim > 1 else pred[0]
            print(f"Predicted credit score: {pred_val:.2f}")

        except Exception as e:
            print("Error:", e)

