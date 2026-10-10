import pandas as pd
import numpy as np

def load_and_fill_nulls(filename, fill_value=0):
    """
    Load data from a CSV file and fill null values with specified value.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file to load
    fill_value : float or int, default=0
        The value to use for filling null/missing values
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with null values filled
    """
    
    # TODO: Use pandas to load the file
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # TODO: Print the number of null values in each column
    print("\nNull values per column before filling:")
    
    # TODO: Fill nulls with fill_value
    
    # TODO: Verify null values after filling
    print("\nNull values per column after filling:")
    
    return df

# ============================
# MAIN: MINIBATCH INFERENCE + ACCURACY
# ============================
if __name__ == "__main__":
    df = load_and_fill_nulls("sample.csv", fill_value=0)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    n_samples, n_features = X.shape

    # Untrained parameters
    W = np.zeros((n_features, 1))
    b = 0.0

    batch_size = 32

    print("\nRunning minibatch inference only...\n")

    preds_all = []

    for i in range(0, n_samples, batch_size):
        Xb = X[i:i+batch_size]
        preds = Xb @ W + b     # forward pass only
        preds_all.append(preds)

        print(f"Batch {i//batch_size + 1}: prediction shape = {preds.shape}")

    # Combine batch predictions
    preds_all = np.vstack(preds_all)

    # ======================
    # Accuracy calculation
    # ======================
    # Convert regression output → class label (0/1)
    y_pred_class = (preds_all >= 0.5).astype(int)

    accuracy = np.mean(y_pred_class == y)

    print("\nInference completed.")
    print("Accuracy (with untrained weights):", accuracy)
    print("Final W:", W.ravel())
    print("Final b:", b)
