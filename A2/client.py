import os
import numpy as np
import pandas as pd
from phe import paillier

# Generate the public and private keys for Paillier encryption
public_key, private_key = paillier.generate_paillier_keypair()

# Encrypt the public and private keys separately
public_key_encrypted = str(public_key.n)
private_key_encrypted = str(private_key.n)

if __name__ == "__main__":
    # Load the training and test data
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    # Separate the features and labels
    X_train = train_data.drop("Outcome", axis=1).to_numpy()
    y_train = train_data["Outcome"].to_numpy()
    X_test = test_data.drop("Outcome", axis=1).to_numpy()
    y_test = test_data["Outcome"].to_numpy()

    # Encrypt the training data and labels
    X_train_encrypted = [[private_key.encrypt(x) for x in row] for row in X_train]
    y_train_encrypted = [private_key.encrypt(y) for y in y_train]
    
    # Encrypt the testing data and labels
    X_test_encrypted = [[private_key.encrypt(x) for x in row] for row in X_test]
    y_test_encrypted = [private_key.encrypt(y) for y in y_test]

    # Save the encrypted data on client disk (to be sent to server)
    if not os.path.exists("encrypted"):
        os.makedirs("encrypted")

    np.save("encrypted/X_train.npy", X_train_encrypted)
    np.save("encrypted/y_train.npy", y_train_encrypted)
    np.save("encrypted/X_test.npy", X_test_encrypted)
    np.save("encrypted/y_test.npy", y_test_encrypted)
    with open('encrypted/public_key.txt', 'w') as f:
        f.write(public_key_encrypted)
    with open('encrypted/private_key.txt', 'w') as f:
        f.write(private_key_encrypted)
