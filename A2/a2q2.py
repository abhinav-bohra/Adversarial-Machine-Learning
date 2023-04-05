import os
import numpy as np
import pandas as pd
from phe import paillier

class Client:

    def __init__(self, key_length):
        # Generate the public and private keys for Paillier encryption
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.private_key = public_key, private_key

        # Encrypt the public and private keys separately
        self.public_key_encrypted, self.private_key_encrypted = str(self.public_key.n), str(self.private_key.n)

    def load_data(train_file, test_file, target_feature):
        # Load the training and test data
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        
        # Separate the features and labels
        X_train = train_data.drop(target_feature, axis=1).to_numpy()
        y_train = train_data[target_feature].to_numpy()
        X_test = test_data.drop(target_feature, axis=1).to_numpy()
        y_test = test_data[target_feature].to_numpy()
        return X_train, y_train, X_test, y_test
    
    def encrypt_data(X_train, y_train, X_test, y_test):
        # Encrypt the training data and labels
        X_train_encrypted = [[self.private_key.encrypt(x) for x in row] for row in X_train]
        y_train_encrypted = [self.private_key.encrypt(y) for y in y_train]
        
        # Encrypt the testing data and labels
        X_test_encrypted = [[self.private_key.encrypt(x) for x in row] for row in X_test]
        y_test_encrypted = [self.private_key.encrypt(y) for y in y_test]

        return X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted


# class Server:

#     def __init__(self, key_length):



if __name__ == "__main__":
    client = Client((10))
    X_train, y_train, X_test, y_test = load_data("client/train.csv", "client/test.csv", "Outcome")
    X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted = encrypt_data(X_train, y_train, X_test, y_test)

    # server = Server()
    print(X_train_encrypted[0])




 
