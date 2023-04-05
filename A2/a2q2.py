import os
import numpy as np
import pandas as pd
from phe import paillier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Client:

    def __init__(self, key_length):
        # Generate the public and private keys for Paillier encryption
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.private_key = public_key, private_key

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
        X_train_encrypted = [[self.public_key.encrypt(x) for x in row] for row in X_train]
        y_train_encrypted = [self.public_key.encrypt(y) for y in y_train]
        
        # Encrypt the testing data and labels
        X_test_encrypted = [[self.public_key.encrypt(x) for x in row] for row in X_test]
        y_test_encrypted = [self.public_key.encrypt(y) for y in y_test]

        return X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted
    
    def eval(self, encrypted_predictions):
        y_pred = [self.private_key.decrypt(y) for y in encrypted_predictions]
        y_test = [self.private_key.decrypt(y) for y in self.y_test_encrypted]
        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy


class Server:

    def __init__(self, public_key):
        self.public_key = public_key #Received from client
        self.X_train_encrypted = X_train_encrypted
        self.y_train_encrypted = y_train_encrypted
        self.X_test_encrypted = X_test_encrypted
        self.y_test_encrypted = y_test_encrypted

    def train_model(self, kernel, C):
        # Decrypt the training data and labels using the public key
        X_train_decrypted = [self.public_key.decrypt(x) for x in self.X_train_encrypted]
        y_train_decrypted = [self.public_key.decrypt(x) for x in self.y_train_encrypted]

        # Train the SVM model 
        svm_model = SVC(kernel=kernel, C=C)
        svm_model.fit(X_train_decrypted, y_train_decrypted)
        return svm_model

    def predict(self, input_model):
        # Decrypt the test data using the public key
        X_test_decrypted = [public_key.decrypt(x) for x in self.X_test_encrypted]
        # Make predictions on the decrypted test data using the trained model
        test_predictions = input_model.predict(X_test_decrypted)
        # Encrypt the test predictions using the private key
        encrypted_predictions = [self.public_key.encrypt(x) for x in test_predictions]
        return encrypted_predictions


if __name__ == "__main__":
    client = Client((1024))
    X_train, y_train, X_test, y_test = load_data("client/train.csv", "client/test.csv", "Outcome")
    X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted = encrypt_data(X_train, y_train, X_test, y_test)

    #Send Encrypted data and public key to server
    server = Server(client.public_key, X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted)
    svm_model = server.train_model("linear", 1)

    #Send Encrypted predictions to client back
    encrypted_preds = server.predict(svm_model)
    test_accuracy = eval(encrypted_preds)
    print('Test accuracy:', test_accuracy)




 
