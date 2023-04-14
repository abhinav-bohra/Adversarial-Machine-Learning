import os
import math
import numpy as np
import pandas as pd
import phe.encoding
from phe import paillier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class ExampleEncodedNumber(phe.encoding.EncodedNumber):
    BASE = 64
    LOG2_BASE = math.log(BASE, 2)

def encrypt_vector(vec, public_key):
  encoded_vector = [ExampleEncodedNumber.encode(public_key, v) for v in vec]
  encrypted_vector = [public_key.encrypt(ev) for ev in encoded_vector]
  return encrypted_vector

def decrypt_vector(vec, key):
  decoded_vector = [key.decrypt_encoded(x, ExampleEncodedNumber) for x in vec]
  decrypted_vector = [d.decode() for d in decoded_vector]
  return decrypted_vector

class Client:

    def __init__(self, key_length):
        # Generate the public and private keys for Paillier encryption
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.private_key = public_key, private_key

    def load_data(self, train_file, test_file, target_feature):
        # Load the training and test data
        train_data = pd.read_csv(train_file).dropna()[:10]
        test_data = pd.read_csv(test_file).dropna()[:10]
        
        # Separate the features and labels
        X_train = train_data.drop(target_feature, axis=1).values.tolist()
        y_train = train_data[target_feature].values.tolist()
        X_test = test_data.drop(target_feature, axis=1).values.tolist()
        y_test = test_data[target_feature].values.tolist()
        return X_train, y_train, X_test, y_test
    
    def encrypt_data(self, X_train, y_train, X_test, y_test):
        # Encrypt the training data and labels
        X_train_encrypted = [encrypt_vector(x, self.public_key) for x in X_train]
        y_train_encrypted = encrypt_vector(y_train, self.public_key)
        
        # Encrypt the testing data and labels
        X_test_encrypted = [encrypt_vector(x, self.public_key) for x in X_test]
        y_test_encrypted = encrypt_vector(y_test, self.public_key)

        return X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted
    
    def eval(self, encrypted_predictions):
        y_pred = [self.private_key.decrypt(y) for y in encrypted_predictions]
        y_test = [self.private_key.decrypt(y) for y in self.y_test_encrypted]
        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy


class Server:

    def __init__(self, public_key, X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted):
        self.public_key = public_key #Received from client
        self.X_train_encrypted = X_train_encrypted
        self.y_train_encrypted = y_train_encrypted
        self.X_test_encrypted = X_test_encrypted
        self.y_test_encrypted = y_test_encrypted

    def train_model(self, kernel, C):
        # Decrypt the training data and labels using the public key
        # X_train_decrypted = [decrypt_vector(x, self.public_key) for x in self.X_train_encrypted]
        # y_train_decrypted = decrypt_vector(self.y_train_encrypted, self.public_key)

        # Train the SVM model 
        svm_model = SVC(kernel=kernel, C=C)
        # svm_model.fit(X_train_encrypted, y_train_encrypted)
        svm_model.fit(X_train_encrypted, y_train_encrypted)
        return svm_model

    def predict(self, input_model):
        # Decrypt the test data using the public key
        X_test_decrypted = [public_key.decrypt(x) for x in self.X_test_encrypted]
        # Make predictions on the decrypted test data using the trained model
        test_predictions = input_model.predict(X_test_decrypted)
        # Encrypt the test predictions using the private key
        encrypted_predictions = [self.public_key.encrypt(x) for x in test_predictions]
        return encrypted_predictions


# if __name__ == "__main__":
client = Client((1024))
X_train, y_train, X_test, y_test = client.load_data("client/train.csv", "client/test.csv", "Outcome")
X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted = client.encrypt_data(X_train, y_train, X_test, y_test)

#Send Encrypted data and public key to server
server = Server(client.public_key, X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted)
svm_model = server.train_model("linear", 1)

# #Send Encrypted predictions to client back
encrypted_preds = server.predict(svm_model)
test_accuracy = client.eval(encrypted_preds)
print('Test accuracy:', test_accuracy)