#Question 2
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

def decrypt_vector(vec, private_key):
  decoded_vector = [private_key.decrypt_encoded(x, ExampleEncodedNumber) for x in vec]
  decrypted_vector = [d.decode() for d in decoded_vector]
  return decrypted_vector

class Client:

    def __init__(self, key_length):
        # Generate the public and private keys for Paillier encryption
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.private_key = public_key, private_key

    def load_data(self, test_file, target_feature):
        # Load the test data and separate the features and labels
        test_data = pd.read_csv(test_file).dropna()[:10]
        X_test = test_data.drop(target_feature, axis=1).values.tolist()
        y_test = test_data[target_feature].values.tolist()
        return X_test, y_test
    
    def encrypt_data(self, X_test, y_test):        
        # Encrypt the testing data and labels
        X_test_encrypted = [encrypt_vector(x, self.public_key) for x in X_test]
        y_test_encrypted = encrypt_vector(y_test, self.public_key)
        self.X_test_encrypted = X_test_encrypted
        self.y_test_encrypted = y_test_encrypted

        return X_test_encrypted, y_test_encrypted
    
    def eval(self, encrypted_predictions):
        logits = decrypt_vector(encrypted_predictions, self.private_key)
        y_pred = [1 if l>0.5 else 0 for l in logits]
        y_test = decrypt_vector(self.y_test_encrypted, self.private_key)
        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy


class Server:

    def __init__(self, public_key, X_test_encrypted, y_test_encrypted, train_file, target_feature):
        self.public_key = public_key #Received from client
        self.X_test_encrypted = X_test_encrypted
        self.y_test_encrypted = y_test_encrypted
        # Load the training data and separate the features and labels
        train_data = pd.read_csv(train_file).dropna()[:10]
        self.X_train = train_data.drop(target_feature, axis=1).values.tolist()
        self.y_train = train_data[target_feature].values.tolist()

    def train_model(self, kernel, C):
        # Train the SVM model 
        svm_model = SVC(kernel=kernel, C=C)
        svm_model.fit(self.X_train, self.y_train)
        return svm_model

    def predict(self, input_model):
        # Make predictions on the encrypted test data using the trained model
        encrypted_logits = []
        w = input_model.coef_[0]
        b = input_model.intercept_[0]
        for x in X_test_encrypted:
          score = b
          for i in range(len(x)):
              score += x[i] * w[i]
          encrypted_logits.append(score)
        return encrypted_logits


if __name__ == "__main__":
  key_length = 1024
  target_feature = "Outcome"

  #Encrypt data on client machine
  client = Client((key_length))
  X_test, y_test = client.load_data("client/test.csv", target_feature)
  X_test_encrypted, y_test_encrypted = client.encrypt_data(X_test, y_test)

  #Send encrypted data and public key to server and train the model
  server = Server(client.public_key, X_test_encrypted, y_test_encrypted, "server/train.csv", target_feature)
  svm_model = server.train_model("linear", 1)

  #Send encrypted predictions back to client and evaluate accuracy
  encrypted_preds = server.predict(svm_model)
  test_accuracy = client.eval(encrypted_preds)
  print('Test accuracy:', test_accuracy)