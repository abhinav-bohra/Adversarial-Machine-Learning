#Question 2
import os
import math
import pandas as pd
import phe.encoding
from phe import paillier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#-------------------------------------------------------------------------------------------------------------------------------------------------
# UTIL FUNCTIONS
#-------------------------------------------------------------------------------------------------------------------------------------------------
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

def load_data(input_file, target_feature):
  # Load data and separate the features and labels
  data = pd.read_csv(input_file).dropna()[:10]
  X = data.drop(target_feature, axis=1).values.tolist()
  y = data[target_feature].values.tolist()
  return X, y
  
#-------------------------------------------------------------------------------------------------------------------------------------------------
# CLIENT CLASS
#-------------------------------------------------------------------------------------------------------------------------------------------------
class Client:

    def __init__(self, key_length):
        # Generate the public and private keys for Paillier encryption
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.private_key = public_key, private_key

    def encrypt_data(self, input_file, target_feature):        
        # Encrypt the testing data and labels
        X_test, y_test = load_data(input_file, target_feature)
        X_test_encrypted = [encrypt_vector(x, self.public_key) for x in X_test]
        y_test_encrypted = encrypt_vector(y_test, self.public_key)
        self.X_test_encrypted = X_test_encrypted
        self.y_test_encrypted = y_test_encrypted

        return X_test_encrypted, y_test_encrypted
    
    def eval(self, encrypted_predictions):
        logits = decrypt_vector(encrypted_predictions, self.private_key)
        y_pred = [1 if l>0 else 0 for l in logits]
        y_test = decrypt_vector(self.y_test_encrypted, self.private_key)
        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy

#-------------------------------------------------------------------------------------------------------------------------------------------------
# SERVER CLASS
#-------------------------------------------------------------------------------------------------------------------------------------------------

class Server:

    def __init__(self, public_key, X_test_encrypted):
        self.public_key = public_key #Received from client
        self.X_test_encrypted = X_test_encrypted
        self.y_test_encrypted = y_test_encrypted

    def train_model(self, X_train, y_train, hyperparams):
        # Train the SVM model
        svm_model = SVC(kernel=hyperparams['kernel'], C=hyperparams['C'], gamma=hyperparams['gamma'])
        svm_model.fit(X_train, y_train)
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

#-------------------------------------------------------------------------------------------------------------------------------------------------
# DRIVER CODE
#-------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  # Parameters
  key_length = 1024
  target_feature = "Outcome"
  hyperparameters = {'kernel':'linear', 'C':1, 'gamma':'auto'}

  # Instantiate Client
  client = Client((key_length))
  # Encrypt data on client machine
  X_test_encrypted, y_test_encrypted = client.encrypt_data("client/test.csv", target_feature)

  # Send encrypted data and public key to server and train the model
  server = Server(client.public_key, X_test_encrypted)
  # Load training data
  X_train, y_train = load_data("server/train.csv", target_feature)
  # Train SVM Classifier
  svm_model = server.train_model(X_train, y_train, hyperparameters)
  encrypted_preds = server.predict(svm_model)

  # Send encrypted predictions back to client and evaluate accuracy
  test_accuracy = client.eval(encrypted_preds)
  print('Test accuracy:', test_accuracy)