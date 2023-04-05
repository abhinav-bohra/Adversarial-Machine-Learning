import numpy as np
from phe import paillier

import numpy as np
import os

from svm import SVM
from phe import paillier

# Load the encrypted keys from disk
with open('encrypted/public_key.txt', 'r') as f:
    public_key = paillier.PaillierPublicKey(int(f.read()))
with open('encrypted/private_key.txt', 'r') as f:
    private_key = paillier.PaillierPrivateKey(int(f.read()))

# Load the encrypted training data and labels from disk
X_train_encrypted = np.load('encrypted/X_train_encrypted.npy', allow_pickle=True)
y_train_encrypted = np.load('encrypted/y_train_encrypted.npy', allow_pickle=True)

# Decrypt the training data and labels
X_train = np.array([[private_key.decrypt(x) for x in row] for row in X_train_encrypted])
y_train = np.array([private_key.decrypt(y) for y in y_train_encrypted])

# Decrypt the testing data and labels
X_test = np.array([[private_key.decrypt(x) for x in row] for row in X_test_encrypted])
y_test = np.array([private_key.decrypt(y) for y in y_test_encrypted])

# Define the SVM model
class SVM:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        n, d = X.shape
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q[i][j] = y[i] * y[j] * np.dot(X[i], X[j])
        P = np.ones(n) * -1
        G = np.eye(n) * -1
        h = np.zeros(n)
        A = y.reshape(1, -1)
        b = np.zeros(1)

        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
        P = matrix(Q)
        q = matrix(P)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        sol = solvers.qp(P=q, q=matrix(P), G=matrix(G), h=matrix(h), A=matrix(A), b=matrix(b))
        alpha = np.array(sol['x']).flatten()

        # Compute the SVM weights and bias
        w = np.zeros(d)
        for i in range(n):
            w += alpha[i] * y[i] * X[i]
        b = 0
        for i in range(n):
            if alpha[i] > 1e-4:
                b = y[i] - np.dot(w, X[i])
                break

if __name__ == "__main__":
        
    # Train the SVM model on the decrypted training data and labels
    svm_model = SVM()
    svm_model.fit(X_train, y_train)

    # Compute the accuracy of the SVM model on the test data
    X_test_encrypted = [[public_key.encrypt(x) for x in row] for row in X_test]
    y_test_encrypted = [public_key.encrypt(y) for y in y_test]
    X_test_decrypted = np.array([[private_key.decrypt(x) for x in row] for row in X_test_encrypted])
    y_test_decrypted = np.array([private_key.decrypt(y) for y in y_test_encrypted])
    y_test_predicted = np.sign(np.dot(X_test_decrypted, svm_model.w) + svm_model.b)
    accuracy = np.mean(y_test_predicted == y_test_decrypted)

    # Print the accuracy of the SVM model on the test data
    print(f"Accuracy of SVM model: {accuracy}")
