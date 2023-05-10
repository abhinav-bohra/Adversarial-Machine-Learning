# Adversarial-Machine-Learning

Adversarial machine learning is a technique that aims to deceive machine learning models by providing misleading input. This method includes both the generation and detection of adversarial examples, which are inputs created to deceive classifiers. An Adversarial Attack is a technique used to find a perturbation that alters the prediction of a machine learning model. The perturbation can be minuscule and imperceptible to human eyes. This project aims to familiarize individuals with adversarial training and example construction.

### Part 1: Generating adversarial samples for Attacks on ML systems for image classification

We generate an adversarial example for a targeted attack by adding random noise to the original image. This adversarial example is designed to fool the Resnet-50 model into misclassifying the image

![Adversarial-ML](https://github.com/abhinav-bohra/Adversarial-Machine-Learning/blob/main/image.png)

###  Part 2: Exploring Adversarial Attack Methods: A Comparative Study

Next, we explored different methods for constructing adversarial examples:

1. **Fast Gradient Sign Method (FGSM):** By varying the epsilon value between 0.1 and 0.001, we applied FGSM to generate five different adversarial examples. FGSM perturbs the original image based on the gradient information to maximize the loss and induce misclassification.
2. **Projected Gradient Descent (PGD):** We utilized PGD to iteratively optimize the perturbation within a specified constraint. We applied PGD with infinity norm to generate adversarial examples and also explored targeted attacks using PGD.
3. **PGD with L2 Norm:** By varying the epsilon and alpha values between 0.1 and 0.001, we applied PGD with L2 norm to generate five adversarial examples for each value. This method aimed to minimize the perturbation in the L2 space while inducing misclassification.

We compared the misclassification rates for all the methods. The results were documented in a table with epsilon, alpha, and iterations as columns, and the optimization methods (FGSM, PGD, PGD with infinity norm, PGD with targeted attack, and PGD with L2 norm) as rows. This comparison helped us assess the effectiveness of each method in generating adversarial examples.

------------


This project was completed as part of the programming assignment for the course **Dependable and Secure AI-ML (AI60006).**
