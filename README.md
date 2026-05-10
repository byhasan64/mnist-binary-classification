MNIST Binary Classification
A binary classifier that answers one question: is this handwritten digit a 5?
Built as a focused introduction to binary classification concepts — cross-validation, confusion matrices, precision/recall tradeoffs, and threshold tuning via the decision function.
Overview
The MNIST dataset contains 70,000 labelled images of handwritten digits (0–9). This project trains an SGD classifier on 60,000 of them, targeting the digit 5 as the positive class, and walks through the key evaluation techniques that go beyond simple accuracy.
Key Concepts Covered

Why accuracy misleads — a naive "never predict 5" classifier scores >90%, since only ~10% of images are 5s
Precision vs Recall — understanding the tradeoff for imbalanced classes
Threshold tuning — using decision_function() instead of predict() to control the classification boundary
Manual cross-validation — implementing StratifiedKFold from scratch to understand what cross_val_score does under the hood

Setup
bashpip install numpy scikit-learn matplotlib
Then run:
bashpython mnist_binary.py
Output

Sample digit visualised from the dataset
Per-fold accuracy from manual cross-validation
Confusion matrix and classification report
Precision, recall, and F1 score at the tuned threshold
Decision scores printed for threshold comparison (0 vs 8000)
