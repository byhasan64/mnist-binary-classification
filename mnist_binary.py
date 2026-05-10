# =============================================================================
# MNIST Binary Classification
# Core: predict whether a given image is the digit 5
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# =============================================================================
# 1. Load Data
# =============================================================================

mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Visualise the first sample
some_digit = X.iloc[0].to_numpy()
plt.imshow(some_digit.reshape(28, 28), cmap="binary")
plt.axis("off")
plt.title(f"Label: {y.iloc[0]}")
plt.show()


# =============================================================================
# 2. Train / Test Split & Binary Target
# =============================================================================

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Binary target: True if digit is 5
y_train_5 = (y_train == 5)
y_test_5  = (y_test  == 5)


# =============================================================================
# 3. Train SGD Classifier
# =============================================================================

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print("Prediction for sample digit:", sgd_clf.predict([some_digit]))


# =============================================================================
# 4. Manual Cross-Validation (educational — mirrors what cross_val_score does)
# =============================================================================

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)

    X_train_fold = X_train.iloc[train_index]
    y_train_fold = y_train_5.iloc[train_index]
    X_test_fold  = X_train.iloc[test_index]
    y_test_fold  = y_train_5.iloc[test_index]

    clone_clf.fit(X_train_fold, y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)

    print(f"Fold accuracy: {sum(y_pred == y_test_fold) / len(y_pred):.4f}")


# =============================================================================
# 5. Why Accuracy Alone Is Misleading
# Only ~10% of images are 5s, so a classifier that always predicts "not 5"
# achieves >90% accuracy — without learning anything useful.
# =============================================================================

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None): return self
    def predict(self, X): return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
print("Never5 accuracy:", cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# =============================================================================
# 6. Confusion Matrix, Precision, Recall & F1
# =============================================================================

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print("\nConfusion Matrix:\n", confusion_matrix(y_train_5, y_train_pred))
print(f"Precision : {precision_score(y_train_5, y_train_pred):.4f}")
print(f"Recall    : {recall_score(y_train_5, y_train_pred):.4f}")
print(f"F1        : {f1_score(y_train_5, y_train_pred):.4f}")


# =============================================================================
# 7. Threshold Tuning via Decision Function
# predict() is equivalent to decision_function() with threshold = 0.
# Raising the threshold increases precision but lowers recall.
# =============================================================================

y_scores = sgd_clf.decision_function([some_digit])
print(f"\nDecision score for sample digit: {y_scores[0]:.2f}")

for threshold in [0, 8000]:
    pred = (y_scores > threshold)
    print(f"Threshold {threshold:>5}: predicted as 5? {pred[0]}")

# Collect scores across all training folds for plotting
y_scores_train = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
