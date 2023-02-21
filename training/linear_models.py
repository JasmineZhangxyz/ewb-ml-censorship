# Imports
from sklearn.linear_model import LogisticRegression
import spacy
import googletrans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
from sklearn import svm
import tensorflow as tf

from preprocessing.clean_data import get_data

# Hyperparameters

batch_size = 4489

# Data setup
train_data, validation_data, test_data = get_data(batch_size, from_file=False)

for input, label in train_data:
    x_train = input.numpy()
    y_train = label.numpy()

for input, label in validation_data:
    x_val = input.numpy()
    y_val = label.numpy()

# Model creation: Logistic Regression
print("Model: Logistic Regression")

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(x_train,y_train)

# Model Testing
y_val_pred = logreg.predict(x_val)

# Evaluation Metrics
print("Validation Accuracy:",metrics.accuracy_score(y_val, y_val_pred))
print("Validation Precision:",metrics.precision_score(y_val, y_val_pred))
print("Validation Recall:",metrics.recall_score(y_val, y_val_pred))
print("\n")
print("-"*100)

# Visualization
cnf_matrix = metrics.confusion_matrix(y_val, y_val_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Validation Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Model creation: Support Vector Machines
print("Model: Support Vector Machines")

clf = svm.SVC(kernel='linear') # Linear Kernel, can also try RBF, poly etc. (when I tried, these were worse but it depends on our actual data)
clf.fit(x_train, y_train)

# Model Testing
y_val_pred = clf.predict(x_val)

# Evaluation Metrics
print("Validation Accuracy:",metrics.accuracy_score(y_val, y_val_pred))
print("Validation Precision:",metrics.precision_score(y_val, y_val_pred))
print("Validation Recall:",metrics.recall_score(y_val, y_val_pred))
print("\n")
print("-"*100)

# Visualization
cnf_matrix = metrics.confusion_matrix(y_val, y_val_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Validation Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
