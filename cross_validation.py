
# loading dataset
from sklearn import datasets
digits = datasets.load_digits()

# doing cross-validation
from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(gamma=0.001, C=100.)
scores = cross_val_score(clf, digits.data, digits.target, cv=5)

print(scores)
