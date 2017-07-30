


# Loading tutorial dataset: digit
from sklearn import datasets
digits = datasets.load_digits()
print('digits:', digits.data.shape, digits.target.shape)

# split data into training data and testing data
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(digits.data, digits.target, test_size=0.2)

print('Training data:', train_X.shape, train_y.shape)
print('Testing data:', test_X.shape, test_y.shape)

# Training
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

print('Now we are going to train SVM model using gamma=0.001, C=100 ... ')
clf.fit(train_X, train_y)

# Testing
print('Now we are going to predict value by the trained model ...')
predict_y = clf.predict(test_X)
print('Prediction of final data: ', predict_y)
print('The real answer of final data: ', test_y)

from sklearn.metrics import accuracy_score
print('Overall accuracy: ', accuracy_score(test_y, predict_y))