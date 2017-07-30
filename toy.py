

'''
Loading tutorial dataset: digit
'''
from sklearn import datasets
digits = datasets.load_digits()

# get training data
X = digits.data[:-1]
y = digits.target[:-1]

# print a sample image of X
print('Sample image (index:0) :', digits.images[0].shape)
print(digits.images[0])
print('X[0]:', X[0].shape)
print(X[0])

# print X, an n_samples x n_features 2D array
print('X:', X.shape)
print(X)

# print Y  an n_samples x 1 1D array
print('y:', y.shape)
print(y)


'''
Training procedure
'''
from sklearn import svm
classifier  = svm.SVC(gamma=0.001, C=100.)

print('Now we are going to train SVM model using gamma=0.001, C=100 ... ')
classifier.fit(X, y)


'''
Testing procedure
'''
print('Now we are going to predict value by the trained model ...')
print('Prediction of final data: ', classifier.predict(digits.data[-1:]))
print('The real answer of final data: ', digits.target[-1])
