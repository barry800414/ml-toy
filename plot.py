
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import datasets
digits = datasets.load_digits()

i = 0

print(digits.target[i])
plt.imshow(digits.images[i], cmap=plt.get_cmap('gray'))
plt.show()