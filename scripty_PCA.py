import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from nolearn.dbn import DBN
from sklearn.metrics import classification_report, accuracy_score

print "reading in data"
# Reading in the dataset
df = pd.read_csv('train.csv', nrows=28000)
Y = df['label'].values
X = df.drop('label', axis=1).values
x_test = pd.read_csv('test.csv')

min_max_scaler = MinMaxScaler() # Create the MinMax object.
X = min_max_scaler.fit_transform(X.astype(float)) # Scale pixel intensities only.
x_test = min_max_scaler.fit_transform(x_test.astype(float)) # Scale pixel intensities only.

# Using Neural Net
print "creating model"
model = DBN([X.shape[1], 700, 10],
                learn_rates = 0.1,
                learn_rate_decays = 0.9,
                epochs = 100,
                dropouts = 0.20,
                verbose = 0)

print "doing PCA"
pca = PCA()
X_transformed = pca.fit_transform(X)

print "fitting data"
model.fit(X_transformed, Y)

x_test_transformed = pca.transform(x_test)

print "predicting test"
predicted = model.predict(x_test_transformed)

print "writing results"
submission = DataFrame(predicted, columns=['Label'], 
                       index=np.arange(1, 28001))
submission.index.names = ['ImageId']

submission.to_csv('submission.csv')

print "Done!"