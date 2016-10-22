import numpy as np
import pandas as pd
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

min_max_scaler = MinMaxScaler() # Create the MinMax object.
X = min_max_scaler.fit_transform(X.astype(float)) # Scale pixel intensities only.

x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                        test_size = 0.2, random_state = 0) # Split training/test.

# Using Neural Net
print "creating model"
model = DBN([x_train.shape[1], 700, 10],
                learn_rates = 0.1,
                learn_rate_decays = 0.9,
                epochs = 100,
                dropouts = 0.20,
                verbose = 0)

print "doing PCA"
pca = PCA()
X_transformed = pca.fit_transform(x_train)
model.fit(X_transformed, y_train)

x_test_transformed = pca.transform(x_test)

#predicted = model.predict(x_test_transformed)

print "fitting data"
#model.fit(x_train, y_train)

print "predicting test"
#predicted = model.predict(x_test)

y_true, y_pred = y_test, model.predict(x_test_transformed) # Get our predictions
print(classification_report(y_true, y_pred)) # Classification on each digit
print 'The accuracy is:', accuracy_score(y_true, y_pred)


