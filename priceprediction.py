import numpy as np
import pandas as pd
data = pd.read_csv("boston.csv")
data.head()
data.columns
data.head(n=10)
data.shape
data.isnull().sum()
data.describe()
data.describe()

import seaborn as sns
sns.histplot(data["Price"])

sns.boxplot(data["Price"])
from sklearn.preprocessing import StandardScaler
# Split the data into input and output variables


X = data.drop('Price', axis=1)
y =data['Price']
# Scale the input features 
scaler =StandardScaler()
X =scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
# Splitthe data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)


from keras.models import Sequential 
from keras.layers import Dense, Dropout


# Define the model architecture
model = Sequential()
model.add(Dense(128,activation = 'relu',input_dim =14))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(1))

print(model.summary())



model.compile(optimizer = 'adam',loss ='mean_squared_error',metrics=['mae'])

x_val=X_train
y_val=y_train



history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))



results = model.evaluate(X_test, y_test)
