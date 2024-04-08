# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Create an LSTM-based Recurrent Neural Network model to predict stock prices.Using google stock price to predict the future value using RNN model.
![image](https://github.com/praveenst13/rnn-stock-price-prediction/assets/118787793/567317b1-b8f5-470b-96e2-6f666eb80b5c)



## Design Steps

### Step 1:
Gather historical stock price data for the stocks you want to predict

### Step 2:
Preprocess the data to make it suitable for training the RNN model
### Step 3:
 Determine which features are relevant for predicting stock prices.
### Step 4:
Fine-tune the model and hyperparameters based on performance on the validation set.



## Program
#### Name:Praveen s
#### Register Number:212222240077
```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
model = Sequential()
model.add(layers.SimpleRNN(77,input_shape=(60,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
print("Name:Praveen S \nRegister Number:212222240077    ")
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset.csv')
dataset_test.head()
test_set = dataset_test.iloc[:,1:2].values

test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []

for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name:Praveen S           Register Number:212222240077 ")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/praveenst13/rnn-stock-price-prediction/assets/118787793/75ac0fcd-e8e5-494b-8f88-96fbea809cc1)

### Mean Square Error

![image](https://github.com/praveenst13/rnn-stock-price-prediction/assets/118787793/9c12c91b-6935-41dd-8327-8c13002e2fc4)


## Result
Develope an LSTM model for stock price prediction was successful created.




