#importing tensorflow
import tensorflow as tf
import keras
import numpy as np
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix

#importing pandas
import pandas as pd

#loading dataset
df = pd.read_csv("gesture.csv", header=None)
df_describe=df.describe()


#checking dataset for missing values
print(df.isnull().sum())

#extracting required columns and storing the dependent and independent values in separate variables
X_data = df.iloc[:,:64].values
Y_data = df.iloc[:,64].values



#splitting training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33)
N, D= x_train.shape




#applying Standard scaling to data
from sklearn.preprocessing import StandardScaler
ac = StandardScaler()
x_train = ac.fit_transform(x_train)
x_test = ac.fit_transform(x_test)



#converting to numpy array
x_train= np.array(x_train, dtype=np.float)
x_test= np.array(x_test, dtype=np.float)

y_train= np.array(y_train, dtype=np.float)
y_test= np.array(y_test, dtype=np.float)




#creating callback
def step_decay(epoch):
  initial_learning_rate=0.01
  decay_rate=1
  lrate=initial_learning_rate * (1/(1+(decay_rate * epoch)))
  return lrate

lrate=tf.keras.callbacks.LearningRateScheduler(step_decay)



#creating NN model
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128,input_shape=(D,),activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(4,activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=100,callbacks=(lrate))

print("Train score: ",model.evaluate(x_train,y_train))
print("Test score: ",model.evaluate(x_test,y_test))

#converting predicted values to 1D array
y_pred_3D=model.predict(x_test)
y_pred=np.array([])
for row in y_pred_3D:
    y_pred=np.append(y_pred,np.argmax(row))

#confusion matrix
matrix = confusion_matrix(y_test,y_pred)
print('Confusion matrix : \n',matrix)

#plot what is returned by model fit
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()



#plot accuracy
plt.plot(r.history['accuracy'],label='accuracy')
plt.plot(r.history['val_accuracy'],label='val_accuracy')
plt.legend()

