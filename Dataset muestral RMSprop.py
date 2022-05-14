 #https://www.tutorialspoint.com/keras/keras_installation.htm
#https://docs.python.org/es/3/tutorial/venv.html
import datetime

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD

dataset = pd.read_csv("creditcard.csv", delimiter= ",")
#dataset.shape
print(dataset.shape)


#Creamos un dataframe con solo una muestra del dataset completo, el cual tiene un porcentaje entre fraudes y no fraudes del 24.6%
fraudes = dataset[dataset["Class"]==1]
nofraudes = dataset[dataset["Class"]==0]
nofraudes1 = nofraudes.sample(n=2000)
dataset1 = pd.concat([fraudes,nofraudes1])
print(dataset1.shape)

x_data = dataset1[["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]]
y_data = dataset1[["Class"]]
print(x_data.shape)
print(y_data.shape)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
print(y_train.shape)
print(x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes=2
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

#para cargar la red:
#modelo_cargado = tf.keras.models.load_model('Prueba 1.h5')

model = Sequential()
model.add(Dense(812, activation='relu', input_shape=(30,)))
#model.add(Dropout(0.2))
model.add(Dense(812, activation='relu'))
model.add(Dense(812, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

log_dir="Graph/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/

history = model.fit(x_train, y_trainc,
                    batch_size=300,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_test, y_testc),
                    callbacks= [tbCallBack])

score = model.evaluate(x_test, y_testc, verbose=1)
a=model.predict(x_test)
#b=model.predict_proba(x_testv)
print(a.shape)
print(a[1])
#Para guardar el modelo en disco
model.save("Prueba 1.h5")

