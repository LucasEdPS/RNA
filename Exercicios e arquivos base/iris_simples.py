import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

base = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'elu', input_dim = 4))
classificador.add(Dropout(0.25))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 15,
                  epochs = 500)

classificador_json = classificador.to_json()
with open('classificador_isis.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_isis.h5')
import numpy as np
novo = np.array([[4.38, 2.56, 5, 1.5]])
previsao = classificador.predict(novo)

if (previsao[0, 0]>=0.5):
    print('Setosa')
if (previsao[0, 1]>=0.5):
    print('Virginica')
if (previsao[0, 2]>=0.5):
    print('Versicolor')