import pandas as pd

previsores = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\entradas_breast.csv')
classe = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

x = 30
co = 16
y= 1

import keras
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
classificador.add(Dense(units=co, activation='relu', kernel_initializer='random_uniform', input_dim=x))
classificador.add(Dense(units=y, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)
previsoes = classificador.predict(previsores_teste)
print(previsoes)