import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\entradas_breast.csv')
classe = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal', input_dim=30))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation='elu',
                            kernel_initializer='normal'))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=1, activation='sigmoid'))
    classificador.compile(optimizer='Adamax', loss='mean_squared_error',
                      metrics=['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criarRede,
                                epochs=2000,
                                batch_size=10)
resultados = cross_val_score(estimator=classificador,
                             X=previsores, y=classe,
                             cv=10, scoring='accuracy')
media = resultados.mean()
desvio = resultados.std()
print(media, desvio)