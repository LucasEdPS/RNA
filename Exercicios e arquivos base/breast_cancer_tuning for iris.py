import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

base = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criarRede(optimizer,drop, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units=neurons, activation=activation, input_dim=4))
    classificador.add(Dropout(drop))
    classificador.add(Dense(units=3, activation='softmax'))
    classificador.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 15, 25],
              'epochs': [100, 250, 500],
              'optimizer': ['adam', 'Adamax'],
              'drop': [0.2, 0.25, 0.3],
              'activation': ['relu', 'elu'],
              'neurons': [4, 6, 8]}
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           cv=10)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhores_parametros, melhor_precisao)

#{'activation': 'elu', 'batch_size': 15, 'drop': 0.25, 'epochs': 500, 'neurons': 8, 'optimizer': 'adam'}
# 0.9733333349227905