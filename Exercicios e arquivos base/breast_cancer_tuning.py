import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\entradas_breast.csv')
classe = pd.read_csv(r'C:\Users\Lobinho\OneDrive\Documentos\Rede Neural Artificial\saidas_breast.csv')

def criarRede(optimizer, loos, kernel_initializer, activation, nc1, nc2, nc3, nc4, drop):
    classificador = Sequential()
    classificador.add(Dense(units = nc1, activation = activation,
                        kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(drop))
    classificador.add(Dense(units = nc2, activation = activation,
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(drop))
    classificador.add(Dense(units=nc3, activation=activation,
                            kernel_initializer=kernel_initializer))
    classificador.add(Dropout(drop))
    classificador.add(Dense(units=nc4, activation=activation,
                            kernel_initializer=kernel_initializer))
    classificador.add(Dropout(drop))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loos,
                      metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [8, 10, 12],
              'epochs': [1000, 2000, 3000, 4000, 5000],
              'optimizer': ['adam', 'Adagrad', 'Adamax', 'Nadam'],
              'loos': ['binary_crossentropy', 'mean_squared_error'],
              'kernel_initializer': ['random_uniform', 'normal', 'zeros', 'ones', 'indentity'],
              'activation': ['relu', 'selu', 'elu'],
              'nc1': [32, 28, 24, 20, 16, 12, 8, 4],
              'nc2': [32, 28, 24, 20, 16, 12, 8, 4],
              'nc3': [32, 28, 24, 20, 16, 12, 8, 4],
              'nc4': [32, 28, 24, 20, 16, 12, 8, 4],
              'drop': [0.1, 0.15, 0.2, 0.25, 0.3]}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhor_precisao, melhores_parametros)