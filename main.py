import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("spheres1d10.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Parámetros
num_particiones = 5
porcentaje_entrenamiento = 0.8

for i in range(num_particiones):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y,
                                                                            test_size=1 - porcentaje_entrenamiento,
                                                                            random_state=i)

    # Crear y entrenar el perceptrón
    weights = np.random.rand(X.shape[1])
    learning_rate = 0.1
    epochs = 10000

    for epoch in range(epochs):
        for j in range(len(X_entrenamiento)):
            prediction = np.dot(X_entrenamiento[j], weights)
            error = y_entrenamiento[j] - (prediction > 0).astype(int)
            weights += learning_rate * error * X_entrenamiento[j]

    # Realizar predicciones en el conjunto de prueba
    y_pred = (np.dot(X_prueba, weights) > 0).astype(int)

    # Calcular la precisión
    accuracy = accuracy_score(y_prueba, y_pred)

    # Imprimir la precisión de esta partición
    print(f"Partición {i + 1} - Precisión: {accuracy:.2f}")
