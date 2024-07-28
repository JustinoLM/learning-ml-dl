import numpy as np

# Fija la semilla del generador de números aleatorios para asegurar la reproducibilidad.
np.random.seed(0)

# Función para crear un dataset.
# `puntos` es el número de puntos por clase.
# `clases` es el número de clases.
def crear_dataset(puntos, clases):
    # Inicializa las matrices X e y con ceros.
    # X tendrá dimensiones (puntos * clases, 2) para almacenar las coordenadas (x, y).
    # y tendrá dimensiones (puntos * clases) para almacenar las etiquetas de clase.
    X = np.zeros((puntos * clases, 2))
    y = np.zeros(puntos * clases, dtype='uint8')
    
    # Itera sobre el número de clases.
    for numero_clase in range(clases):
        # Define el rango de índices para la clase actual.
        ix = range(puntos * numero_clase, puntos * (numero_clase + 1))
        
        # Crea un array de radios que varían linealmente de 0 a 1.
        r = np.linspace(0.0, 1, puntos)
        
        # Crea un array de ángulos que varían linealmente más una componente aleatoria.
        t = np.linspace(numero_clase * 4, (numero_clase + 1) * 4, puntos) + np.random.randn(puntos) * 0.2
        
        # Asigna las coordenadas (x, y) usando las fórmulas de transformación polar a cartesiano.
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        
        # Asigna las etiquetas de clase.
        y[ix] = numero_clase
    
    return X, y

import matplotlib.pyplot as plt

'''# Imprime un mensaje para indicar que se está ejecutando el código.
print('Funciona')

# Crea un dataset con 100 puntos por clase y 3 clases.
X, y = crear_dataset(100, 3)

# Muestra un gráfico de dispersión de los puntos sin colorear.
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Muestra un gráfico de dispersión de los puntos coloreados por clase.
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
'''