# Creando una capa

## creando neuronas 

### Cada entrada tiene su propio peso
entradas = [1, 2, 3, 2.5]

### Los pesos son las conexiones entre las neuronas
pesos1 = [0.2, 0.8, -0.5, 1.0]
pesos2 = [0.5, -0.91, 0.26, -0.5]
pesos3 = [-0.26, -0.27, 0.17, 0.87]

### Cada neurona solo tiene un bias, 
bias1 = 2
bias2 = 3
bias3 = 0.5


### La cantidad de salidas es proporcional a la cantidad de pesos

'''
salidas = [entradas[0] * pesos1[0] + entradas[1] * pesos1[1] + entradas[2] * pesos1[2] + entradas[3] * pesos1[3] + bias1,
		entradas[0] * pesos2[0] + entradas[1] * pesos2[1] + entradas[2] * pesos2[2] + entradas[3] * pesos2[3] + bias2,
		entradas[0] * pesos3[0] + entradas[1] * pesos3[1] + entradas[2] * pesos3[2] + entradas[3] * pesos3[3] + bias3]

print(salidas)
'''
## Simplificacion del codigo

pesos = [[0.2, 0.8, -0.5, 1.0],
		[0.5, -0.91, 0.26, -0.5],
		[-0.26, -0.27, 0.17, 0.87]]


biases = [2, 3, 0.5]

'''

~~ Valor de entrada ~~

valor = -0.5

~~ PESO ~~

Multiplica la entrada para ajustar su importancia.
Se ajustan para minimizar el error en las predicciones de la red.

peso = 0.7

resultado_peso = peso * valor

# El resultado es -0.35

_______

~~ BIAS ~~

	Suma un valor extra para ajustar el resultado final.

bias = 0.7

resultado_bias = bias + valor

 	El resultado después de añadir el bias es 0.2



'''


### Inicializa una lista para almacenar las salidas de cada neurona en la capa
salidas_capa = []

### Itera a través de los pesos y los biases de cada neurona en la capa
for pesos_neuronales, biases_neuronales in zip(pesos, biases):

    salida_neuronal = 0
    
    # Calcula la salida de la neurona como la suma ponderada de las entradas
    for cantidad_entradas, peso in zip(entradas, pesos_neuronales):
        salida_neuronal += cantidad_entradas * peso

    salida_neuronal += biases_neuronales
    
    salidas_capa.append(salida_neuronal)


print("Funcion vieja")
print(salidas_capa)


## Formas y Numpy



### Shape es el tamaño de cada dimension

# (l de List)
l = [1, 2, 3, 4]

	# El shape de esta lista es (4,) porque solo tiene una dimension con cuatro elementos
	# Dentro de numpy es llamado 1D array y en matematicas es lo que conocemos como Vector

# (lol de List of Lists)
lol = [[1, 2, 4, 5,],
		[5, 4, 1, 3,]]

	# El shape de esta lista es (2,4) porque tiene 2 listas con cuatro elementos haciendo 2 dimensiones
	# Dentro de numpy es llamado 2D array y en matematicas es lo que conocemos como Matriz

# (lolol de List of Lists of Lists)

lolol =[[[1, 2, 4, 5,],
		[5, 4, 1, 3,]],

		[[10, 9, 42, 8,],
		[3, 44, 11, 30,]],

		[[11, 22, 43, 521,],
		[52, 41, 31, 353,]]]

	# El shape de esta lista es (3,2,4) porque tiene 3 listas con 2 listas con cuatro elementos haciendo 3 dimensiones
	# Dentro de numpy es llamado 3D array

import numpy as np 

### Producto punto para realizar el calculo de las salidas


'''
Es importante que siempre pongamos los pesos primeros en la funcion.
Esto se debe a que estamos calculando 3 neuronas y cada neurona tiene 3 pesos.
Y queremos que la indexacion se realice en base a esos sets de pesos.

Si tenemos 3 neuronas y cada una tiene 3 pesos, entonces pesos es una matriz de 3x3.

Dado que pesos es una matriz de 3x3 y entradas es un vector de 3 elementos, 
el resultado del producto punto es un vector de 3 elementos,
cada uno correspondiente a la salida antes de aplicar el bias.
'''

salidas_producto_punto = np.dot(pesos, entradas) + biases

print("\nFuncion Producto punto")
print(salidas_producto_punto)
print("\n")

## Batches

'''
Los batches son la cantidad de elementos que vamos a procesar por epochs
Se realizaran los calculos en paralelo.
'''

'''
Las entradas representan una entrada del sistema en un punto especifico
Si queremos ver nuestro aprendizaje si pasamos samples de manera individual veremos que la linea
siempre se va a mover, mientras mas grande sea nuestro batch size menos variacion va a haber en la linea de aprendizaje

Usualmente se trabaja hasta 64 maximo, mas de eso puede generar overfitting y hacer que el modelo se sobre entrene
'''

# Entradas antes
# entradas = [1, 2, 3, 2.5]

entradas = [[1, 2, 3, 2.5],
			[2.0, 5.0, -1.0, 2.0],
			[-1.5, 2.7, 3.3, -0.8],]

pesos = [[0.2, 0.8, -0.5, 1.0],
		[0.5, -0.91, 0.26, -0.5],
		[-0.26, -0.27, 0.17, 0.87]]


biases = [2, 3, 0.5]


'''
Ahora que ambos la entrada y los pesos son matrices debemos modificar nuestra funcion para realizar
un producto punto donde vamos a sacar una matriz de datos donde los valores son la multiplicacion 
de la filas de entredas y las columnas de pesos

'''

#salidas_producto_punto_2 = np.dot(pesos, entradas) + biases

#print("\nFuncion Producto punto")
#print(salidas_producto_punto_2)
#print("\n")



'''
Esto nos va a producir un error

ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)

Basicamente en nuestras entradas tenemos listas con 4 valores, en los pesos
tenemos listas con 4 valores.

Lo que necesitamos es arreglar los valores para que hayan 4 filas y 4 columnas,
actualmente tenemos 4 columnas y 3 filas.

Para resolver este problema tenemos que utilizar la transpuesta en una de las variables,
esto nos va a permitir tener 4 filas en entreda y 4 columnas en peso.
'''

salidas_producto_punto_matrices = np.dot(entradas, np.array(pesos).T) + biases

print("\nFuncion Producto punto")
print(salidas_producto_punto_matrices)
print("\n")



## Agregar otra capa

pesos_2 = [[0.1, -0.14, 0.5],
		[0.5, 0.12, -0.33],
		[-0.44, 0.73, -0.13]]

biases_2 = [-1, 2, -0.5]



## Conectando las capas

capa_1_salidas = np.dot(entradas, np.array(pesos).T) + biases 

# Las salidas de la capa 1 se vuelven las entradas de la capa 2
capa_2_salidas = np.dot(capa_1_salidas, np.array(pesos_2).T) + biases_2

#_____

# Objetificacion de los procesos

np.random.seed(0)

X = [[1, 2, 3, 2.5],
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]]

class Capa_Dense:
	def __init__(self, n_entradas, n_neuronas):

		# Las entradas son el tamaño de una sola muestra(sample), tenemos un batch de 3, cada muestra tiene 4

		# 4 = n_entradas, en este caso

		# n_neuronas es la cantidad de neuronas que queramos.

		# El 0.10 lo usamos para mantenernos en un rango adecuado

		''' 
		Cuando trabajamos los pesos trabajamos desde -1 a 1 porque arriba o abajo de eso los numeros
		crecen, 1.5, 5 o numeros mas grandes aumentan el tamaño de los valores y lo que queremos es que
		los valores queden chicos.
		Un buen punto de partida es usar "-0.1 - 0.1"
		'''

		# Si utilizamos el orden de entrada primero y luego neuronas, no tendremos que hacer una transpuesta

		self.pesos = 0.10 * np.random.randn(n_entradas, n_neuronas)


		'''
		Con el tema de los biases, usualmente se inicializa todo como 0 porque no genera, solo algunas veces
		donde las neuronas mandan un valor muy chico y termina siendo 0 y todo lo que multipliquemos con 0 va
		a dar 0, en esos casos se pueden inicializar con numeros que no son 0
		'''

		self.biases = np.zeros((1, n_neuronas))
		pass

	def adelante(self, entradas):
		self.salida = np.dot(entradas, self.pesos) + self.biases
		pass


print("# Ejemplo de pesos")
print (0.10 * np.random.randn(4, 3))


# Uso de la clase
capa1 = Capa_Dense(4,5)
# La capa 2 necesita las entradas de la capa 1, asi que las entradas son el tamaño de neuronas de la capa anterior.
capa2 = Capa_Dense(5,5)

capa1.adelante(X)
capa2.adelante(capa1.salida)


