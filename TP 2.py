# TP 2
# Vamo los pibes

#%% ------------------------------------------------------------------------------------------

# Importamos librerias
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd

#%% ------------------------------------------------------------------------------------------

# Importamos el dataset
carpeta = "~/Downloads/"
numeros = pd.read_csv(carpeta+'TMNIST_Data.csv')
#%% ------------------------------------------------------------------------------------------
# Analisis exploratorio

numeros.info()
# No hay nulls
numeros.dropna
#%% ------------------------------------------------------------------------------------------
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(pixel.iloc[1]).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 
# Tenemos 29900 filas con imagenes de numeros, 
# 784 conforman la parte que corresponde a los colores de la imagen, 
# la columna labels que indica de que numero se trata,
# Los atributos mas relevantes son los que generan la imagen, labels podria ser usado para para clasificar las filas por numerom, el nombre de
# la fuente podria ser descartada, y quiza algunos valores que conforman el fondo de la imagen, donde el numero no esta presente
#%% ------------------------------------------------------------------------------------------
# 1a)
# Promedio de cada pixel
promedio_de_cada_pixel = pixel.mean() 
# Grafico de barras para la intensidad promedio de cada pixel 
plt.bar(promedio_de_cada_pixel.index, promedio_de_cada_pixel.values)
plt.xlabel("Pixel")
plt.ylabel("Promedio")
plt.title("Promedio de intensidad de cada pixel")
# Cambiar los numerso que se muestran en el eje x 
plt.xticks(ticks=range(0, len(promedio_de_cada_pixel), 60), 
           labels=[f'{i}' for i in range(0, len(promedio_de_cada_pixel), 60)])
plt.show()


#%% ------------------------------------------------------------------------------------------
# 1b)
labels = numeros.iloc[:, 1:]
# Filtro las imágenes de los dígitos 1 y 3
digit_1_imagen = labels[labels['labels'] == 1].iloc[:, 1:]  
digit_3_imagen = labels[labels['labels'] == 3].iloc[:, 1:]

# Calculo la media de cada columna con un píxel, para cada dígito
# Media por columna
digit_1_media = digit_1_imagen.mean(axis=0)  
digit_3_media = digit_3_imagen.mean(axis=0)  

# Creo un histograma para comparar las medias de cada columna
plt.figure(figsize=(12, 6))
plt.bar(range(784), digit_1_media, alpha=0.5, label='Dígito 1', color='blue')
plt.bar(range(784), digit_3_media, alpha=0.5, label='Dígito 3', color='red')

plt.title('Comparación de medias de píxeles entre dígitos 1 y 3')
plt.xlabel('Columnas de pixeles')
plt.ylabel('Valor medio de pixeles')
plt.legend()
plt.grid()
# Cambie el rango que muestro en el grafico porque los dema no tenian datos relevantes
plt.xlim(87, 720)
ticks = np.arange(90, 720, 20) 
plt.xticks(ticks=ticks)
plt.show()

# Creo un grafico de dispersion para comparar la intensidad media en cada pixel
plt.figure(figsize=(10, 6))

# Para el dígito 1
plt.scatter(range(len(digit_1_media)), digit_1_media, color='blue', label='Dígito 1', alpha=0.6)

# Para el dígito 3
plt.scatter(range(len(digit_3_media)), digit_3_media, color='red', label='Dígito 3', alpha=0.6)

plt.xlabel("Píxel")
plt.ylabel("Media de intensidad")
plt.title("Scatterplot de medias de intensidad de píxeles para dígitos 1 y 3")
plt.legend()
plt.grid(True)
plt.show()

#-----------------------------------------------------------------------------------------
# Filtro las imágenes de los dígitos 3 y 8
digit_3_imagen = labels[labels['labels'] == 3].iloc[:, 1:]  
digit_8_imagen = labels[labels['labels'] == 8].iloc[:, 1:]

digit_3_media = digit_3_imagen.mean(axis=0)  
digit_8_media = digit_8_imagen.mean(axis=0)  

# Creo un histograma para comparar las medias de cada columna
plt.figure(figsize=(12, 6))
plt.bar(range(784), digit_3_media, alpha=0.5, label='Dígito 3', color='green')
plt.bar(range(784), digit_8_media, alpha=0.5, label='Dígito 8', color='red')

plt.title('Comparación de medias de pixeles entre digitos 3 y 8')
plt.xlabel('Columnas de pixeles')
plt.ylabel('Valor medio de pixeles')
plt.legend()
plt.grid()
plt.xlim(100, 720)
ticks = np.arange(100, 720, 20) 
plt.xticks(ticks=ticks)
plt.show()

# Creo un grafico de dispersion para comparar la intensidad media en cada pixel
plt.figure(figsize=(10, 6))

# Para el dígito 3
plt.scatter(range(len(digit_3_media)), digit_3_media, color='green', label='Dígito 3', alpha=0.6)

# Para el dígito 8
plt.scatter(range(len(digit_8_media)), digit_8_media, color='orange', label='Dígito 8', alpha=0.6)

plt.xlabel("Píxel")
plt.ylabel("Media de intensidad")
plt.title("Scatterplot de medias de intensidad de píxeles para dígitos 3 y 8")
plt.legend()
plt.grid(True)
plt.show()

#%% ------------------------------------------------------------------------------------------
# 1c)


