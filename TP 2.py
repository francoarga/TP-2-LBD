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
numeros = numeros.dropna
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

# Con este grafico visuaizamos la importancia de los 'pixeles'
# Grafico usando arboles de decision
labels = numeros.iloc[:, 1:]
rf = RandomForestClassifier(n_estimators=100)
rf.fit(labels.drop(columns='labels'), labels['labels'])
importances = rf.feature_importances_.reshape(28, 28)
plt.imshow(importances, cmap='hot')
plt.title('Importancia de cada pixel')
intervalo = 2  
plt.xticks(ticks=np.arange(0, 28, intervalo), labels=np.arange(0, 28, intervalo))
plt.yticks(ticks=np.arange(0, 28, intervalo), labels=np.arange(0, 28, intervalo))
plt.colorbar()
plt.show()
# Vemos en el grafico que ciertas columnas no tienen relevancia para distinguir las imagenes, de cero a 4 y de 24 a 28 puede ser descartado
# en lo que respecta al eje x y lo que es el eje y de cero a dos
rf = RandomForestClassifier(n_estimators=100)
rf.fit(labels.drop(columns='labels'), labels['labels'])
importances = rf.feature_importances_.reshape(28, 28)

# Recorto la parte de la imagen que me interesa
# Desde  5 a 22 en x, desde 3 a 27 en y
filtro = importances[3:28, 5:23]  

# Graficar las importancias de cada pixel
plt.imshow(filtro, cmap='hot')
plt.title('Importancia de cada pixel, desde 5 a 22 en x y desde 3 a 27 en y')
intervalo = 2  
plt.xticks(ticks=np.arange(0, filtro.shape[1], intervalo), 
           labels=np.arange(5, 23, intervalo))  
plt.yticks(ticks=np.arange(0, filtro.shape[0], intervalo), 
           labels=np.arange(3, 28, intervalo))  
plt.colorbar()
plt.show()
# Estos son los datos mas importantes para entrenar el modelo
#%% ------------------------------------------------------------------------------------------
# 1b)

# Filtrar las imágenes de los dígitos 1 y 3
digit_1_imagen = labels[labels['labels'] == 1].iloc[:, 1:]  
digit_3_imagen = labels[labels['labels'] == 3].iloc[:, 1:]

# Calcular la media de cada columna con un píxel, para cada dígito
# Media por columna
digit_1_media = digit_1_imagen.mean(axis=0)  
digit_3_media = digit_3_imagen.mean(axis=0)  

# Crear un histograma para comparar las medias de cada columna
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

# Filtrar las imágenes de los dígitos 3 y 8
digit_3_imagen = labels[labels['labels'] == 3].iloc[:, 1:]  
digit_8_imagen = labels[labels['labels'] == 8].iloc[:, 1:]

digit_1_media = digit_3_imagen.mean(axis=0)  
digit_3_media = digit_8_imagen.mean(axis=0)  

# Crear un histograma para comparar las medias de cada columna
plt.figure(figsize=(12, 6))
plt.bar(range(784), digit_1_media, alpha=0.5, label='Dígito 3', color='green')
plt.bar(range(784), digit_3_media, alpha=0.5, label='Dígito 8', color='red')

plt.title('Comparación de medias de pixeles entre digitos 3 y 8')
plt.xlabel('Columnas de pixeles')
plt.ylabel('Valor medio de pixeles')
plt.legend()
plt.grid()
plt.xlim(100, 720)
ticks = np.arange(100, 720, 20) 
plt.xticks(ticks=ticks)
plt.show()

# Hay numeros que son parecidos, ya que en media las intensidades de cada columna son similares, 
# en este caso, vemos en los graficos que 3 y 8 son mas parecidos que 1 y 3 porque hay mas superposicion 
