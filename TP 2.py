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
carpeta = "C:/Users/franc/Documents/Facultad/Laboratorio de datos/TP - 2/"
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

# Con este grafico visuaizamos la importancia de los 'pixeles'
# Grafico usando arboles de decision
labels = numeros.iloc[:, 1:]
rf = RandomForestClassifier(n_estimators=100)
rf.fit(labels.drop(columns='labels'), labels['labels'])
importances = rf.feature_importances_.reshape(28, 28)
plt.imshow(importances, cmap='hot')
plt.title('Importancia de cada pixel')
interval = 2  
plt.xticks(ticks=np.arange(0, 28, interval), labels=np.arange(0, 28, interval))
plt.yticks(ticks=np.arange(0, 28, interval), labels=np.arange(0, 28, interval))
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
plt.title('Importancia de cada pixel (desde 5 a 22 en x y desde 3 a 27 en y)')
interval = 2  
plt.xticks(ticks=np.arange(0, filtro.shape[1], interval), 
           labels=np.arange(5, 23, interval))  
plt.yticks(ticks=np.arange(0, filtro.shape[0], interval), 
           labels=np.arange(3, 28, interval))  
plt.colorbar()
plt.show()
# Estos son los datos mas importantes para entrenar el modelo
#%% ------------------------------------------------------------------------------------------
# 1b)
