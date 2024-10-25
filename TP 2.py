# TP-2 Algarrobas
# Los pibes
#%% ------------------------------------------------------------------------------------------
# Importamos librerias
from sklearn.datasets import load_iris
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

# Informacion 
numeros.info()

# Miro los numeros de cada fila
# Para eso filtro las columnas que no forman parte de la imagen, las primeras dos
pixel = numeros.iloc[:, 2:]
pixel
# Mostrar la imagen del numero por fila, cambiando el iloc
# Plot imágen 
img = np.array(pixel.iloc[29899]).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 