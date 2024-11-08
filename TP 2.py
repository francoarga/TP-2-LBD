# TP 2
# Vamo los pibes

#%% ------------------------------------------------------------------------------------------

# Importamos librerias

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from itertools import combinations, islice
import random

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
#%% ------------------------------------------------------------------------------------------
# 1a)
# Promedio de cada pixel
promedio_de_cada_pixel = pixel.mean() 
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(promedio_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

desviacion_de_cada_pixel = pixel.std(axis=0)
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(desviacion_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

#%% ------------------------------------------------------------------------------------------
# 1b)
labels = numeros.iloc[:, 1:]

# Filtro las imágenes de los dígitos 1 y 3
uno_y_tres = labels[(labels['labels'] == 1 ) | (labels['labels'] == 3)].iloc[:, 1:]  

promedio_de_cada_pixel = uno_y_tres.mean() 
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(promedio_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

desviacion_de_cada_pixel = uno_y_tres.std(axis=0)
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(desviacion_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 
#-----------------------------------------------------------------------------------------
# Filtro las imágenes de los dígitos 3 y 8
tres_y_ocho = labels[(labels['labels'] == 3 ) | (labels['labels'] == 8)].iloc[:, 1:]  

promedio_de_cada_pixel = tres_y_ocho.mean() 
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(promedio_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

desviacion_de_cada_pixel = tres_y_ocho.std(axis=0)
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(desviacion_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 


#%% ------------------------------------------------------------------------------------------
# 1c)

cero = labels[labels['labels'] == 0].iloc[:, 1:]  

promedio_de_cada_pixel = cero.mean() 
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(promedio_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

desviacion_de_cada_pixel = cero.std(axis=0)
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(desviacion_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 


#%% ------------------------------------------------------------------------------------------
# 2a)

# Dataframe con las imagenes correspondientes a ceros y unos
cero_o_uno = labels[(labels['labels'] == 0 ) | (labels['labels'] == 1)].iloc[:, :-1]
cero_o_uno.info()
# Hay 5980 imagenes, de las cuales la mitad son ceros y la otra mitad uno, asi que podriamos decir que esta balanceado

#%% ------------------------------------------------------------------------------------------

# 2b)

# Separo el dataframe en datos de train y test
X = cero_o_uno.drop(columns=['labels'])  
y = cero_o_uno['labels']  

# Dividir en conjuntos, x intensidades, y etiquetas, 80%  para train
# 20% para para test, respetando la proprcion de mitad y mitad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#%% ------------------------------------------------------------------------------------------

# 2c)

# Genero todas las combinaciones posibles de 3 atributos de X
todas_combinaciones_3_atributos = list(combinations(X.columns, 3))

# Selecciono 4 combinaciones aleatorias porque sino varia solo el ultimo 
combinaciones_3_atributos = random.sample(todas_combinaciones_3_atributos, 4)

# Evaluar cada combinación de atributos
for combinacion in combinaciones_3_atributos:
    # Seleccionar los atributos del conjunto de entrenamiento y prueba
    X_train_comb = X_train[list(combinacion)]
    X_test_comb = X_test[list(combinacion)]
    
    # Crear y entrenar el modelo KNN con k=3
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_comb, y_train)
    
    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test_comb)
    
    # Exactitud del modelo
    print("Exactitud del modelo:", metrics.accuracy_score(y_test, y_pred))
    
    # Matriz de confusión
    print(f"Matriz de confusión con atributos {combinacion}:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("-" * 50)

#%%
    
# Pruebo con 2 atributos

# Genero todas las combinaciones posibles de 2 atributos
todas_combinaciones_2_atributos = list(combinations(X.columns, 2))

# Selecciono 4 combinaciones aleatorias 
combinaciones_2_atributos = random.sample(todas_combinaciones_2_atributos, 4)

# Evaluar cada combinación de atributos
for combinacion in combinaciones_2_atributos:
    # Seleccionar los atributos del conjunto de entrenamiento y prueba
    X_train_comb = X_train[list(combinacion)]
    X_test_comb = X_test[list(combinacion)]
    
    # Crear y entrenar el modelo KNN con k=3
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_comb, y_train)
    
    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test_comb)
    
    # Exactitud del modelo
    print("Exactitud del modelo:", metrics.accuracy_score(y_test, y_pred))
    
    # Matriz de confusión
    print(f"Matriz de confusión con atributos {combinacion}:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Con mas de 3 atributos da error porque hay demasiada cantidad de combinaciones
# Hay mejor exactitud con 3 atributos
#%%
# Obtengo las primeras 4 combinaciones de 4 atributos, pero no varian casi nada.. sirve?
combinaciones_4_atributos = list(islice(combinations(X.columns, 4), 4))

for combinacion in combinaciones_4_atributos:
    # Seleccionar los atributos del conjunto de entrenamiento y prueba
    X_train_comb = X_train[list(combinacion)]
    X_test_comb = X_test[list(combinacion)]
    
    # Crear y entrenar el modelo KNN con k=3
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_comb, y_train)
    
    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test_comb)
    
    # Exactitud del modelo
    print("Exactitud del modelo:", metrics.accuracy_score(y_test, y_pred))
    
    # Matriz de confusión
    print(f"Matriz de confusión con atributos {combinacion}:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("-" * 50)

#%% ------------------------------------------------------------------------------------------
# 2d)

Nrep = 1
valores_k = range(1, 20)
valores_n = [3, 10, 50, 100]

# Inicialización de matrices para almacenar resultados de test y entrenamiento
resultados_test = np.zeros((len(valores_n), len(valores_k)))
resultados_train = np.zeros((len(valores_n), len(valores_k)))

for i, n_atributos in enumerate(valores_n):
    for rep in range(Nrep):
        # Selección de atributos
        columnas_aleatorias = X.sample(n=n_atributos, axis=1, random_state=rep)  
        X_selected = columnas_aleatorias.values
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=rep)
        for j, k in enumerate(valores_k):
            # Creación y ajuste del modelo KNN con k vecinos
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            Y_pred = model.predict(X_test)
            Y_pred_train = model.predict(X_train)
            
            # Exactitud
            acc_test = metrics.accuracy_score(y_test, Y_pred)
            acc_train = metrics.accuracy_score(y_train, Y_pred_train)
            
            # Almacena los resultados de exactitud
            resultados_test[i, j] += acc_test / Nrep  
            resultados_train[i, j] += acc_train / Nrep  
            
            # Matriz de confusión para el conjunto de prueba
            conf_matrix_test = confusion_matrix(y_test, Y_pred)
            print(f"Matriz de confusión para n_atributos={n_atributos}, k={k}, repetición {rep+1}:\n{conf_matrix_test}\n")

# Graficar promedios de exactitud para cada cantidad de atributos 
for i, n_atributos in enumerate(valores_n):
    plt.plot(valores_k, resultados_train[i, :], label=f'Train (n={n_atributos})')
    plt.plot(valores_k, resultados_test[i, :], label=f'Test (n={n_atributos})')

plt.legend()
plt.title('Exactitud del modelo de KNN')
plt.xlabel('Cantidad de vecinos (k)')
plt.ylabel('Exactitud (accuracy)')
plt.show()

#%%

#3) a)
# Separamos en desarrollo y validacion 
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,random_state=1,test_size=0.1)

#%%
#b)

alturas = [1,2,3,5,10]
nsplits = 5
kf = KFold(n_splits=nsplits)

resultados = np.zeros((nsplits, len(alturas)))
# una fila por cada fold, una columna por cada modelo

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        
        arbol = tree.DecisionTreeClassifier(max_depth = hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        cm = confusion_matrix(kf_y_test.values, pred)
        tp, fn, fp, tn = cm.ravel()
        score = (tp+tn) / (tp+tn+fp+fn)
        
        resultados[i, j] = score

# Mostrar los resultados de exactitud por cada combinación de pliegue y max_depth
for j, hmax in enumerate(alturas):
    print(f"Exactitud promedio para max_depth={hmax}: {np.mean(resultados[:, j]):.4f}")
#%%
#c)
#Definimos los hiperparámetros

max_depth_values = [1,2,3,5,7,10]
min_samples_split_values = [2, 5, 10]
min_samples_leaf_values = [1, 2, 4]

# Configuramos la validación cruzada
nsplits = 5
kf = KFold(n_splits= nsplits)

resultados = []

# Consultar como hacer el arbol, si con la combinacion de los hiperparamtros o distintos con cada uno
