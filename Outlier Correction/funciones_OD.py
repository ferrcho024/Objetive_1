# FUNCIONES AUXILIARES
import pandas as pd
import numpy as np
import random as rd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import *
import math as math


################## HBOS ##############################

def calcula_k(df,variables,redondeo):
    # Calcula la cantidad de intervalos estáticos o  dinámicos y la cantidad de valores en cada intervalo.
    # df --> DataFrame
    # variables --> Variables para la estrategia HBOS
    # redondeo --> Cantidad de unidades decimales para el redondeo del valor resultante
    
    # Halla la variable que tiene el mayor valor para usarlo como variable princial.
    mayor = 0
    for var in variables:
        var = var + '_dif'
        if df[var].max() > mayor:
            mayor = df[var].max()
            sensor = var
    
    # Calculo de intervalos k (bins) por el método dinámico.
    valores = df[sensor].tolist()  # Saca una lista todos lo valores del sensor
    valores.sort()
    
    kd = int(np.sqrt(len(df[sensor])))
    N = len(df[sensor]) # -> Cantidad de elementos
    unicos = list(dict.fromkeys(valores))

    if kd > len(unicos):
        kd = len(unicos)

    #Cantidad de valores por intervalo 
    n = int(N/kd)
    
    ks = int(1 + 3.322 * np.log10(len(df[sensor]))) # Calculo para k por regla de Sturges
    
    rango = valores.pop() - valores[0]
    tam_bin = rango/ks
    
    bins = []
    for i in range(ks):
#        if round(((i+1)*tam_bin+valores[0]),redondeo) <= df[sensor].max():
            bins.append(round(((i+1)*tam_bin+valores[0]),redondeo))

    return ks,kd,n,tam_bin,bins


def asigna_kd(n,valores,k):
    # Asigna los valores a cada intervalo calculado de forma dinámica
    
    intervalos = []
    
    # Si el ultimo valor del intervalo n es igual al primer valor del intervalo n+1, se van pasando todos los 
    # valores iguales al intervalo n
    
    i = 0
    while i <= len(valores):
        if i + n > len(valores):
            pos = len(valores)+1
        else:
            pos = i + n
            while ((pos+1) < len(valores)) and (valores[pos] == valores[pos+1]):
                #print("Mientras",valores[pos],"==",valores[pos+1],"y",pos+1,"sea menor o igual a:",len(valores))
                pos += 1
    
        intervalo = valores[i:pos+1]
        intervalos.append(intervalo)
        i = pos + 1
        
    
    return intervalos

def asigna_ks(valores,bins):
    # Asigna los valores a cada intervalo calculado de forma estática

    valores1 = valores.tolist()
    valores1.sort()
    
    intervalos = {}
    pos = 0
    while len(valores1) > 0:
        i = 0
        while (i < len(valores1)) and valores1[i] <= bins[pos]:
            i +=1
        
        intervalos[bins[pos]] = valores1[0:i]
        del(valores1[0:i])
        pos += 1
    
    return intervalos


def calcula_ponderaciones(intervalos, redondeo):
    # Ponderación por frecuencia ---  como lo haría yo... La idea es normalizar a 1 el intervalo o valor con mayor 
    # frecuencia, y luego se calculan las demás ponderaciones tomando como referencia el de mayor frecuencia.
    ponderaciones = {}
    mayor_len = 0
    mayor_hist = 0
    l_min = 0

    # Se cuentan la cantidad de valores de cada intervalo para calcular las ponderaciones (normalización a 1). Esto se hace para 
    # la cantidad de valores en todo el data set y parala cantidad de valores en cada bin del histograma
    for u in intervalos.keys():
        ponderaciones[u] = [len(intervalos[u])]
        if mayor_len < len(intervalos[u]):
            mayor_len = len(intervalos[u])
    
        l_max = u
        hist = len(intervalos[u])/((l_max - l_min)+1)
        ponderaciones[u].append(round(hist,redondeo))
        l_min = l_max
        if mayor_hist < hist:
            mayor_hist = hist

    # Se calculan las ponderaciones de acuerdo con las cantidades calculadas tomando como referencia el mayor valor encontrado.    
    for valor in ponderaciones.keys():
        pon = ponderaciones[valor][0]/mayor_len
        ponderaciones[valor].append(round(pon,redondeo))

        pon = ponderaciones[valor][1]/mayor_hist
        ponderaciones[valor].append(round(pon,redondeo))

    
        
    #bins = list(map(float,ponderaciones.keys())) # Pasa a una lista los key del diccionario y los convierte a float
    
    return ponderaciones


def calcula_HBOS(ponderaciones,HBOS):
    # Cálculo del HBOS para cada bin
    k = 0
    
    for i in ponderaciones.keys():
        if k not in list(map(float,HBOS.keys())): # Pasa a una lista los key del diccionario y los convierte a float
            HBOS[k] = 0
        
        if ponderaciones[i][2] == 0:
            HBOS[k] = HBOS[k] + 10
        else: 
            HBOS[k] = HBOS[k] + np.log10(1/ponderaciones[i][2]) # HBOS con base en las cantidades del dataframe
#            HBOS[k] = HBOS[k] + np.log10(1/ponderaciones[i][3]) # HBOS con base en las cantidades de cada bin
        
        k += 1
        
    return HBOS

def verifica_HBOS(HBOS,valor,bins):
    # Verifica el score HBOS de un valor dado.

    #keys = list(map(float,HBOS.keys())) # Pasa a una lista los key del diccionario y los convierte a float
    if valor >= max(bins):
        pos = len(bins)-1
    else:
        pos,val = min(enumerate(bins), key=lambda x: x[1]<=valor) # Busca el valor mayor más cercano al valor de referencia
    
    return HBOS[pos]
    
def diferencias (df,variables,redondeo):
    # Calcula la diferencia entre dos valores contiguos i y i-1 y devuelver el valor positivo
    # df -> Dataframe
    # variables -> Lista de variables
    # redondeo -> Entero del número de decimales. 

    for var in variables:
        shape = df.shape
        df.insert(shape[1]-2, var+'_dif', 0, allow_duplicates=False) # Agrega la columna para el valor de la diferencia
        for i in range (1,len(df)):
            #print(varriables)
            #print("Valor de i ", i)
            
            df.loc[i,var+'_dif']= round(abs(df[var][i] - df[var][i-1]),redondeo)
    
    return df

def diferencias_pos (df,variables,redondeo):
    # Calcula la diferencia entre dos valores contiguos i y i-1
    # df -> Dataframe
    # variables -> Lista de variables
    # redondeo -> Entero del número de decimales. 

    for var in variables:
        shape = df.shape
        df.insert(shape[1]-2, var+'_dif', 0, allow_duplicates=False) # Agrega la columna para el valor de la diferencia
        for i in range (1,len(df)):
            #print(varriables)
            #print("Valor de i ", i)
            
            df.loc[i,var+'_dif']= round((df[var][i] - df[var][i-1]),redondeo)
    
    return df

def synthetic_data(datos,porcentaje,memoria,margen='auto'):
    # Agrega datos sintéticos a un dataframe. Se aumenta o disminuye el valor del dato original en 100 de acuerdo con la 
    # tendencia de los últimos datos.
    # datos ->  Dataframe con los datos originales. En esta primera versión solo funciona parala variable pm25, por esta razón, el dataframe debe tener los datos de esta variable.
    # porcentaje -> Porcentaje normalizado a 1 (0 - 1), de datos sintéticos a agregar
    # memoria -> Margen desde el cual quiere agregar los datos sntéticos. Este margen sirve para salvaguardar los primeros datos originales del dataset en caso de alguna operación que sea necesaria. 
    # margen -> Cantidad a agregar o quitar la valor original. 'auto' es la opción por defecto que agrega o quita 100 al dato original.

    
    if porcentaje > 1:
        print("Digite el valor del porcentaje de 0 a 1, siendo 1 el 100%")
        sys.exit()
    
    outliers = []
    datos2 = datos.copy()
    datos2=datos2.assign(pm25_outlier="N")
    for i in range (round(len(datos)*porcentaje)):
        pos = rd.randint(memoria+1, len(datos)-1)
        while pos in outliers:
            pos = rd.randint(memoria+1, len(datos)-1)

        outliers.append(pos)
        index = pos #datos.index.tolist()[pos]
        #cambio = rd.randint(50, 20)
        if datos.pm25[pos] < 100:
            #valor = (datos.pm25[pos] + cambio)
            valor = (datos.pm25[pos] + 100)
        else:
            #valor = (datos.pm25[pos] - cambio)
            valor = (datos.pm25[pos] - 100)

        datos2.loc[index,("pm25_outlier")] = datos2.pm25[pos]

        if margen == 'auto':
            datos2.loc[index,("pm25")] = valor
        else:
            datos2.loc[index,("pm25")] = datos.pm25[pos] + margen

    outliers.sort()
    print("Se incluyeron", len(outliers), "Outliers")
    return datos, datos2, outliers


def df_mix (df,porcentaje):
    # Crea dos nuevos dataframe a partir de un dataframe. El df_1 contiene las fechas en orden aleatorio
    # de acuerdo con el porcentaje dado. El df_2 contiene las fechas restantes
    # df -> Dataframe original
    # porcentaje -> float del porcentaje normalizado a 1

    df_mixed_1 = pd.DataFrame()
    df_mixed_2 = pd.DataFrame()
    nodos = df.codigoSerial.unique().tolist()

    for n in nodos:
        fild = df.loc[df.loc[:,"codigoSerial"] == n]
    
        dates = fild.fecha.unique().tolist()
    
        fechas = []
        for d in range (round(len(dates)*porcentaje)):
            pos = rd.randint(0, len(dates)-1)
            while dates[pos] in fechas:
                pos = rd.randint(0, len(dates)-1)

            fechas.append(dates[pos])
    
        for f in fechas:
            fil = fild.loc[fild.loc[:,"fecha"] == f]
            df_mixed_1 = pd.concat([df_mixed_1,fil],ignore_index=True)
        
        diff = list(set(dates) - set(fechas))
        for di in diff:
            fil = fild.loc[fild.loc[:,"fecha"] == di]
            df_mixed_2 = pd.concat([df_mixed_2,fil],ignore_index=True)

    
    return df_mixed_1, df_mixed_2


def matrix_conf (df,tecnica):

# MATRIZ DE CONFUSIÓN (Confusion Matrix)

# 0: No es outlier
# 1: Es Outlier
#
#                        Real
#                    |  0  |  1  |
#     Predicho   | 0 |  TN |  FN |
#                | 1 |  FP |  TP |
#
# TN: True Negative - No outlier identificado como No outlier
# FN: False Negative - Outlier identificado como No outlier
# FP: False Positive - No outlier identificado como Ourlier
# TP: True Positivo - Outlier identificado como Outlier

    
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    
    #matriz_pm25_media = np.zeros((2, 2))

    for i in range (len(df)):
        if (df.pm25_outlier[i] == "N") and (df[tecnica][i] == "N"):
            TN += 1

        if (df.pm25_outlier[i] != "N") and (df[tecnica][i] == "N"):
            FN += 1

        if (df.pm25_outlier[i] == "N") and (df[tecnica][i] == "S"):
            FP += 1

        if (df.pm25_outlier[i] != "N") and (df[tecnica][i] == "S"):
            TP += 1
  
    print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN)
    print('Total values:',TP+TN+FP+FN)
    print('*************************************')
    
    # PRECISIÓN
    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP/(TP+FP)
    print("Precision:",precision)

    # EXHAUSTIVIDAD
    if (TP + FN) == 0:
        exhaustividad = 0
    else:
        exhaustividad = TP/(TP+FN)
    print("Recall:",exhaustividad)

    # F1
    if (precision + exhaustividad) == 0:
        F1 = 0
    else:
        F1 = 2*((precision*exhaustividad)/(precision+exhaustividad))
    print("F1:",F1)

    # F1
    # if (2*TP + FP + FN) == 0:
    #     F1 = 0
    # else:
    #     F1 = 2*TP/(2*TP + FP + FN)
    # print("F1:",F1)

    # EXACTITUD
    if (TP+TN+FP+FN) == 0:
        exactitud = 0
    else:
        exactitud = (TP+TN)/(TP+TN+FP+FN)
    print("Accuracy:",exactitud)
    print("")

    return TN, FN, FP, TP

def confu_matrix (real,predicho):
    # MATRIZ DE CONFUSIÓN (Confusion Matrix)

    # 0: No es outlier
    # 1: Es Outlier
    #
    #                         Real
    #                    |  0  |  1  |
    #     Predicho   | 0 |  TN |  FN |
    #                | 1 |  FP |  TP |
    #
    # TN: True Negative - No outlier identificado como No outlier
    # FN: False Negative - Outlier identificado como No outlier
    # FP: False Positive - No outlier identificado como Ourlier
    # TP: True Positivo - Outlier identificado como Outlier

    # real -> Lista de datos originales
    # predicho -> Lista de datos predichos

    # Confusion Matrix
    cm = confusion_matrix(real, predicho)

    # Accuracy
    acc = accuracy_score(real, predicho)

    # Recall
    re = recall_score(real, predicho, average=None)

    # Precision
    pre = precision_score(real, predicho, average=None)

    #F1
    f1 = f1_score(real, predicho, average=None)

    #print("Precision:",pre[1])
    #print("Recall:",re[1])
    #print("F1:",f1[1])
    #print("Accuracy:",acc)
    #print("")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.matshow(cm)
    plt.set_cmap('Blues')
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Real', fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    for (i,j), z in np.ndenumerate(cm):
        if j == 0 and i == 0:
            ax.text(j,i,'{:0.0f}'.format(z), ha='center', va='center', color='white', fontsize='xx-large')
        else:
            ax.text(j,i,'{:0.0f}'.format(z), ha='center', va='center', fontsize='xx-large')
    
    plt.savefig("CM.eps", dpi=200, bbox_inches='tight')

    return acc, re[1], pre[1], f1[1]

def confu_matrix_no_print (real,predicho):
    # MATRIZ DE CONFUSIÓN (Confusion Matrix)

    # 0: No es outlier
    # 1: Es Outlier
    #
    #                         Real
    #                    |  0  |  1  |
    #     Predicho   | 0 |  TN |  FN |
    #                | 1 |  FP |  TP |
    #
    # TN: True Negative - No outlier identificado como No outlier
    # FN: False Negative - Outlier identificado como No outlier
    # FP: False Positive - No outlier identificado como Ourlier
    # TP: True Positivo - Outlier identificado como Outlier

    # real -> Lista de datos originales
    # predicho -> Lista de datos predichos

    # Confusion Matrix
    cm = confusion_matrix(real, predicho)

    # Accuracy
    acc = accuracy_score(real, predicho)

    # Recall
    re = recall_score(real, predicho, average=None)

    # Precision
    pre = precision_score(real, predicho, average=None)

    #F1
    f1 = f1_score(real, predicho, average=None)

    return acc, re[1], pre[1], f1[1]


# Fórmula de Haversine para calcular las dictancias
def haversine(lon1, lat1, lon2, lat2):
    #lon1 = Longitud punto 1
    #lat1 = Latitud punto 1
    #lon2 = Longitud punto 2
    #lat2 = Latitud punto 2
    
    # Radio de la tierra
    R = 6378  
    
    #Convertir grados decimales en radianes
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    #Formula
    dlon = lon2 - lon1 #Distancia entre longitudes
    dlat = lat2 - lat1 #Distancia entre latitudes
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2 
    c = 2 * math.asin(math.sqrt(a))
    return c * R