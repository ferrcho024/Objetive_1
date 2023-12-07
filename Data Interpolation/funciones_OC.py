# FUNCIONES PARA LA CORRECCIÓN DE OUTLIERS

import pandas as pd
import numpy as np
import random
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import datetime
import random
import math
from scipy import stats

# distributions = [
#     norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
#     gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
#     laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
#     halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
#     mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, 
#     semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
#     exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
#     ]

from scipy.stats import (
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
        gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
        laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
        halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
        mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, 
        semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
        exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist, chi, chi2, f
    )

def corregir_ten(datos, variable, index, dif):
# Función que corrige el valor identificado como outlier. Lo que se hace es sacar el promedio de las últimas 
# diferecnias (valor de variable time_step) y sumarlo al valor anterior al outlier. Ese será el nuevo valor.
# datos --> Datafrae con los datos correctos y los datos a corregir.
# variable --> nombre de la columna en la que están los datos en el Dataframe
# index --> Posición (fila) del valor que se va a corregir
# dif --> Lista de las diferencias de los últimos "time-step" valores
 
    if (datos.variable[index] - datos.variable[index-1]) >= 0:
        valor = datos.variable[index-1] + abs(np.mean(dif))
    else:
        valor = variable[index-1] - abs(np.mean(dif))
    
    if valor < 0:
        valor = 0
    
    return round(valor,4)

def corregir_ponderado(datos, outliers_detected):
# Función que corrige el valor identificado como outlier. Se asigna una ponderación a los últimos 5 y a los posteriores 5 valores 
# tomando como referencia el valor identificado como outliers. Dentro de la selección de los 5 se omiten valores que hayan sido 
# corregidos o que se ahayan indentificado como outlier para evitar calcular datos nuevos con base en datos artificiales.
# Devuelve un dataframe con los datos identificados como outlier corregidos.
# datos --> Datafrae con los datos correctos y los datos a corregir.
# outliers_detected --> Diccionario de index de los valores identificados como outliers por sensor de cada nodo.

    mem = 5
    datos_corregidos = pd.DataFrame()
    probs = [0.26, 0.13, 0.065, 0.03, 0.015]
    for nube in outliers_detected.keys():
        datos_nube = datos.loc[datos.loc[:,'codigoSerial'] == nube]
        for sensor in ['pm25_df','pm25_nova']:
            datos_nube[sensor+'_corr'] = datos_nube[sensor]
            datos_nube.reset_index(inplace=True, drop=True)
            posiciones = outliers_detected[nube][sensor]
            datos_nube.loc[posiciones, 'out_'+sensor] = 'O'
            if len(posiciones) > 10:
                for index in posiciones:
                    val_ant = []
                    val_pos = []
                    i = 1

                    if index > mem or index < (len(datos_nube)- mem):

                        while len(val_ant) < mem and len(val_pos) < mem:
                            if index > mem and ((index-i) not in posiciones or len(val_ant) < mem):
                                val_ant.append(datos_nube.loc[(index-i), sensor])
                            else:
                                val_ant = val_pos
                            
                            if index < (len(datos_nube)- mem) and ((index+i) not in posiciones or len(val_pos) < mem):
                                val_pos.append(datos_nube.loc[(index+i), sensor])
                            else: 
                                val_pos = val_ant
                            
                            i += 1
                    
                    ant = np.multiply(val_ant,probs[:len(val_ant)])
                    pos = np.multiply(val_pos,probs[:len(val_pos)])

                    val = sum(ant) + sum(pos)
                    datos_nube.loc[index, sensor+'_corr'] = val

        datos_corregidos = pd.concat([datos_corregidos,datos_nube],ignore_index=True)

    return datos_corregidos

def estacionalidad(datos, variable):
# Define las ventanas de estacionalidad de la señal, mediante el cálculo de los estadísticos
# Develve un Dataframe con los estádisticos por hora por día por sensor, de acuerdo con la variable indicada.
# datos -> Dataframe con los datos
# variable -> Variable a la cuál se le calcularán los estadísticos (df o nova)

    df = pd.DataFrame()

    for i in range(24):
        hora = str(i)+':'
        estadisticos = datos.loc[datos.loc[:,"hora"].str.contains(r'^'+hora, regex=True)]
        estadisticos['time'] = i
        df = pd.concat([df,estadisticos],ignore_index=True)   

    estadisticos = pd.DataFrame(columns=['codigoSerial', 'fecha', 'hora', 'media', 'des_std', 'var', 'min', '1/4', '2/4', '3/4', 'max'])

    for nube in df.codigoSerial.unique().tolist():
        temp = df.loc[df.loc[:,'codigoSerial'] == nube]
        for dia in temp.fecha.unique().tolist():
            temp1 = temp.loc[temp.loc[:,'fecha'] == dia]
            for i in range (24):
                temp2 = temp1.loc[temp1.loc[:,'time'] == i]
                # estadisticos=estadisticos.append({'codigoSerial': nube,
                #                 'fecha': dia, 
                #                  'hora': i, 
                #                  'media': temp2[variable].mean(), 
                #                  'des_std': temp2[variable].std(ddof=0), 
                #                  'var': temp2[variable].var(ddof=0),
                #                  'min': temp2[variable].min(),
                #                  '1/4': temp2[variable].quantile(0.25), 
                #                  '2/4': temp2[variable].quantile(0.5), 
                #                  '3/4': temp2[variable].quantile(0.75), 
                #                  'max': temp2[variable].max()})] , ignore_index=True)

                # estadisticos=pd.concat([estadisticos, pd.DataFrame([nube, dia, i, temp2[variable].mean(), 
                #                  temp2[variable].std(ddof=0), temp2[variable].var(ddof=0), temp2[variable].min(),
                #                  temp2[variable].quantile(0.25), temp2[variable].quantile(0.5), 
                #                  temp2[variable].quantile(0.75), temp2[variable].max()])] , ignore_index=True)
                estadisticos.loc[len(estadisticos)] = [nube, dia, i, temp2[variable].mean(), 
                                 temp2[variable].std(ddof=0), temp2[variable].var(ddof=0), temp2[variable].min(),
                                 temp2[variable].quantile(0.25), temp2[variable].quantile(0.5), 
                                 temp2[variable].quantile(0.75), temp2[variable].max()]

    
    del nube, temp, temp1, temp2, dia
    return df, estadisticos


def est_consol(datos, ref):
# Consolida los estadísticos por día, de acuerdo con una columna de referencia.
# Devuelve un Dataframe con los estadísticos consolidados por hora 
# datos -> Los datos estadísticos generales.
# ref -> Columna que se utilizará como referencia para el consolidado.

    est_consol = pd.DataFrame(columns=['hora', 'media', 'des_std', 'var', 'min', '1/4', '2/4', '3/4', 'max'])

    for h in datos.hora.unique().tolist():
        temp = datos.loc[datos.loc[:,'hora'] == h]
        est_consol=est_consol.append({'hora': h, 
                                 'media': temp[ref].mean(), 
                                 'des_std': temp[ref].std(ddof=0), 
                                 'var': temp[ref].var(ddof=0),
                                 'min': temp[ref].min(),
                                 '1/4': temp[ref].quantile(0.25), 
                                 '2/4': temp[ref].quantile(0.5), 
                                 '3/4': temp[ref].quantile(0.75), 
                                 'max': temp[ref].max()} , ignore_index=True)
    
    del temp, h
    return est_consol

def auto_outliers(datos, porcentaje, columnas):
# Genera índices de outliers de forma aleatoria, de acuerdo con un porcentaje indicado de los datos totales.
# Devuelve un diccionario con los números de los nodos como índice y con listas de ínidces para cada nodo. 
# datos --> Dataframe con los datos.
# porcentaje --> porcentaje de los datos que seran identificados como outliers.
# columnas --> Lista de los nombres de las columnas sobre las que se referenciarán los outliers

    outliers_generated = {}
    for nube in datos.codigoSerial.unique().tolist():
        val = {}
        cant = len(datos.loc[datos.loc[:,'codigoSerial'] == nube])

        for col in columnas:
            val[col] = random_numbers = random.sample(range(cant), round(cant*porcentaje))

        outliers_generated[nube] = val
    
    return outliers_generated

def corregir_densidad(datos, outlier, variable, dist, debug='N'):
    # Basado en: https://towardsdatascience.com/how-to-find-probability-from-probability-density-plots-7c392b218bab
    
    df = datos.copy()
    fecha = df[outlier:outlier+1].index

    filtro = df[df.index.hour == fecha.hour[0]][variable]
    datos = filtro[filtro.notnull()].values

    mejor = dist.loc[fecha.hour[0], ]
    funct = mejor.iloc[-1]

    # # Cross-Validation del parámetro bandwidth
    # grid = GridSearchCV(KernelDensity(),
    #                     {'bandwidth': np.linspace(0.1, 1.0, 30)},
    #                     cv=20) # 20-fold cross-validation
    # grid.fit(datos[:, None])
    # bw = grid.best_estimator_.bandwidth

    #kd_lunch = funct.fit(np.sort(datos)[:, np.newaxis])
    kd_lunch = funct.fit(datos[:, None])

    #inter, df = intervalo(df, outlier,variable)
    cont = 1
    while pd.isna(df.iat[outlier-cont, df.columns.get_loc(variable)]):
        cont +=1
    inf = df.iat[outlier-cont, df.columns.get_loc(variable)]
    df.iat[outlier-cont, df.columns.get_loc(variable)] = np.NaN
    cont = 1
    while pd.isna(df.iat[outlier+cont, df.columns.get_loc(variable)]):
        cont +=1
    sup = df.iat[outlier+cont, df.columns.get_loc(variable)]
    df.iat[outlier+cont, df.columns.get_loc(variable)] = np.NaN
    inter = (inf,sup)

    probable = min(inter)-10
    while not ((probable >= min(inter)) and  (probable <= max(inter)+1)):
        probable = float(kd_lunch.sample())

    # ban = True
    # while ban:  
    #     cont = 1
    #     while pd.isna(df.iat[outlier-cont, df.columns.get_loc(variable)]):
    #         cont +=1
    #     inf = df.iat[outlier-cont, df.columns.get_loc(variable)]
    #     df.iat[outlier-cont, df.columns.get_loc(variable)] = np.NaN
    #     cont = 1
    #     while pd.isna(df.iat[outlier+cont, df.columns.get_loc(variable)]):
    #         cont +=1
    #     sup = df.iat[outlier+cont, df.columns.get_loc(variable)]
    #     df.iat[outlier+cont, df.columns.get_loc(variable)] = np.NaN

    #     #prob = 0.001

    #     prob = pd.DataFrame()
    #     prob['valores'] = list(range(int(min(inf, sup)), int(max(inf, sup)+1)))

    #     prob['prob'] = prob['valores'].apply(probabilidad, args=[kd_lunch])

    #     if prob['prob'].max() > 0.001:
    #         probable = prob['valores'][prob.idxmax()['prob']]
    #         ban = False

        
        # for n in range(int(min(inf, sup)), int(max(inf, sup)+1)):
        #     start = n
        #     end = n+1
        #     N = 100

        #     step = (end - start) / (N - 1) 
            
        #     x = np.linspace(start, end, N)[:, np.newaxis]

        #     kd_vals = np.exp(kd_lunch.score_samples(x)) 
        #     probability = np.sum(kd_vals * step)
            
        #     if probability > prob:
        #         prob = probability
        #         probable = n
        #         ban = False
    
    #probable += random.uniform(0, 1)
    #probable += random.random()
    #probable += float(kd_lunch.sample())%1  # Valor decimal aleatorio usando la función de probabilidad de los datos
    #print(outlier,'----',probable,'----',prob)

    #print('imputacion despues de ',cont,'veces: ',probable)

    if debug == 'S':
        filtro.plot(kind='density', figsize=(10,4))
        plt.show()
        filtro.plot(kind='box', figsize=(10,4))
        plt.show()

    return round(probable,4)

def corregir_probabilidad(datos, outlier, variable, dist):
    # Basado en: https://towardsdatascience.com/how-to-find-probability-from-probability-density-plots-7c392b218bab
    
    df = datos.copy()
    fecha = df[outlier:outlier+1].index

    mejor = dist.loc[fecha.hour[0], ]
    #print(dist)
    funct = mejor.iloc[-1] # Selecciono la mejor distribución
    #print('hora:', fecha.hour[0])
    params = mejor.iloc[2:-1]
    params = [p for p in params if ~np.isnan(p)]

    inter = intervalo(df, outlier, variable)
    # print('--------->',inter)

    #probable = min(inter)-10
    cont = 1
    ban = True
    #print('Hora:',fecha.hour[0], 'Rango:',inter, 'Función:',mejor.iloc[0], 'Parámetros:',params)
    while ban:
        probables = np.array(funct.rvs(*params, size=10))
        #probables = funct.rvs(*params, size=10)
        #print(probables)
        try:
            probable = probables[np.where((probables >= min(inter)) & (probables <= max(inter)+1))[0][0]]
            ban = False
        except:
            cont += 1
            if (cont%1000) == 0:
                inter = intervalo(df, outlier, variable)
                #print('*****','Hora:',fecha.hour[0], 'Rango:',inter, 'Función:',mejor.iloc[0], 'Parámetros:',params)
             
    #print(cont)

    # while not ((probable >= min(inter)) and  (probable <= max(inter)+1)):
    #     probable = float(funct.rvs(*params, size=1))
    #     print(probable)

    #     cont += 1
    
    #print(probable, 'despues de', cont, 'intentos')

    return round(probable,4)

def probabilidad(n, kd_lunch):
    
    start = n
    end = n+1
    N = 100

    step = (end - start) / (N - 1) 
    
    x = np.linspace(start, end, N)[:, np.newaxis]

    kd_vals = np.exp(kd_lunch.score_samples(x)) 
    probability = np.sum(kd_vals * step)
    
    return probability

def intervalo(df, outlier, variable):

    cont = 1
    while pd.isna(df.iat[outlier-cont, df.columns.get_loc(variable)]):
        cont +=1
    inf = df.iat[outlier-cont, df.columns.get_loc(variable)]
    df.iat[outlier-cont, df.columns.get_loc(variable)] = np.NaN
    cont = 1
    while pd.isna(df.iat[outlier+cont, df.columns.get_loc(variable)]):
        cont +=1
    sup = df.iat[outlier+cont, df.columns.get_loc(variable)]
    df.iat[outlier+cont, df.columns.get_loc(variable)] = np.NaN

    return (inf,sup)

# KS test for goodness of fit
def kstest(data, distname, paramtup):
    ksN = len(data)           # Kolmogorov-Smirnov KS test for goodness of fit: samples
    ks = stats.kstest(data, distname, paramtup, ksN)[1]   # return p-value
    return ks             # return p-value

# distribution fitter and call to KS test
def fitdist(data, dist):    
    fitted = dist.fit(data)
    ks = kstest(data, dist.name, fitted)
    res = (dist.name, ks, *fitted)
    return res

def check_function_scipy(filtro):
    # Fuente: https://towardsdatascience.com/probability-distributions-with-pythons-scipy-3da89bf60565

    # distributions = [
    #     norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
    #     gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
    #     laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
    #     halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
    #     mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, 
    #     semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
    #     exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist, chi, chi2, f
    #     ]
    
    
    # distributions = [
    #     norm, expon, gamma, logistic, lognorm, uniform, powernorm, weibull_max, 
    #      weibull_min, laplace, alpha, invgauss, invgamma, loglaplace, loggamma, 
    #      maxwell, pareto, invweibull, cosine, t
    #      ]
    
    # Distribuciones en el dominio de los valores positivos
    distributions = [
          norm, lognorm, gamma, expon, rayleigh, pareto, weibull_max, weibull_min, beta, 
          uniform, chi2, invgamma, invgauss, f, levy, loglaplace, nakagami, rice, halfnorm,
        ]
    
    data = filtro[filtro.notnull()].values

    # call fitting function for all distributions in list
    res = [fitdist(data,D) for D in distributions]
    #print(res)

    # convert the fitted list of tuples to dataframe
    pd.options.display.float_format = '{:,.3f}'.format
    df = pd.DataFrame(res, columns=["distribution", "KS p-value", "param1", "param2", "param3", "param4"])
    df["distobj"] = distributions
    df.sort_values(by=["KS p-value"], inplace=True, ascending=False)
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)
    
    return df.iloc[0]

def check_function_sklearn(filtro):
    # Basado en: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    # Bassado en: https://www.cienciadedatos.net/documentos/pystats02-kernel-density-estimation-kde-python.html

    # Cross-Validation de los parámetros kernel y bandwidth
    data = filtro[filtro.notnull()].values
    bandwidths = 10 ** np.linspace(-2, 1, 20)
    #bandwidths = np.linspace(inf, sup, 20)

    mlh = float('-inf')
    kernels = ["gaussian", "tophat"]#, "epanechnikov", "exponential", "linear", "cosine"]
    #kernels = ["gaussian", "exponential"]
    # ban = True
    # for i in range(1):
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': bandwidths,
                        'kernel': kernels},
                        cv=20) # 20-fold cross-validation
    grid.fit(data[:, None])
        #print(grid.best_estimator_)

        # if (grid.best_score_ > mlh):
        #     mlh = grid.best_score_
        #     pos = list(bandwidths).index(grid.best_params_['bandwidth'])
        #     if pos==0:
        #         inf, sup= bandwidths[pos], bandwidths[pos+1]
        #     elif pos == len(bandwidths)-1:
        #         inf, sup = bandwidths[pos-1], bandwidths[pos]
        #     else:
        #         inf, sup = bandwidths[pos-1], bandwidths[pos+1]

        #     bandwidths = 10 ** np.linspace(np.log10(inf), np.log10(sup), 20)
        #     #bandwidths = np.linspace(inf, sup, 20)
        #     bandwidths = np.append(bandwidths, grid.best_params_['bandwidth'])
        #     bandwidths = np.sort(bandwidths)

        # else:
        #     i +=3
        #     ban = False

    return {'distribution':grid.best_estimator_.kernel,
                        'bandwidth':grid.best_estimator_.bandwidth,
                        'mlh':mlh,
                        'distobj':grid.best_estimator_}






