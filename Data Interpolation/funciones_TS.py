import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.special import inv_boxcox
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import datetime
from IPython.display import display

def var_divi(datos, N):
# Función que divide el dataset y calcula la varianza de cada parte
# Imprime un gráfico boxplot de las varianzas de las partes
# Imprime una lista con las varianzas de cada parte y la deviación de estas varianzas
# datos --> Array con los datos
# N --> Número de partes en las que se dividirán los datos
    
    cantidad = len(datos)//N
    varianzas = []

    for i in range(N):
        parte = datos[i*cantidad:(i+1)*cantidad]
        varianzas.append(parte.var())

    print('\nVarianza general',datos.var(),'\nVarianzas.',varianzas, '\nDesviación de las varianzas', np.std(varianzas))
    plt.figure(figsize=(6, 3), dpi=100)
    plt.boxplot(varianzas)
    plt.show()


def flinger_test(datos, N, debug='N'):
# Función que divide el dataset y compara las varianzas de las partes utilizando el test de flinger-killeen
# Imprime el resultado del test de flinger indicando si se rechaza o no H0
# Devuelve True si se acepta H0 o False si se rechaza H0
# datos --> Array con los datos
# N --> Número de partes en las que se dividirán los datos

# TEST DE FLINGER-KILLEEN
# H0: Las varianzas son estadísticamente iguales con un intervalo de confianza del 95%.
# Si el valor-p está por encima del valor crítico de 0.05 (para un 95% de confiabilidad), entonces no se rechaza H0

    cantidad = len(datos)//N
    com_fligner = 'stats.fligner(parte0'

    for i in range(N):
        parte = datos[i*cantidad:(i+1)*cantidad]
        globals()["parte" + str(i)] = parte.tolist()
        if i!=0:
            com_fligner = com_fligner + "," + "parte" + str(i)
    com_fligner = com_fligner + ")"

    fligner_test = eval(com_fligner)
    if debug == 'Y':
        print('\n**** Test de Flinger-Killeen ****')
        print('H0: Las varianzas son iguales')
        print('El estadístico es:', fligner_test.statistic)
        print('El p-value es:', fligner_test.pvalue)

    if fligner_test.pvalue > 0.05:
        flinger = True
        if debug == 'Y':
            print("NO se rechaza H0, la serie es estacionaria en varianza")
    else:
        flinger = False
        if debug == 'Y':
            print("Se rechaza H0, la serie NO es estacionaria en varianza")
    
    return flinger

def dickey_aum_test(datos, debug='N'):
# Función que verifica estacionalidad en media utilizando el test de Dickey-Fuller Aumentada
# Imprime el resultado del test de Dickey-Fuller Aumentada indicando si se rechaza o no H0
# Devuelve True si se acepta H0 o False si se rechaza H0
# datos --> Array con los datos

# Dickey Fuller Aumentada
# H0: Existe raiz unitaria en la serie
# Si el valor-p está por encima del valor crítico de 0.05 (para un 95% de confiabilidad), entonces no se rechaza que 
# haya una raiz unitaria
# If the p-value is above a critical size, then we cannot reject that there is a unit root.

    adf = stattools.adfuller(datos, autolag='AIC')

    if debug == 'Y':
        print('\n**** Test de Dickey Fuller Aumentada ****')
        print('H0: Existe raiz unitaria en la serie')
        print('El estadístico es:', adf[0])
        print('El p-value es:', adf[1])
        print('El número de lag es:', adf[2])
        print('Los valores críticos son:', adf[4])
    if adf[1] < 0.05:
        adfr = True
        if debug == 'Y':
            print("De acuerdo con el p-value, Se rechaza H0, la serie es estacionaria en media")
    else:
        adfr = False
        if debug == 'Y':
            print("De acuerdo con el p-value, No se rechaza H0, la serie NO es estacionaria en media")
    
    return adfr

def med_divi(datos, N):
# Función que divide el dataset y calcula la media de cada parte
# Imprime un gráfico boxplot de las varianzas de las partes
# Imprime una lista con las medias de cada parte y la deviación de estas medias
# datos --> Array con los datos
# N --> Número de partes en las que se dividirán los datos

    cantidad = len(datos)//N
    medias = []

    for i in range(N):
        parte = datos[i*cantidad:(i+1)*cantidad]
        medias.append(parte.mean())

    print('\nMedia general',datos.mean(),'\nMedias',medias, '\nDesviación de las medias', np.std(medias))
    plt.figure(figsize=(6, 3), dpi=100)
    plt.boxplot(medias)
    plt.show()

def comp_graf(datosTest, sensor, nsensor):
# Realiza el la comparación de gráficos antes y después de modificar datos
# datosTest --> Dataframe con los datos anteriores y los nuevos datos
# sensor --> nombre de la columna con los datos anteriores
# nsensor --> nombre de la columna con los datos modificados
# Imprime los gráficos de comparación

    fig = plt.figure(figsize=(12, 3), dpi=100)
    ax1=fig.add_subplot(121)
    ax1.plot(datosTest[sensor])
    ax1.axhline(y=datosTest[sensor].mean(), color='r', linestyle='-')
    ax1.tick_params(rotation=90)
    ax1.set_title(sensor)

    ax2=fig.add_subplot(122)
    ax2.plot(datosTest[nsensor])
    ax2.axhline(y=datosTest[nsensor].mean(), color='r', linestyle='-')
    ax2.tick_params(rotation=90)
    ax2.set_title(nsensor)
    plt.show()

    
def arima_model(datos, feature, N, debug='N'):
# Calcula los valores de los factores p, d y q del modelo ARIMA
# Devuelve los valores de p, d y q, además de los valores del modelo
# datos --> Dataset con los datos a analizar, DEBE SER UN DATA SET O APLICA PARA LISTA O ARRAY
# feature --> Nombre de la columna del dataset al cual se quiere realizar el análisis
# N --> Número de partes en las que se quiere dividir el dataset para las comparaciones de media y varianza
# debug --> Se imprimen gráficos. Por fecto está en "N" (NO). Asignar "Y" para activar

    # Inicialización del orden para el modelo ARIMA
    p = d = q = 0   # AR - I - AM
    
    sensores = [feature]
    sensor = nsensor = sensores[-1]
    datosTest = datos.copy()
    datosTest = datosTest[datosTest[nsensor].notnull()] # Los valores nulos afectan el cálculo del lambda para la transformación
    
    
    # Verificación de estacionariedad en varianza (homocedasticidad) del dataset dividido en N partes
    #print('\nDatos:', nsensor)
    flinger = flinger_test(datosTest[nsensor], N, debug=debug)

    lmbda = np.NaN
    if not flinger: # Si no se cumple homocedasticidad, se realiza transformación de datos 
        sensores.append(nsensor+'_tranf')
        if len(sensores) > 2:
            sensor = sensores[-2]

        # Transformación logaritmica de los datos usando la transformación de boxcox
        xt,lmbda = stats.boxcox(datosTest[nsensor])
        nsensor = sensores[-1]
        datosTest[nsensor] = xt
        if debug == "Y":
            print('*** Se realizó transformación logarítmica de los datos para la estacionaridad en varianza ***')

    # Se imprimen gráficos de los datos si está activado el 
    if debug == "Y":
        comp_graf(datosTest, sensor, nsensor)
    
    # Verificación de estacionariedad en media mediante la verificación de existencia de raiz unitaria
    #print('\nDatos:', nsensor)
    dickey = dickey_aum_test(datosTest[nsensor], debug=debug)
    while not dickey:
        d += 1
        sensores.append(nsensor+'_diff')

        nsensor = sensores[-1]
        if len(sensores) > 2:
            sensor = sensores[-2]

        datosTest[nsensor] = datosTest[sensor] - datosTest[sensor].shift(1)
        datosTest = datosTest.dropna(subset=[nsensor]) # Para eliminar la fila con NaN solo en la columna indicada
        
        dickey = dickey_aum_test(datosTest[nsensor], debug=debug)

    lags = 30
    if debug == "Y":
        comp_graf(datosTest, sensor, nsensor)

        plt.rc("figure", figsize=(12,3), dpi=100)
        plot_acf(datosTest[nsensor], lags=lags)
        plt.title('ACF '+nsensor)
        plt.show()

        plt.rc("figure", figsize=(12,3), dpi=100)
        plot_pacf(datosTest[nsensor], lags=lags, method='ywm')
        plt.title('PACF '+nsensor)
        plt.show()

    acf, acf_conf = stattools.acf(datosTest[nsensor], nlags=lags, alpha=0.05)
    pacf, pacf_conf = stattools.pacf(datosTest[nsensor], nlags=lags, method='ywm', alpha=0.05)
    banda = round(2/np.sqrt(len(datosTest[nsensor])),4)

    for i in pacf_conf[0:]:
        if i[0] < 0:
            break
        p += 1

    for i in acf_conf[0:]:
        if i[0] < 0:
            break
        q += 1

    if p == (lags-1):
        p = 0
    
    if q == (lags-1):
        q = 0

    if debug == "Y":
        print('\nDatos:', nsensor)
        print('AR -->',p)
        print('I  -->',d)
        print('MA -->',q)
    
    return p, d, q, (datosTest[nsensor], lmbda)

def arima_fit(datos, p, d, q, debug='N'):
# Evalua el modelo, calcula el rmse y muestra el gráfico de los últimos 50 datos entre los datos reales 
# y los predichos por el modelo
# datos --> Son los datos del modelo (transformados, difetenciado o como haya resultado)
# p, d, q --> Son los valores de los parámetros para AR(p), I(d) y MA(q)
# debug --> Se desactiva si no se quiere imprimir la imagen
    
    df = pd.DataFrame(datos[0])
    df.columns = ['datos']

    a = len(df)-50
    b = len(df)
    #print(len(df))
    
    order = (p, d, q)
    try:
        modelo1 = sm.tsa.ARIMA(df['datos'], order=order)
        modelo2 = sm.tsa.ARIMA(df['datos']*(-1), order=order)
        resultados1 = modelo1.fit()
        resultados2 = modelo2.fit()
    except:
        #print('No se pudo evaluar el modelo', order)
        return np.NaN, np.NaN, '' 

    df['ARIMA'] = resultados1.fittedvalues.shift(-1)  
    df['ARIMA_inv'] = resultados2.fittedvalues.shift(-1)
    mse1 = ((df['ARIMA'] - df['datos']) ** 2).mean()
    mse2 = ((df['ARIMA_inv'] - df['datos']) ** 2).mean()

    if mse1 < mse2:
        label = 'ARIMA'
        mse = mse1
        resultados = resultados1
        inv=''
    else:
        label = 'ARIMA_inv'
        mse = mse2
        resultados = resultados2
        inv = '*'

    if debug == 'Y':
        print('The Root Mean Squared Error of',label,order,'trend is', format(round(np.sqrt(mse), 2)))
        plt.figure(figsize=(10,5))
        plt.plot(df['datos'][a:b], label='Originales', marker="o")
        plt.plot(df[label][a:b], label=label+str(order), marker="o")
        plt.legend(loc='best')
        plt.show()

    return resultados, round(np.sqrt(mse), 2), inv

def elige_mejor(datos, p, d, q):
# Evalua todos los posibles modelos y muestra cuáles fueron los dos mejores
# Imprime los dos mejores modelos de acuerdo con rmse calculado
# datos --> Datos del modelo a evaluar
# p, d, q --> Son los valores de los parámetros para AR(p), I(d) y MA(q)
    
    error = pd.DataFrame(columns=['Modelo', 'RMSE'])
    res = ''
    min = 9999

    for i in range(p+1):
        for j in range(d+1):
            for k in range(q+1):
                try:
                    resultados, rmse, inv = arima_fit(datos, i, j, k, debug='N')
                    if rmse < min:
                        min = rmse
                        res = resultados
                except:
                    continue
                
                lista1 = {'Modelo':str((i, j, k))+inv, 'RMSE': rmse}
                error = error.append(lista1, ignore_index=True)

    error = error.sort_values('RMSE')
    #display(error.head(1))
    #print(error.head(5).to_markdown())

    return res, error

def correccion(datos, datos_modelo, outliers_detected, nube, resultados):
# Corrige los outliers detectados usando el mejor modelo ARIMA para los datos
# Entrega el Dataframe con los datos originales y los datos corregidos
# datos --> Datos generales
# datos_modelo --> Datos con los que se generó el mejor modelo
# datos_nube --> Datos filtrados del nodo
# outliers_detected --> Diccionario con los índices de los outliers detectados por cada nodo
# nube --> Número de el nodo que se está trabajando
# variable --> Nombre de la variable que se está corrigiendo (pm25_df o pm25_nova)
# resultados --> resultados del modelo arima ajustado con los que se realiza pa predicción

    ''' Hay que optimizar el código, sobre todo en el uso de las variables'''
    
    datos_nube = datos[datos['codigoSerial'] == nube]
    df = datos.copy()
    lmbda_ori = datos_modelo[1]
    datos_modelo = creaDatosModelo(datos_nube, datos_modelo) # Para tener un modelo de datos por minuto
    
    name = datos_modelo.name
    modificaciones = name.split('_')
    variable = 'pm25_'+modificaciones[1]
    outliers = outliers_detected[nube][variable]
    dfp = pd.DataFrame(datos_modelo)
    #display(dfp.head(15))
    dfp = dfp.rename(columns={name:modificaciones[-1]})
    
    if 'tranf' in modificaciones: 
        xt = stats.boxcox(datos_nube[variable][datos_nube[variable].notnull()], lmbda=lmbda_ori)
        xt = xt[modificaciones.count('diff'):]
        dfp['tranf'] = xt

    else:
        dfp['tranf'] = datos_nube[variable]

    cont = 0
    for outlier in outliers:
        cont += 1
        fecha = fechaCheck(datos_nube, dfp, outlier, 0, 1)#datos_nube[outlier:outlier+1].index[0]
       
        if outlier <= 60:
            fechaRound = fechaCheck(datos_nube, dfp, outlier+(62-outlier), 0, 1).ceil('H')#fecha.round('60min')
        else:
            fechaRound = fecha.ceil('H')#fecha.round('60min')     
        
        #display((fecha, fechaRound))
        # if outlier == 778:
        #     display((dfp.loc[fecha, modificaciones[-1]], round(resultados.predict(start=fechaRound, end=fechaRound,dynamic=True)[0],4)))
        dfp.at[fecha, modificaciones[-1]] = min(dfp.loc[fecha, modificaciones[-1]], abs(round(resultados.predict(start=fechaRound, end=fechaRound,dynamic=True)[0],4)))
        #dfp.at[fecha, modificaciones[-1]] = round(resultados.predict(start=fechaRound, end=fechaRound,dynamic=True)[0],4)
        # if outlier == 778:
        #     display(dfp.loc[fecha, modificaciones[-1]])
        if 'diff' in modificaciones:
            fechaBack = fechaCheck(datos_nube, dfp, outlier, 1, 0)
            if fechaBack:
                dfp.at[fecha, 'tranf'] = max(0,dfp.loc[fechaBack, 'tranf'] + dfp.loc[fecha, modificaciones[-1]])
            else:
                dfp.at[fecha, 'tranf'] = dfp.loc[fecha, modificaciones[-1]]
        else:
            dfp.at[fecha, 'tranf'] = dfp.loc[fecha, modificaciones[-1]]
        # if outlier == 778:
        #     display(dfp.loc[fecha,:])
    
    inicio = fechaCheck(datos_nube, dfp, outliers[0], 0, 1)
    fin = fechaCheck(datos_nube, dfp, outliers[-1], 0, 3)
    periodo = dfp[inicio:fin].copy()
    
    sensor = 'Pred_'+modificaciones[1]#variable.split('_')[1]
    if 'tranf' in modificaciones:     
        periodo[sensor] = inv_boxcox(periodo['tranf'], lmbda_ori)
    else:
        periodo[sensor] = periodo['tranf']
    #display(periodo.head(15), lmbda)
    periodo.drop([modificaciones[-1], 'tranf'], axis=1, inplace=True)

    periodo['codigoSerial'] = nube
    periodo = periodo.groupby(['codigoSerial', 'fechaHora', ]).mean()

    df = df.groupby(['codigoSerial', 'fechaHora', ]).mean()
    if sensor not in df.columns:
        df[sensor] = df[variable]
    
    df.update(periodo, overwrite=True)
    df.reset_index(level='codigoSerial', inplace=True)
    print("Outliers Corrected:",cont)
    return df


def fechaCheck(datos, modelo, number, atras, adelante):

    fecha = datos[number-atras:number+adelante].index[0]
    fechaOk = False
    cont = 0
    while not fechaOk:
        try:
            modelo.loc[fecha, modelo.columns[0]]
            fechaOk = True
        except KeyError:
            cont += 1
            number -=1
            fecha = datos[number-atras:number+adelante].index[0]
        
        if cont > 10:
            return False
    
    return fecha



def creaDatosModelo(datos, datos_modelo):

    name = datos_modelo[0].name
    modificaciones = name.split('_')
    lmbda_ori = datos_modelo[1]

    variable = 'pm25_'+modificaciones[1]

    datos_modelo = pd.DataFrame(datos[variable][datos[variable].notnull()])#datos[[variable]].notnull()
    for v in modificaciones[2:]:
        if v == 'tranf':
            xt, lmbda = stats.boxcox(datos_modelo[variable])
            datos_modelo[variable+'_tranf'] = xt
            datos_modelo.drop([variable], axis=1, inplace=True)
            variable = variable+'_tranf'
        
        if v == 'diff':
            datos_modelo[variable+'_diff'] = datos_modelo[variable] - datos_modelo[variable].shift(1)          
            datos_modelo = datos_modelo.dropna(subset=[variable+'_diff']) # Para eliminar la fila con NaN solo en la columna indicada
            datos_modelo.drop([variable], axis=1, inplace=True)
            variable = variable+'_diff'
    
    return datos_modelo[variable]

