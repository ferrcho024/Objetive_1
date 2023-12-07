# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:41:55 2021

@author: Fernando Avila
"""

import pandas as pd
import numpy as np
import seaborn as sns

#CORRELACIONES PERIÓDICAS ENTRE LAS VARIABLES DF Y NOVA

# 1. IMPORTANCIÓN DE DATOS Y FILTRADO
datos = pd.read_csv("F:\PhD\Datos SIATA\Análisis\Descriptivo\datosCoordenados_CS.csv",sep=",")
pm25 = datos.loc[:,["codigoSerial", "fecha", "hora", "pm25_df", "pm25_nova"]]


# 2. CREACIÓN DE LOS DATAFRAME PARA ALMACENAR LAS CORRELACIONES
corr_mes = pd.DataFrame()
corr_sem = pd.DataFrame()
#corr_dia = pd.DataFrame()
#corr_mes = pd.DataFrame(columns=['codigoSerial', 'mes1', 'mes2', 'mes3'])
#corr_sem = pd.DataFrame(columns=['codigoSerial', 'semana1', 'semana2', 'semana3', 'semana4'])
corr_dia = pd.DataFrame(columns=['codigoSerial', 'dia1', 'dia2', 'dia3', 'dia4', 'dia5', 'dia6', 'dia7',
                                'dia8', 'dia9', 'dia10', 'dia11', 'dia12', 'dia13', 'dia14', 'dia15', 'dia16',
                                'dia17', 'dia18', 'dia19', 'dia20', 'dia21', 'dia22', 'dia23', 'dia24', 'dia25', 'dia26',
                                'dia27', 'dia28', 'dia29', 'dia30', 'dia31'])

fechas = pm25.fecha.unique().tolist()
nodos = pm25['codigoSerial'].unique().tolist()
nodos.sort()

# 3. CALCULO DE CORRELACIONES *******************************************************************

# 3.1 Correlación mensual
for j in nodos:
    cont = 1
    mes = {'codigoSerial':int(j)}
    pm25_corr = pm25.loc[pm25.loc[:,"codigoSerial"] == j]
    pm25_corr.reset_index(inplace=True, drop=True)  # Reinicia índice del DataFrame
    mes["mes"+str(cont)] = round(np.corrcoef(pm25_corr["pm25_df"],pm25_corr["pm25_nova"])[0][1],2)


# 3.2 correlación diaria y semanal
    pm25_corr = pm25_corr.drop(range(0, len(pm25_corr),1),axis=0)
    sem = {'codigoSerial':int(j)}
    dia = {'codigoSerial':int(j)}
    for i in fechas:
        pm25_dia = pm25.loc[pm25.loc[:,"fecha"] == i]
        pm25_dia = pm25_dia.loc[pm25_dia.loc[:,"codigoSerial"] == j]
        dia["dia"+str(cont)]= round(np.corrcoef(pm25_dia["pm25_df"],pm25_dia["pm25_nova"])[0][1],2)
        
        pm25_corr = pd.concat([pm25_corr, pm25_dia])
        
        if cont%7 == 0:
            sem["semana"+str(int(cont/7))] = round(np.corrcoef(pm25_corr["pm25_df"],pm25_corr["pm25_nova"])[0][1],2)
            pm25_corr.reset_index(inplace=True, drop=True)  # Reinicia índice del DataFrame
            pm25_corr = pm25_corr.drop(range(0, len(pm25_corr),1),axis=0)
        
        cont += 1

# 3.3 Almacenamiento de la información en los dataframe    
    corr_mes = corr_mes.append(mes, ignore_index=True)
    corr_sem = corr_sem.append(sem, ignore_index=True) 
    corr_dia = corr_dia.append(dia, ignore_index=True)


# 4. GRÁFICOS DE LAS CORRELACIONES********************************************

corr_sem_cp = corr_sem.copy()

# 4.1 Transposición la matríz para dejar las semanas como filas
corr_sem_cp=corr_sem_cp.set_index('codigoSerial')
corr_sem_cp = corr_sem_cp.T

# 4.2 Modificacines de las columnas de datos para que las semanas queden en una colomna "Semanas"
corr_sem_cp['Semanas'] = corr_sem_cp.index
corr_sem_cp=corr_sem_cp.set_index('Semanas')
corr_sem_long = corr_sem_cp.reset_index().melt(id_vars="Semanas")

# 4.3 Copia de los datos para 9 nodos - 4 datos por nodo
corr_sem_long = corr_sem_long.head(36)

# 4.4 Datos del gráfico, con col_wrap se indica la cantidad de gráficos por fila
g = sns.FacetGrid(corr_sem_long, col="codigoSerial", height=3, col_wrap=3)
g = g.map(sns.pointplot, "Semanas", "value", order=['semana1', 'semana2', 'semana3', 'semana4'], color=".3")
g.fig.suptitle("Correlaciones semanales", fontsize=16, weight="bold", y=1.05)
#Text(0.5,1.05,'Cambio de la correlación semanal por estación CS')
#g.savefig("Correlaciones semanales.eps")


# corr_mes.to_csv('corr_mes.csv', index=False) 
# corr_sem.to_csv('corr_sem.csv', index=False)
# corr_dia.to_csv('corr_dia.csv', index=False)