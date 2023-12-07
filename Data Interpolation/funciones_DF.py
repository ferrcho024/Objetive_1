import pandas as pd
import numpy as np
import DQM as f_DQM
from IPython.display import display
from datetime import date, timedelta

def lagrangian (vari,varj):
# Halla el valor máximo de la varianza de xij
# Devuelve los valores de a y b  --> a y b
    # vari = Varianza del sensor i
    # varj = Varianza del sensor j
    
    a = varj/(vari + varj)
    b = vari/(vari + varj)
    
    return a,b

#def OLDF (xi,xj,vari,varj):
def OLDF (dataf):
# Optimal Linear Data Fusion (OLDF) technique
# Devuelve el valor esperado calculado de dos formas, aunque el resultado es el mismo --> f1 y f2
    # xi = observación del sensor i
    # xj = observación del sensor j
    # vari = Varianza del sensor i
    # varj = Varianza del sensor j
     
    '''
    ### Si podemos calcular los valores de a y b
    a = langragian(vari,varj)[0]
    b = 1-a
    f1 = (a*xi) + (b*xj)
        
    varXij_1 = (a**2)*vari + (b**2)*varj
    
    ### Si no podemos calcular los valores de a y b por alguna razón.
    f2 = ((varj*xi)+(vari*xj))/(vari + varj)
    
    varXij_2 = ((1/vari)+(1/varj))**(-1)
    '''

    datos = pd.DataFrame()

    if 'Pred_df' in dataf.columns:
        df = 'Pred_df'
    else:
        df = 'pm25_df'

    if 'Pred_nova' in dataf.columns:
        nova = 'Pred_nova'
    else:
        nova = 'pm25_nova'

    for n in dataf.codigoSerial.unique().tolist():
        nube = dataf[dataf['codigoSerial'] == n]
        var_df = np.var(nube[df])
        var_nova = np.var(nube[nova])
        #var_df = max(0,(nube[df].std()/nube[df].mean()))
        #var_nova = max(0,(nube[nova].std()/nube[nova].mean()))


        try:
            ### Si podemos calcular los valores de a y b
            a = lagrangian(var_df,var_nova)[0]
            b = 1-a
            nube['Fusion'] = (a*nube[df][nube[df].notnull()]) + (b*nube[nova][nube[nova].notnull()])
                
            varXij_1 = (a**2)*var_df + (b**2)*var_df
        
        except:
            ### Si no podemos calcular los valores de a y b por alguna razón.
            nube['Fusion'] = ((var_df * nube[df][nube[df].notnull()])+(var_nova * nube[nova][nube[nova].notnull()]))/(var_df + var_nova)
            
            varXij_2 = ((1/var_df)+(1/var_nova))**(-1)   

        datos = pd.concat([datos, nube])

    return datos

def OLDF_new (dataf, nodos):
# Optimal Linear Data Fusion (OLDF) technique
# Devuelve el valor esperado calculado de dos formas, aunque el resultado es el mismo --> f1 y f2
    # xi = observación del sensor i
    # xj = observación del sensor j
    # vari = Varianza del sensor i
    # varj = Varianza del sensor j
     
    datos = pd.DataFrame()

    if 'Pred_df' in dataf.columns:
        df = 'Pred_df'
    else:
        df = 'pm25_df'

    if 'Pred_nova' in dataf.columns:
        nova = 'Pred_nova'
    else:
        nova = 'pm25_nova'

    for n in nodos:
        nube = dataf[dataf['codigoSerial'] == n]
        dq_measure = pd.DataFrame()
        dq_measure, prec_detallado = f_DQM.precision(nube, [n], dq_measure, debug = 'N')
        fechas = list(set(nube.index.date))
        fechas.sort()
        for fecha in fechas:
            if (fecha - fechas[0]).days < 6:
                inicio = fechas[0]
            else:
                inicio = fecha - timedelta(weeks=1)
            last_week = nube[str(inicio):str(fecha)]
            #display(last_week)
            #last_week['hora'] = last_week.index.hour
            #last_week.boxplot(column=['Pred_nova'], by='hora', rot=90, figsize=(13, 5))
            last_week = last_week.groupby([last_week.index.hour])[[df, nova]].quantile([0.1,0.8])

            #display(cuartiles)
            #display(cuartiles.loc[0,0.75][df])
            #display(prec_detallado.loc[n, fecha.month, fecha.day][['pm25_df_hour','pm25_nova_hour']])

            filtro = nube.loc[str(fecha)]
            cuartiles = filtro.groupby([filtro.index.hour])[[df, nova]].quantile([0.1,0.8])
            filtro = filtro.resample('H').mean()
            #display(filtro)
            lista_a = []
            lista_b = []
            pen_df = pen_nova = 1
            for index, i in filtro.iterrows():
                diff_df = abs(last_week.loc[index.hour, 0.1][df] - last_week.loc[index.hour, 0.8][df])
                diff_nova = abs(last_week.loc[index.hour, 0.1][nova] - last_week.loc[index.hour, 0.8][nova])
                prec_df = prec_detallado.loc[n, fecha.month, fecha.day, index.hour]['pm25_df_hour']
                prec_nova = prec_detallado.loc[n, fecha.month, fecha.day, index.hour]['pm25_nova_hour']
                if abs(cuartiles.loc[index.hour, 0.1][df] - cuartiles.loc[index.hour, 0.8][df]) < 0.5*diff_df:
                    pen_df = 1.5
                if abs(cuartiles.loc[index.hour, 0.1][nova] - cuartiles.loc[index.hour, 0.8][nova]) < 0.5*diff_nova:
                    pen_nova = 1.5
                
                a,_ = lagrangian(1 - min(0.99, prec_df*pen_df), 1 - min(0.99,prec_nova*pen_nova))
                lista_a.append(a)
                lista_b.append(1-a)
                # prec_df.append(prec_detallado.loc[n, fecha.month, fecha.day, index.hour]['pm25_df_hour'])
                # prec_nova.append(prec_detallado.loc[n, fecha.month, fecha.day, index.hour]['pm25_nova_hour'])
            filtro['a'] = lista_a
            filtro['b'] = lista_b
            # filtro['Prec_df'] = prec_df
            # filtro['Prec_nova'] = prec_nova

            filtro['Fusion'] = (filtro['a']*filtro[df]) + (filtro['b']*filtro[nova])

            #filtro.drop(['a','b',], axis=1, inplace=True)
            #display(filtro.head(5))

            datos = pd.concat([datos, filtro])  
                
                

            



            




        # var_df = np.var(nube[df])
        # var_nova = np.var(nube[nova])
        # #var_df = max(0,(nube[df].std()/nube[df].mean()))
        # #var_nova = max(0,(nube[nova].std()/nube[nova].mean()))


        # try:
        #     ### Si podemos calcular los valores de a y b
        #     a = lagrangian(var_df,var_nova)[0]
        #     b = 1-a
        #     nube['Fusion'] = (a*nube[df][nube[df].notnull()]) + (b*nube[nova][nube[nova].notnull()])
                
        #     varXij_1 = (a**2)*var_df + (b**2)*var_df
        
        # except:
        #     ### Si no podemos calcular los valores de a y b por alguna razón.
        #     nube['Fusion'] = ((var_df * nube[df][nube[df].notnull()])+(var_nova * nube[nova][nube[nova].notnull()]))/(var_df + var_nova)
            
        #     varXij_2 = ((1/var_df)+(1/var_nova))**(-1)   

        # datos = pd.concat([datos, nube])

    return datos

def DF_new():
    # Pensar en plausabilidad
    # Evaluar variabilidad de los datos para determinar si es una linea recta, si lo es, penalizar.
    # Compararla con el comportamiento esperado diario de la señal.
    # Artificialidad de los datos.
    # Valor de la precisión de los datos.
    # Completitud de los datos



    

    return None

def estadisticos(pm25):
# Calcula los estadísticos (media, desviación y varianza) de los sensores df y nova de cada nodo
# Devuelve un dataframe con los estadistivos de cada nodo  --> estadisticos
    #pm25 --> Dataframe con los datos de todos los de los sensored df y nova de los nodos a calcularles los estadísticos

    nodos = pm25.codigoSerial.unique().tolist()

    estadisticos = pd.DataFrame(columns=['codigoSerial', 
                                        'media_df', 
                                        'mediana_df', 
                                        'desviacion_df', 
                                        'varianza_df', 
                                        'media_nova', 
                                        'mediana_nova', 
                                        'desviacion_nova', 
                                        'varianza_nova'] )
    for i in nodos:
        pm25_1 = pm25.loc[pm25.loc[:,"codigoSerial"] == i]
        pm25_1.reset_index(inplace=True, drop=True)  # Reinicia índice del DataFrame
        estadisticos = estadisticos.append({'codigoSerial': i, 
                                            'media_df': round(np.mean(pm25_1["pm25_df"]),2), 
                                            'mediana_df': round(np.median(pm25_1["pm25_df"]),2), 
                                            'desviacion_df': round(np.std(pm25_1["pm25_df"]),2), 
                                            'varianza_df': round(np.var(pm25_1["pm25_df"]),2),
                                            'media_nova': round(np.mean(pm25_1["pm25_nova"]),2), 
                                            'mediana_nova': round(np.median(pm25_1["pm25_nova"]),2), 
                                            'desviacion_nova': round(np.std(pm25_1["pm25_nova"]),2), 
                                            'varianza_nova': round(np.var(pm25_1["pm25_nova"]),2)}, 
                                        ignore_index=True)
    
    del nodos, pm25_1, i
    return estadisticos