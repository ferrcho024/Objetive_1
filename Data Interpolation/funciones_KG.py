
import pandas as pd
import numpy as np
import math as math

def extrac_data(estacionSIATA, fecha):    
# Función para extraer los datos de los sensores de acuerdo con la edtación SIATA más cercana
# Devuelve una lista de los nodos CS que están cercanos al nodo SIATA de referencia  --> nodos_CS
# Devuelve un dataframe con los datos de los nodos cercanos al nodo SIATA de referencia  --> pm25_c
# estacionSIATA --> Estación SIATA de referencia
# fecha  --> Fecha en la cuál se hara la comparación

    ruta = "F:/PhD/Datos SIATA/Análisis/Descriptivo/Datos/"
    datos = pd.read_csv(ruta+"datosCoordenados_CS.csv",sep=",")
    clusters = pd.read_csv(ruta+"clusters.csv",sep=",")

    # Exptracción de los datos necesarios de los archivos
    pm25 = datos.loc[:,["codigoSerial", "fecha", "hora", "pm25_df", "pm25_nova"]]
    pm25 = pm25.loc[pm25.loc[:,"fecha"] == fecha]
    pm25.reset_index(inplace=True, drop=True)

    # Tomar los nodos del cluster y pasarlos a una lista.
    # Se toma el daraframe de clusters y toma la fila que coincide con el valor de la estación,
    # el resultado que entrega son el índice y los valores, por lo que se toman solo los valores con el .value
    # luego se converte a una lista y se toma la posición 0, esto es un string
    # por ultimo se agregan el split y el strip para elimiar las llaves y tomar las comas como separador de la lista.  
    nodos_CS = clusters.nodosCS.loc[clusters['codigoSIATA']==estacionSIATA].values.tolist()[0].strip('][').split(', ')
    nodos_CS = [int(x) for x in nodos_CS]
    nodos_CS.sort()

    # Filtrado de datos solo de los nodos del cluster
    pm25_c = pd.DataFrame(columns=["codigoSerial", "fecha", "hora", "pm25_df", "pm25_nova"])
    for i in nodos_CS:
        pm25_c = pd.concat([pm25_c, pm25.loc[pm25.loc[:,"codigoSerial"] == int(i)]])

    # Elimina de la lista los nodos del cluster que no tienen datos en la fecha indicada
    nod = nodos_CS.copy()
    for i in nod:
        filtro = pm25_c.loc[pm25_c.loc[:,"codigoSerial"] == int(i)]
        if len(filtro.codigoSerial) == 0:
            nodos_CS.remove(i)
    

    del ruta, datos, cluster, pm25, nod, filtro, i
    return pm25_c, nodos_CS


def haversine(lon1, lat1, lon2, lat2):
# Fórmula de Haversine para calcular la distancia entre dos puntos
# Devuleve la distancia entre dos puntos de acuerdo con la fórmula de haversine en kms
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

    del dlon, dlat, a, lon1, lat1, lon2, lat2
    return c * R

def distances(df_CS, df_SIATA, estacionSIATA, nodos_CS):
# Calculo de las distancias entre nodos del cluster
# Devuelve un diccionario con las distancias de cada nodo CS con los demás nodos CS y con el nodo SIATA --> distancias
# estacionSIATA --> Estación SIATA de referencia
# nodos_CS --> Lista de nodos cercanos a la estación SIATA de referencia. Esta lista la entrega la función extrac_data(estacionSIATA, fecha)
    
    ruta = "F:/PhD/Datos SIATA/Análisis/Descriptivo/Datos/"
    coor_SIATA = pd.read_csv(ruta+"coordenadas_SIATA.csv",sep=",")
    coor_CS = pd.read_csv(ruta+"coordenadas_CS.csv",sep=",")

    lon_SIATA = float(coor_SIATA.longitud.loc[coor_SIATA.codigoSerial==estacionSIATA])
    lat_SIATA = float(coor_SIATA.latitud.loc[coor_SIATA.codigoSerial==estacionSIATA])

    distancias = {}
    for i in nodos_CS:
        #cluster = []
        dist1 = {} # Distancias desde un nodo SIATA a cada nodo CS
        dist = haversine(lon_SIATA, 
                        lat_SIATA, 
                        coor_CS.longitud.loc[coor_CS["codigoSerial"]==i], 
                        coor_CS.latitud.loc[coor_CS["codigoSerial"]==i])
        dist1["SIATA"] = dist
        for j in nodos_CS:
            dist = haversine(coor_CS.longitud.loc[coor_CS["codigoSerial"]==i], 
                            coor_CS.latitud.loc[coor_CS["codigoSerial"]==i], 
                            coor_CS.longitud.loc[coor_CS["codigoSerial"]==j], 
                            coor_CS.latitud.loc[coor_CS["codigoSerial"]==j])
            dist1[str(j)] = dist

        distancias[str(i)] = dist1
    
    del ruta,coor_SIATA, coor_CS, lon_SIATA, lat_SIATA, dist1, i

    '''
    distancias = df_CS.groupby(df_CS[df_CS.codigoSerial.isin(nodos_CS)]['codigoSerial'])['latitud','longitud'].mean()
    coor_SIATA = df_SIATA.groupby(df_SIATA[df_SIATA.codigoSerial == estacionSIATA]['codigoSerial'])['latitud','longitud'].mean()
    distancias = distancias.append(coor_SIATA)
    distancias['Dist'] = distancias.apply(lambda row : haversine(row['latitud'], 
                                                                row['longitud'], 
                                                                distancias.loc[estacionSIATA][0], 
                                                                distancias.loc[estacionSIATA][1]), axis=1)
    '''
    return distancias

def calculo_kriging(df_CS, df_SIATA, estacionSIATA, nodos_CS, distancias):
# Calcula el valos estimado en un punto geográfico mediante la estrategia de Kriging
# Devuelve un dataframe con los valores esperados en cada instante de tiempo en el punto referenciado  --> esperados
# nodos_CS --> Lista de nodos cercanos a la estación SIATA de referencia. Esta lista la entrega la función extrac_data(estacionSIATA, fecha)
# distancias --> Diccionario de las distancias de los nodos CS. Este diccionario lo entrega la función distancias(estacionSIATA, nodos_CS)
# pm25_c --> Dataframe con los datos de los nodos CS cercanos al nodo SIATA de referencia. Este dataframe lo entrga la función extrac_data(estacionSIATA, fecha)
    
    #distancias = distances(df_CS, df_SIATA, estacionSIATA, nodos_CS)
    
    # GENERACIÓN DE LA MATRIZ *******************************************
    # Calculo de gamma_ij
    matriz_var = np.ones((len(nodos_CS)+1, len(nodos_CS)+1))
    matriz_var[len(nodos_CS), len(nodos_CS)]=0
    vector_res = np.ones((len(nodos_CS)+1, 1))
    fila = 0
    for i in nodos_CS:
        col = 0
        for j in nodos_CS:
            resta = distancias[str(i)][str(j)]
            media = round(np.mean(resta**2),2) # Para que el np.mean????
            gamma = round(media/2,2)
            matriz_var[fila,col] = gamma
            col += 1
        resta = distancias[str(i)]["SIATA"]
        media = round(np.mean(resta**2),2) # Para que el np.mean????
        gamma = round(media/2,2)
        vector_res[fila,0]=gamma
        fila += 1

    # RESOLVIENDO LAS ECUACIONES LINEALES Y HALLANDO LAS INCOGNITAS.
    lamb = np.linalg.solve(matriz_var, vector_res)
    lamb = lamb.flatten().tolist()

    print(lamb)

    for i in range(len(lamb)):
        if (i+1) == len(lamb):
            print("Miu =", lamb[i])
            continue
        print("lambda", i+1, "=", lamb[i])


    # 5. HALLANDO EL VALOR ESTIMADO *****************************************
    esperados = pd.DataFrame(columns=["hora", "valorEsperado"])
    horas = df_CS.hora.unique().tolist()
    for h in horas:
        pm25 = df_CS.loc[df_CS.loc[:,"hora"] == h]
        #print(pm25)
        nod = nodos_CS#pm25.codigoSerial.unique().tolist()
        Zo = 0
        j = 0
        for i in nod:
            Zo += lamb[nodos_CS.index(i)]*(pm25.pm25_df.loc[pm25.codigoSerial == i].values[0])          
            j+= 1
        esperados = esperados.append({"hora":h,
                                    "valorEsperado":Zo},ignore_index=True)

    del matriz_var, vector_res, fila, col, resta, media, gamma, lamb, horas, pm25, nod, Zo, j, i
    return esperados