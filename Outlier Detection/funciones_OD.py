# FUNCIONES AUXILIARES
import numpy as np

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
        
        if ponderaciones[i][1] == 0:
            HBOS[k] = HBOS[k] + 10
        else: 
            HBOS[k] = HBOS[k] + np.log10(1/ponderaciones[i][2]) # HBOS con base en las cantidades del dataframe
#            HBOS[k] = HBOS[k] + np.log10(1/ponderaciones[i][3]) # HBOS con base en las cantidades de cada bin
        
        k += 1
        
    return HBOS

def verifica_HBOS(HBOS,valor,bins):
    # Verifica el score HBOS de un valor dado.

    #keys = list(map(float,HBOS.keys())) # Pasa a una lista los key del diccionario y los convierte a float
    pos,val = min(enumerate(bins), key=lambda x: x[1]<=valor) # Busca el valor mayor más cercano al valor de referencia
    
    return HBOS[pos]
    
def diferencias (df,variables,redondeo):
    # Calcula la diferencia entre dos valores contiguos i y i-1

    for var in variables:
        shape = df.shape
        df.insert(shape[1]-2, var+'_dif', 0, allow_duplicates=False) # Agrega la columna para el valor de la diferencia
        for i in range (1,len(df)):
            df.loc[i,var+'_dif']= round(abs(df[var][i] - df[var][i-1]),redondeo)
    
    return df