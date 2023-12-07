#ALGORITMO AUTOENCODER EN FUNCIONES

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
import funciones_OD as f_OD

    
def create_sequences(values, time_steps):
# Generated training sequences for use in the model.
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


def entrenamiento(df, variable, TIME_STEP, graficos='N', debug='N'):
# Entrena el modelo ocn los datos de entrenamiento
# Devuelve el modelo entrenado, el history del modelo ajustado y la secuencia del entrenamiento 
# df --> Dataframe con los valores de entrenamiento
# variable --> Nombre de la columna que tiene los datos en el dataframe
# TIME_STEP  --> Cantidad de valores que entregará el autoencoder

    entrenamiento = df.copy()
    entrenamiento["fechaHora"] = entrenamiento["fecha"] + " " + entrenamiento["hora"]
    #f_OD.diferencias (entrenamiento,[variable],4)

    df_small_noise = entrenamiento.loc[:,["fechaHora", variable]]
    df_small_noise.set_index("fechaHora", inplace=True)

    training_mean = df_small_noise.mean()
    training_std = df_small_noise.std()
    df_training_value = (df_small_noise - training_mean) / training_std

    if debug == 'S':
        print("Number of training samples:", len(df_training_value))

    x_train = create_sequences(df_training_value.values, TIME_STEP)
    if debug == 'S':
        print("Training input shape: ", x_train.shape)

    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

    if graficos == 'S':
        model.summary()
    #plot_model(model, to_file='model_plot_convo.png', show_shapes=True, show_layer_names=True)

    history = model.fit(
        x_train,
        x_train,
        verbose=0,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        validation_data=(x_train,x_train),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    if graficos == 'S':
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Error Loss')
        plt.legend()
        #plt.savefig("Auto_training.eps", dpi=200, bbox_inches='tight')
        plt.show()

    del df_training_value, df_small_noise, entrenamiento
    return model, history, x_train, training_mean, training_std


def umbral(model, x_train, graficos='N', debug='N'):
# Calcula el treshold de los datos de acuerdo con el modelo entrenado
# Devuelve el threshold de los datos, la prediccion del entrenamiento
# model --> Modelo entrenado
# x_train  --> Secuencia de entrenamiento 

    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    if graficos == 'S':
        plt.hist(train_mae_loss, bins=50)
        plt.xlabel("Train MAE loss")
        plt.ylabel("No of samples")
        plt.show()


    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    if debug == 'S':
        print("Reconstruction error threshold: ", threshold)

    del train_mae_loss
    return threshold, x_train_pred


def learn_check(x_train_pred, x_train, graficos='N'):
# Chequeo gráfico del entrenamiento con los datos de entrenamiento y la predicción del entrenamiento
# Devuelve un gráfico del resultado del entreamiento
# x_train_pred --> Predicción de los datos de entrenamiento
# x_train --> Secuencia de entrenamiento

    train = []
    for i in range (len(x_train)):
        train.append(x_train[i][0][0])

    train_pred = []
    for i in range (len(x_train_pred)):
        train_pred.append(np.mean(x_train_pred[i]))

    if graficos == 'S':
        plt.plot(train[0:100], label="Training")
        plt.plot(train_pred[0:100], label="Training prediction")
        plt.legend()
        # plt.savefig("Auto_prediction_training.eps", dpi=200, bbox_inches='tight')
        plt.show()

    del train, train_pred
    return None

def deteccion(prediccion, variable, model, training_mean, training_std, threshold, TIME_STEP, graficos='N', debug='N'):
# Realiza la detección de los outliers de acuerdo con el modelo entrenado
# Devuelve la lista de los valores identificados como outliers
# prediccion --> Dataset con los valores que tienen outliers
# variable --> Nombre de la columna que tiene los datos en el dataframe
# model --> Modelo entrenado
# training_mean, training_std  --> Valores de media y desviación de los valores de entrenamiento (los entrega la función de entrenamiento)
# threshold --> Umbral para criterio de identificación como outlier
# TIME_STEP  --> Cantidad de valores que entregará el autoencoder


    #prediccion["fechaHora"] = prediccion["fecha"] + " " + prediccion["hora"]
    #f_OD.diferencias (prediccion,[variable],4)

    #df_daily_jumpsup = prediccion.loc[:,["fechaHora", variable]]
    df_daily_jumpsup = prediccion[[variable]]
    #df_daily_jumpsup.set_index("fechaHora", inplace=True)

    df_test_value = (df_daily_jumpsup - training_mean) / training_std
    
    if graficos == 'S':
        fig, ax = plt.subplots()
        df_test_value.plot(legend=False, ax=ax)
        plt.show()

    # Create sequences from test values.
    x_test = create_sequences(df_test_value.values, TIME_STEP)
    if debug == 'S':
        print("Test input shape: ", x_test.shape)
        
    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    if graficos == 'S':
        plt.hist(test_mae_loss[0:100], bins=50)
        plt.xlabel("test MAE loss")
        plt.ylabel("No of samples")
        plt.show()

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    if debug == 'S':
        print("Number of anomaly samples: ", np.sum(anomalies))
        print("Indices of anomaly samples: ", list(np.where(anomalies))[0])

    del df_daily_jumpsup, df_test_value, x_test, x_test_pred, test_mae_loss
    return list(np.where(anomalies)[0])


def graf_deteccion(prediccion, variable, anomalies, ventana):
# Realiza un gráfico de los valores identificados como outliers
# variable --> Nombre de la columna que tiene los datos en el dataframe
# Devuelve un gráfico de una porción de los datos con los valores indetificados como outliers
# prediccion --> Dataset con los valores que tienen outliers
# anomalies --> Lista de valores identificados como outliers (la entrega la función de detección)
# ventana --> Número de valores que se quieren graficar

    #index = np.where(anomalies)
    prediccion['detected'] = prediccion.loc[anomalies,variable]
    
    if len(anomalies) != 0:
        inicio = anomalies[0]
    else:
        inicio = 0
    
    fin = inicio + ventana

    pred = prediccion[inicio:fin]

    plt.plot(pred.index, pred[variable], color="green",label=str(variable))
    plt.plot(pred.index, pred.detected, 'o', color="red",label="Outliers")
    plt.title('Autoencoder Outlier Detection')
    plt.xlabel('Data')
    plt.ylabel('Value')
    plt.legend(loc='upper center')
    #plt.savefig("Auto_result.eps", dpi=200, bbox_inches='tight')
    plt.show()

    del inicio, fin, pred
    return None