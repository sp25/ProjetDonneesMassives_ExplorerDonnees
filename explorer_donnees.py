import numpy as np
import random
import csv
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing

def explorer_donnees():
    # ************************************************************************************
    # Explorer les attributs reliés à la MÉTÉO
    datasetMeteo = pandas.read_table('dataset/train.csv', delimiter=',', header=0, usecols=[4, 5, 6, 7, 8, 11])

    # Générer des histogrammes pour observer la distribution des données
    datasetMeteo.hist('windspeed', bins=20)
    datasetMeteo.hist('temp', bins=20)
    datasetMeteo.hist('atemp', bins=20)
    datasetMeteo.hist('humidity', bins=20)

    # Générer des box-plot pour voir la médiane, les valeurs min-max, valeurs Q1 et Q3, et valeurs abberantes.
    datasetMeteo.boxplot(column='windspeed')
    datasetMeteo.boxplot(column='temp')
    datasetMeteo.boxplot(column='atemp')
    datasetMeteo.boxplot(column='humidity')

    # normaliser les données selon le z-score
    datasetMeteo['winds'] = preprocessing.scale(datasetMeteo['windspeed'])
    datasetMeteo['tempC'] = preprocessing.scale(datasetMeteo['temp'])
    datasetMeteo['tempaC'] = preprocessing.scale(datasetMeteo['atemp'])
    datasetMeteo['humid'] = preprocessing.scale(datasetMeteo['humidity'])

    # retirer les anciennes colonnes du dataset en mémoire
    datasetMeteo.drop('windspeed', axis=1, inplace=True)
    datasetMeteo.drop('temp', axis=1, inplace=True)
    datasetMeteo.drop('atemp', axis=1, inplace=True)
    datasetMeteo.drop('humidity', axis=1, inplace=True)

    # Générer la matrice des nuages de points (scatter)
    pandas.scatter_matrix(datasetMeteo, figsize=(6, 6))
    plt.show()

    # Générer la matrice de corrélation
    plt.matshow(datasetMeteo.corr())
    plt.xticks(range(len(datasetMeteo.columns)), datasetMeteo.columns)
    plt.yticks(range(len(datasetMeteo.columns)), datasetMeteo.columns)
    plt.colorbar()
    plt.show()

    #*******************************************************************************************
    # Explorer les autres attributs. Le plus significatif de la météo (temp) est conservé.
    datasetTotal = pandas.read_table('dataset/train.csv', delimiter=',', header=0, usecols=[0, 1, 2, 3, 5, 9, 10, 11])

    # Conserver l'attribut temperature qui avait montré le plus d'intérêts à l'étape précédente
    datasetTotal['tempC'] = preprocessing.scale(datasetTotal['temp'])
    datasetTotal.drop('temp', axis=1, inplace=True)

    # À partir de l'attribut datetime, crééer des nouveaux attributs (month, day, hours)
    datasetTotal['time'] = pandas.to_datetime(datasetTotal['datetime'])
    datasetTotal['month'] = datasetTotal.time.dt.month
    datasetTotal['day'] = datasetTotal.time.dt.dayofweek
    datasetTotal['hour'] = datasetTotal.time.dt.hour
    datasetTotal.drop('time', axis=1, inplace=True)
    datasetTotal.drop('datetime', axis=1, inplace=True)

    # Générer des histogrammes pour observer la distribution des données
    datasetTotal.hist('month', bins=20)
    datasetTotal.hist('day', bins=20)
    datasetTotal.hist('hour', bins=20)
    datasetTotal.hist('workingday', bins=20)
    datasetTotal.hist('holiday', bins=20)
    datasetTotal.hist('season', bins=20)
    datasetTotal.hist('casual', bins=20)
    datasetTotal.hist('registered', bins=20)

    # Générer des box-plot pour voir la médiane, les valeurs min-max, valeurs Q1 et Q3, et valeurs abberantes.
    datasetTotal.boxplot(column='casual')
    datasetTotal.boxplot(column='registered')

    # Matrice de nuages de point entre les différents attributs (scatter)
    pandas.scatter_matrix(datasetTotal, figsize=(6, 6))
    plt.show()

    # Matrice de corrélation entre les différents attributs.
    plt.matshow(datasetTotal.corr())
    plt.xticks(range(len(datasetTotal.columns)), datasetTotal.columns)
    plt.yticks(range(len(datasetTotal.columns)), datasetTotal.columns)
    plt.colorbar()
    plt.show()

    # Nuage de points entre count et registered, dont la corrélation trouvée est élevée.
    xdata = datasetTotal['registered']
    ydata = datasetTotal['count']
    fig, ax = plt.subplots()
    ax.set_xlabel("registered")
    ax.set_ylabel("count")
    ax.scatter(xdata, ydata)
    fig.show()

    # Nuage de points entre count et casual, dont la corrélation trouvée est élevée.
    xdata = datasetTotal['casual']
    ydata = datasetTotal['count']
    fig, ax = plt.subplots()
    ax.set_xlabel("casual")
    ax.set_ylabel("count")
    ax.scatter(xdata, ydata)
    fig.show()

    # Nuage de points entre registered et count, dont la corrélation trouvée est élevée.
    xdata = datasetTotal['hour']
    ydata = datasetTotal['registered']
    fig, ax = plt.subplots()
    ax.set_xlabel("hour")
    ax.set_ylabel("registered")
    ax.scatter(xdata, ydata)
    fig.show()

    # Nuage de points entre hour et casual, dont la corrélation trouvée est élevée.
    xdata = datasetTotal['hour']
    ydata = datasetTotal['casual']
    fig, ax = plt.subplots()
    ax.set_xlabel("hour")
    ax.set_ylabel("casual")
    ax.scatter(xdata, ydata)
    fig.show()

    # Nuage de points entre temperature et casual, dont la corrélation trouvée est élevée.
    xdata = datasetTotal['tempC']
    ydata = datasetTotal['casual']
    fig, ax = plt.subplots()
    ax.set_xlabel("temperature")
    ax.set_ylabel("casual")
    ax.scatter(xdata, ydata)
    fig.show()

    return 1

