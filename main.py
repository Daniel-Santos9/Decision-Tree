#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:32:10 2019

@author: Daniel Santos
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#importando dataset
base = pd.read_csv('lung_cancer.csv')

#dividindo base entre previsores e classe
previsores = base.iloc[:,2:6].values
classe = base.iloc[:,6].values

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.30,random_state=0)
classificador = DecisionTreeClassifier(criterion='entropy',random_state=0)
#classificador = SVC(kernel = 'linear',random_state=1)

classificador.fit(previsores_treinamento,classe_treinamento)
predict = classificador.predict(previsores_teste)

#calculando a acurácia
acc_baseTrein = classificador.score(previsores_treinamento,classe_treinamento)

acc_baseTest = classificador.score(previsores_teste,classe_teste)

acc = accuracy_score(classe_teste, predict)

print(acc_baseTrein)
print(acc_baseTest)
print(acc)