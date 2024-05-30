import streamlit as st

st.header("My first web app")
#!pip install -U scikit-learn
#For running this code you need t20.csv file or any dataset to be uploaded.
import pandas as pd
import numpy as np

dataset = pd.read_csv('t20.csv')
X = dataset.iloc[:,[7,8,9,12,13]].values #Input features
y = dataset.iloc[:, 14].values #Label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
lin = RandomForestRegressor(n_estimators=100,max_features=None)
lin.fit(X_train,y_train)

# from sklearn.linear_model import LinearRegression
# lin = LinearRegression()
# lin.fit(X_train,y_train)

y_pred = lin.predict(X_test)
score = lin.score(X_test,y_test)*100

run = st.number_input("Total Runs Scored till now: ")
wickets = st.number_input("Total Wickets fallen till now: ")
overs = st.number_input("Total Overs bowled till now: ")
striker_score = st.number_input("Striker Score: ")
non_striker_score = st.number_input("Non Striker Score: ")
st.write("Predicted Score: ",lin.predict(sc.transform(np.array([[run,wickets,overs,striker_score,non_striker_score]]))))
st.write("Accuracy: ",score,"%")

# new_prediction = lin.predict(sc.transform(np.array([[148,2,10,54,13]])))
# #runs, wickets, overs, striker score, non-striker score

# st.write(new_prediction)