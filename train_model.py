import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle

df = pd.read_excel("Employee_income.xlsx")
scaler = MinMaxScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

km = KMeans(n_clusters=3, random_state=42)
km.fit(df[['Age', 'Income']])

pickle.dump(km, open("kmeans.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Saved: kmeans.pkl and scaler.pkl")
