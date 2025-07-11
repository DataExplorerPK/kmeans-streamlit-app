import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
import os
model_path = os.path.join(os.path.dirname(__file__), "kmeans.pkl")
model = pickle.load(open(model_path, "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))
n_clusters = model.n_clusters
st.write(f"✅ Model has `{n_clusters}` clusters.")

# Streamlit UI
st.set_page_config(page_title="Employee Clustering App", page_icon="🧠", layout="centered")
st.title("🧠 Employee Income Clustering App")
st.markdown("This app classifies employees into clusters based on **Age** and **Income** using a trained KMeans model.")

# Input fields
age = st.number_input("Enter Age", min_value=18, max_value=100, value=25)
income = st.number_input("Enter Income (₹)", min_value=1000, value=50000)

if st.button("Predict Cluster"):
    # Prepare input
    input_df = pd.DataFrame([[age, income]], columns=["Age", "Income"])
    scaled_input = scaler.transform(input_df)
    cluster = int(model.predict(scaled_input)[0])

    # Output result
    st.subheader("📊 Results:")
    st.write(f"🔹 **Predicted Cluster Number**: `{cluster}`")

    # Load and scale training data
    df = pd.read_excel("Employee_income.xlsx")
    df_scaled = scaler.transform(df[['Age', 'Income']])
    df_scaled = pd.DataFrame(df_scaled, columns=["Age", "Income"])
    df_scaled['Cluster'] = model.predict(df_scaled)

    # Prepare plot
    fig, ax = plt.subplots()
    colors = sns.color_palette("hsv", n_clusters)

    # Plot training data points by cluster
    for i in range(n_clusters):
        cluster_data = df_scaled[df_scaled['Cluster'] == i]
        ax.scatter(
            cluster_data['Age'],
            cluster_data['Income'],
            label=f"Cluster {i}",
            color=colors[i],
            alpha=0.5
        )

    # Plot cluster centers
    centroids = model.cluster_centers_
    for i in range(n_clusters):
        ax.scatter(
            centroids[i][0], centroids[i][1],
            color=colors[i],
            marker='*', s=200,
            edgecolors='black',
            label=f"Cluster {i} Center"
        )

    # Plot user input
    ax.scatter(
        scaled_input[0][0], scaled_input[0][1],
        color='red', s=120, edgecolors='black', label="Your Input"
    )

    padding = 0.05
    x_vals = df_scaled['Age'].tolist() + [scaled_input[0][0]]
    y_vals = df_scaled['Income'].tolist() + [scaled_input[0][1]]

    ax.set_xlim(min(x_vals) - padding, max(x_vals) + padding)
    ax.set_ylim(min(y_vals) - padding, max(y_vals) + padding)
    ax.set_xlabel("Scaled Age")
    ax.set_ylabel("Scaled Income")
    ax.set_title("📊 Clustered Employees + Your Input")
    ax.legend()
    st.pyplot(fig)
