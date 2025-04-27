import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio

# Initialize Flask app
app = Flask(__name__)

# Load the customer data (ensure you have a CSV or similar data)
DATA_PATH = 'customer_data.csv'  # path to your dataset

# Function to load and preprocess data
def load_data():
    df = pd.read_csv(DATA_PATH)
    
    # Ensure the columns you're working with exist
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Example encoding
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    return df

# Function to perform KMeans clustering
def perform_clustering(df, n_clusters=3):
    # Select features for clustering
    features = ['feature1', 'feature2', 'Orders', 'Revenue', 'Cost']  # Example features
    df_cluster = df[features]
    
    # Standardize the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)
    
    return df

# Define the route for the home page
@app.route('/')
def index():
    # Load and process the data
    df = load_data()
    
    # Perform clustering
    df = perform_clustering(df)
    
    # Create a scatter plot for visualization
    fig = px.scatter(df, x='Orders', y='Revenue', color='cluster', title="Customer Segmentation")
    graph_html = pio.to_html(fig, full_html=False)
    
    return render_template('index.html', plot=graph_html)

# API route to get cluster information
@app.route('/get_clusters', methods=['POST'])
def get_clusters():
    # Load and process the data
    df = load_data()
    
    # Number of clusters from the request, default is 3
    n_clusters = int(request.form.get('n_clusters', 3))
    
    # Perform clustering
    df = perform_clustering(df, n_clusters)
    
    # Return the cluster information as a JSON response
    cluster_data = df[['Customer_ID', 'cluster']].to_dict(orient='records')
    return jsonify(cluster_data)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
