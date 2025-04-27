import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the app
st.set_page_config(page_title="E-commerce Customer Segmentation", layout="wide")

st.title("🛒 E-commerce Customer Segmentation Dashboard")

# Load the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📄 Raw Data")
    st.dataframe(df)

    st.subheader("📊 Data Summary")
    st.write(df.describe())

    st.subheader("🧹 Null Values")
    st.write(df.isnull().sum())

    st.subheader("📈 Visualization")

    # Example: Plot distributions
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numeric_cols:
        col = st.selectbox("Select a numeric column to plot", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)
    
    st.subheader("📂 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

else:
    st.info('Please upload a CSV file to start.')

