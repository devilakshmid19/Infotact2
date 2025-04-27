import os
import streamlit as st
import tarfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit app header
st.title("Amazon Music Recommendation System")

# Upload .tar.xz file
uploaded_file = st.file_uploader("Upload your .tar.xz file", type=["tar.xz"])

# Function to extract and load data from the tar.xz
def load_data_from_tar(file):
    with tarfile.open(file, 'r:xz') as tar_ref:
        # List all files in the tar archive
        file_names = tar_ref.getnames()
        st.write("Files in TAR.XZ:", file_names)
        
        # Specify the CSV file to be extracted (you can modify this if it's a different file)
        csv_file_path = 'AmazonMusic/amazon_music_metadata.csv'
        
        # Check if the expected CSV file exists
        if csv_file_path in file_names:
            # Extract and read the CSV file
            with tar_ref.extractfile(csv_file_path) as my_file:
                df = pd.read_csv(my_file)
            return df
        else:
            st.error(f"CSV file '{csv_file_path}' not found in the tar archive.")
            return None
# Path to the .tar.xz archive
archive_path = r'C:\Users\devil\OneDrive\Desktop\infotact\project-3\AmazonMusicCompact.tar.xz'

# Check if the file exists
if os.path.exists(archive_path):
    print(f"File found: {archive_path}")
    @st.cache_data
    def load_data():
        # Check the current directory to ensure the file is present
        st.write(f"Current directory: {os.getcwd()}")
    
        # Check if the file exists
        file_path = 'amazon_music_metadata.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            st.write(f"Data loaded successfully! Shape: {df.shape}")
            return df
        else:
            st.write("File not found!")
            return None
    
    # Open the .tar.xz archive
    with tarfile.open(archive_path, 'r:xz') as tar:
        # Extract the CSV file directly
        csv_filename = 'AmazonMusic/amazon_music_metadata.csv'
        
        try:
            # Open and read the CSV file directly from the tar archive
            with tar.extractfile(csv_filename) as file:
                # Load the CSV data into a DataFrame
                df = pd.read_csv(file)
                
                # Check the first few rows of the data
                print(df.head())
                
                # Preprocess the data (drop rows with missing titles)
                df = df.dropna(subset=['title'])
                
                # Initialize the TF-IDF Vectorizer to process the song titles
                tfidf = TfidfVectorizer(stop_words='english')
                
                # Fit the model and transform the song titles into numerical vectors
                tfidf_matrix = tfidf.fit_transform(df['title'])
                
                # Compute the cosine similarity between the song titles
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                # Create a DataFrame for easy access to similarity scores
                cosine_sim_df = pd.DataFrame(cosine_sim, index=df['title'], columns=df['title'])
                
                # Function to recommend songs based on a given song title
                def recommend_songs(song_title, cosine_sim_df, top_n=5):
                    # Get the similarity scores for the song
                    similar_songs = cosine_sim_df[song_title]
                    
                    # Sort the scores and return the top 'n' most similar songs
                    similar_songs = similar_songs.sort_values(ascending=False)[1:top_n+1]  # Exclude the song itself
                    return similar_songs
                
                # Example: Recommend songs similar to 'Memory of Trees'
                recommended_songs = recommend_songs('Memory of Trees', cosine_sim_df, top_n=5)
                print("Recommended Songs for 'Memory of Trees':")
                print(recommended_songs)
                
                # Optionally, you can add more logic to make it interactive, like user input
                user_input = input("Enter a song title for recommendations: ")
                if user_input in df['title'].values:
                    recommended_songs = recommend_songs(user_input, cosine_sim_df, top_n=5)
                    print(f"Recommended Songs for '{user_input}':")
                    print(recommended_songs)
                else:
                    print("Song title not found in the dataset.")
                    
        except KeyError:
            print(f"The file '{csv_filename}' was not found inside the archive.")
else:
    print(f"File not found: {archive_path}")

# Main function to manage the app interface
def main():
    if uploaded_file is not None:
        # Load the data from the uploaded tar.xz file
        df = load_data_from_tar(uploaded_file)
        
        if df is not None:
            # Display the first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(df.head())

            # Display the columns of the dataset
            st.subheader("Columns of Dataset")
            st.write(df.columns)

            # Check for the genre column and display the distribution
            if 'genre' in df.columns:
                st.subheader("Genre Distribution")
                genre_counts = df['genre'].value_counts()
                st.bar_chart(genre_counts)
            else:
                st.warning("The 'genre' column was not found in the dataset.")

            # Additional analysis or recommendations can be added here
            # For example, showing the top 5 most popular songs based on other columns

            st.subheader("Top 5 Songs by Popularity")
            if 'popularity' in df.columns:
                top_songs = df[['track_name', 'popularity']].sort_values(by='popularity', ascending=False).head(5)
                st.write(top_songs)
            else:
                st.warning("The 'popularity' column was not found in the dataset.")
            
            # You can add more interactive features here, like song recommendations, etc.

if __name__ == "__main__":
    main()

