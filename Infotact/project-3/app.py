import os
import tarfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to the .tar.xz archive
archive_path = r'C:\Users\devil\OneDrive\Desktop\infotact\project-3\AmazonMusicCompact.tar.xz'

# Check if the file exists
if os.path.exists(archive_path):
    print(f"File found: {archive_path}")
    
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

