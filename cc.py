import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# Reading data
user_ratings_df = pd.read_csv("../input/the-movies-dataset/ratings.csv")
movie_metadata = pd.read_csv("../input/the-movies-dataset/movies_metadata.csv")

# Selecting necessary columns
movie_metadata = movie_metadata[['title', 'genres']]

# Creating user-item matrix
user_item_matrix = user_ratings_df.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)

# Define a KNN model on cosine similarity
cf_knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

# Fitting the model on our matrix
cf_knn_model.fit(user_item_matrix)

def movie_recommender_engine(movie_name, matrix, cf_model, n_recs):
    # Fit model on matrix
    cf_knn_model.fit(matrix)
    
    # Extract input movie ID
    movie_id = process.extractOne(movie_name, movie_metadata['title'])[2]
    
    # Calculate neighbour distances
    distances, indices = cf_model.kneighbors(matrix.iloc[movie_id].values.reshape(1, -1), n_neighbors=n_recs)
    movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    # List to store recommendations
    cf_recs = []
    for i in movie_rec_ids:
        cf_recs.append({'Title': movie_metadata['title'].iloc[i[0]], 'Distance': i[1]})
    
    # Select top number of recommendations needed
    df = pd.DataFrame(cf_recs, index=range(1, n_recs))
     
    return df

n_recs = 10
recommendations = movie_recommender_engine('Batman', user_item_matrix, cf_knn_model, n_recs)
print(recommendations)