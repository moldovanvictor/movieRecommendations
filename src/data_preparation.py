import pandas as pd
import ast


def load_data():
    movies = pd.read_csv('data/movies_metadata.csv', low_memory=False)
    ratings = pd.read_csv('data/ratings.csv')
    return movies, ratings


def parse_json(data):
    try:
        return ast.literal_eval(data)
    except ValueError:
        return []


def preprocess_movies(movies):
    movies['genres'] = movies['genres'].apply(parse_json)
    movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    return movies


def preprocess_ratings(ratings, movies):
    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    ratings = ratings.dropna(subset=['movieId'])
    movies = movies.dropna(subset=['id'])
    ratings = ratings.merge(movies[['id', 'title']], left_on='movieId', right_on='id')
    ratings = ratings[['userId', 'movieId', 'rating', 'title']]
    return ratings