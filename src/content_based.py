from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def clean_data(data):
    if isinstance(data, list):
        return [str.lower(i.replace(" ", "")) for i in data]
    else:
        if isinstance(data, str):
            return str.lower(data.replace(" ", ""))
        else:
            return ''


def create_movies_soup(row):
    return ' '.join(row['genres'])


def build_content_based_model(movies):
    features = ['genres']
    for feature in features:
        movies[feature] = movies[feature].apply(clean_data)
    movies['soup'] = movies.apply(create_movies_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim


def get_recommendations(title, movies, cosine_sim):
    movies = movies.reset_index()
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return "Movie not found."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]