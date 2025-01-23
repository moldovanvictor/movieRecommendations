from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split


def build_collaborative_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    train_set, test_set = train_test_split(data, test_size=0.25)
    algo = SVD()
    algo.fit(train_set)
    return algo


def get_recommendations(model, user_id, ratings, n=10):
    user_ratings = ratings[ratings['userId'] == user_id]
    watched_movies = user_ratings['movieId'].tolist()
    all_movie_ids = ratings['movieId'].unique()
    recommendations = []

    for movie_id in all_movie_ids:
        if movie_id not in watched_movies:
            prediction = model.predict(user_id, movie_id)
            recommendations.append((movie_id, prediction.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:n]
    return [movie_id for movie_id, _ in top_recommendations]