from src.data_preparation import load_data, preprocess_movies, preprocess_ratings
from src.content_based import build_content_based_model, get_recommendations as get_content_recommendations
from src.collaborative_filtering import build_collaborative_model, get_recommendations as get_collaborative_recommendations


def main():
    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings, movies)

    # Use Case 1: Finding Movies with Similar Themes and Genres
    cosine_sim = build_content_based_model(movies)
    movie_title = 'Interstellar'
    content_recommendations = get_content_recommendations(movie_title, movies, cosine_sim)
    print("Content-Based Recommendations for 'Inception':")
    print(content_recommendations)

    # Use Case 2: Personalized Movie Recommendations based on User Preferences
    algo = build_collaborative_model(ratings)
    user_id = 1
    collaborative_recommendations = get_collaborative_recommendations(algo, user_id, ratings)
    recommended_titles = ratings[ratings['movieId'].isin(collaborative_recommendations)]['title'].unique()
    print(f"Personalized Movie Recommendations for User {user_id}:")
    print(recommended_titles)


if __name__ == "__main__":
    main()