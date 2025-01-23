# Movie Recommendation System

This project is a movie recommendation system that provides both content-based and collaborative filtering recommendations. The system uses movie metadata and user ratings to generate personalized movie recommendations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Functions](#functions)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/moldovanvictor/movieRecommendations.git
    cd movieRecommendations
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
   
   2.1. If you encounter any errors regarding some dependencies, try using Conda to install them:
    ```sh
    conda install -c conda-forge <package_name>
    ```

## Usage

1. Download the dataset from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/ , create a new directory named `data` and add the following files to it:
    - `movies_metadata.csv`
    - `ratings.csv`

2. Run the main script:
    ```sh
    python main.py
    ```

## Project Structure

The project is structured as follows:

```
movieRecommendations/
│
├── data/
│   ├── movies_metadata.csv
│   ├── ratings.csv
│
├── src/
│   ├── data_preparation.py
│   ├── content_based.py
│   └── collaborative_filtering.py
│
├── main.py
├── README.md
└── requirements.txt
```

## Functions

### Data Preparation (`src/data_preparation.py`)

- `load_data()`: Loads movies and ratings data from CSV files.
- `parse_json(data)`: Parses JSON data from a string.
- `preprocess_movies(movies)`: Preprocesses movies DataFrame by parsing genres.
- `preprocess_ratings(ratings, movies)`: Preprocesses ratings DataFrame and merges with movies DataFrame.

### Content-Based Recommendations (`src/content_based.py`)

- `clean_data(data)`: Cleans data by converting to lowercase and removing spaces.
- `create_movies_soup(row)`: Creates a 'soup' of genres for each movie.
- `build_content_based_model(movies)`: Builds a content-based recommendation model using genres.
- `get_recommendations(title, movies, cosine_sim)`: Gets movie recommendations based on content similarity.

### Collaborative Filtering Recommendations (`src/collaborative_filtering.py`)

- `build_collaborative_model(ratings)`: Builds a collaborative filtering recommendation model using SVD.
- `get_recommendations(model, user_id, ratings, n=10)`: Gets movie recommendations for a user based on collaborative filtering.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.