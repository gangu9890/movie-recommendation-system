from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from pathlib import Path

PROJECT_ROOT = Path(settings.BASE_DIR).parent 
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"

movies = pd.read_pickle(ARTIFACTS_PATH / "movies_with_embeddings.pkl")
sim_overview = np.load(ARTIFACTS_PATH / "sim_overview.npy")
sim_tags = np.load(ARTIFACTS_PATH / "sim_tags.npy")

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path'] if 'poster_path' in data else ""
    except:
        return ""

def get_weights(genres):
    if "Drama" in genres or "Documentary" in genres or "Comedy" in genres:
        return 0.6, 0.4
    elif "Action" in genres or "Adventure" in genres or "ScienceFiction" in genres:
        return 0.3, 0.7
    else:
        return 0.5, 0.5

def recommend(movie_title, top_n=4):
    if movie_title not in movies['title'].values:
        return [], [], [], []

    index = movies[movies['title'] == movie_title].index[0]
    input_genres = movies.iloc[index]['genres']

    w_overview, w_tags = get_weights(input_genres)
    sim_scores = w_overview * sim_overview[index] + w_tags * sim_tags[index]

    def genre_match(target_genres):
        match_count = sum(1 for g in input_genres if g in target_genres)
        return match_count >= max(len(input_genres) - 1, 1)

    filtered_indices = movies[movies['genres'].apply(genre_match)].index.tolist()
    sim_scores = [(i, sim_scores[i]) for i in filtered_indices if i != index]
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

    titles, posters, years, ratings = [], [], [], []
    for i, _ in sorted_scores:
        movie_row = movies.iloc[i]
        titles.append(movie_row['title'])
        posters.append(fetch_poster(movie_row['movie_id']))
        years.append(movie_row['release_date'][:4] if pd.notna(movie_row['release_date']) else "N/A")
        ratings.append(f"{movie_row['vote_average']:.1f}")
    return titles, posters, years, ratings

def home(request):
    if request.method == "POST":
        movie_title = request.POST.get("movie")
        titles, posters, years, ratings = recommend(movie_title)
        input_movie = movie_title
        input_index = movies[movies['title'] == movie_title].index[0]
        input_poster = fetch_poster(movies.iloc[input_index].movie_id)
        recommendations = zip(titles, posters, years, ratings)
        return render(request, "myapp/home.html", {
            "recommendations": recommendations,
            "movie_title": movie_title,
            "input_movie": input_movie,
            "input_poster": input_poster
        })
    return render(request, "myapp/home.html")

def get_titles(request):
    titles = movies['title'].tolist()
    return JsonResponse(titles, safe=False)
