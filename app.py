import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="MovieMatch | Ultimate AI Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }

    .stButton>button {
        background: linear-gradient(90deg, #ff00cc 0%, #3333ff 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(255, 0, 204, 0.4);
        color: white;
    }

    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    .movie-card:hover {
        transform: scale(1.05);
        border: 1px solid rgba(255, 0, 204, 0.5);
        background: rgba(255, 255, 255, 0.08);
    }

    .movie-poster {
        border-radius: 15px;
        width: 100%;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    .movie-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 5px;
        color: #fff;
    }

    .movie-meta {
        font-size: 0.85rem;
        color: #aaa;
    }

    .rating-badge {
        background: rgba(255, 215, 0, 0.2);
        color: #ffd700;
        padding: 2px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.8rem;
    }

    h1, h2, h3 {
        color: white !important;
        font-weight: 800 !important;
    }

    .hero-section {
        padding: 4rem 0;
        text-align: center;
    }

    .hero-title {
        font-size: 4rem;
        background: -webkit-linear-gradient(#eee, #333);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .stSelectbox label {
        color: #ccc !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Data Loading ---
ARTIFACTS_PATH = Path("artifacts")

@st.cache_data
def load_data():
    movies = pd.read_pickle(ARTIFACTS_PATH / "movies_with_embeddings.pkl")
    sim_overview = np.load(ARTIFACTS_PATH / "sim_overview.npy")
    sim_tags = np.load(ARTIFACTS_PATH / "sim_tags.npy")
    return movies, sim_overview, sim_tags

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

def get_weights(genres):
    if any(g in genres for g in ["Drama", "Documentary", "Comedy"]):
        return 0.6, 0.4
    elif any(g in genres for g in ["Action", "Adventure", "ScienceFiction"]):
        return 0.3, 0.7
    else:
        return 0.5, 0.5

def recommend(movie_title, movies, sim_overview, sim_tags, top_n=5):
    if movie_title not in movies['title'].values:
        return []

    index = movies[movies['title'] == movie_title].index[0]
    input_genres = movies.iloc[index]['genres']

    w_overview, w_tags = get_weights(input_genres)
    sim_scores = w_overview * sim_overview[index] + w_tags * sim_tags[index]

    def genre_match(target_genres):
        match_count = sum(1 for g in input_genres if g in target_genres)
        return match_count >= max(len(input_genres) - 1, 1)

    filtered_indices = movies[movies['genres'].apply(genre_match)].index.tolist()
    
    # Combined similarity with weight
    sim_scores_filtered = [(i, sim_scores[i]) for i in filtered_indices if i != index]
    sorted_scores = sorted(sim_scores_filtered, key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = []
    for i, _ in sorted_scores:
        movie_row = movies.iloc[i]
        recommendations.append({
            'title': movie_row['title'],
            'poster': fetch_poster(movie_row['movie_id']),
            'year': movie_row['release_date'][:4] if pd.notna(movie_row['release_date']) else "N/A",
            'rating': f"{movie_row['vote_average']:.1f}"
        })
    return recommendations

# --- Load Data ---
try:
    movies_df, sim_overview, sim_tags = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure artifacts are in the 'artifacts' folder.")
    st.stop()

# --- App UI ---
st.markdown('<div class="hero-section"><h1 class="hero-title">MovieMatch</h1><p style="font-size: 1.2rem; color: #888;">AI-Powered Cinematic Discovery</p></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_movie = st.selectbox(
        "Search for a movie you love:",
        movies_df['title'].values,
        index=None,
        placeholder="Type a movie name..."
    )
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        discover = st.button("Discover Magic ✨")

if selected_movie and discover:
    st.markdown("---")
    
    # Input Movie Display
    input_idx = movies_df[movies_df['title'] == selected_movie].index[0]
    input_row = movies_df.iloc[input_idx]
    
    st.write("### Based on your interest in:")
    
    rec_col1, rec_col2, rec_col3 = st.columns([1, 1, 1])
    with rec_col2:
        st.markdown(f"""
            <div class="movie-card">
                <img src="{fetch_poster(input_row['movie_id'])}" class="movie-poster">
                <div class="movie-title">{selected_movie}</div>
                <div class="movie-meta">{input_row['release_date'][:4] if pd.notna(input_row['release_date']) else "N/A"} • <span class="rating-badge">★ {input_row['vote_average']:.1f}</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.write("### We recommend:")
    
    recommendations = recommend(selected_movie, movies_df, sim_overview, sim_tags)
    
    if recommendations:
        cols = st.columns(len(recommendations))
        for i, rec in enumerate(recommendations):
            with cols[i]:
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{rec['poster']}" class="movie-poster">
                        <div class="movie-title">{rec['title']}</div>
                        <div class="movie-meta">{rec['year']} • <span class="rating-badge">★ {rec['rating']}</span></div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for this movie.")

# --- Footer ---
st.markdown("""
<div style="margin-top: 100px; text-align: center; color: #555; font-size: 0.8rem; padding-bottom: 20px;">
    Powered by TMDB API & Sentence Transformers | Built with Streamlit
</div>
""", unsafe_allow_html=True)
