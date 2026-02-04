import pandas as pd
import numpy as np
import ast
import nltk
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore", category=UserWarning)

lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('all-MiniLM-L6-v2')

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def top3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_text(text):
    words = nltk.word_tokenize(text)
    tagged = pos_tag(words)
    return " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged])

# Load data
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date', 'vote_average']]
movies.dropna(inplace=True)
movies.reset_index(drop=True, inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(top3)
movies['crew'] = movies['crew'].apply(director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

for col in ['genres', 'keywords', 'cast', 'crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

tfidf_kw = TfidfVectorizer(max_features=300, stop_words='english')
keyword_str = movies['keywords'].apply(lambda x: " ".join(x))
tfidf_kw_matrix = tfidf_kw.fit_transform(keyword_str)
kw_feature_names = tfidf_kw.get_feature_names_out()

def get_top_keywords(row, tfidf_vector, top_n=5):
    vector = tfidf_vector[row.name].toarray().flatten()
    top_indices = vector.argsort()[-top_n:][::-1]
    return [kw_feature_names[i] for i in top_indices if vector[i] > 0]

movies['filtered_keywords'] = movies.apply(lambda row: get_top_keywords(row, tfidf_kw_matrix), axis=1)

def create_tags(row):
    cast = ["cast_" + i for i in row['cast']]
    crew = ["crew_" + i for i in row['crew']]
    keywords = ["kw_" + i for i in row['filtered_keywords']]
    return " ".join(cast + crew + keywords).lower()

movies['tag_str'] = movies.apply(create_tags, axis=1)
movies['tag_str'] = movies['tag_str'].apply(lemmatize_text)
movies['overview_str'] = movies['overview'].apply(lambda x: " ".join(x))

movies['overview_emb'] = list(model.encode(movies['overview_str'].tolist(), show_progress_bar=True))
movies['tag_emb'] = list(model.encode(movies['tag_str'].tolist(), show_progress_bar=True))

movies.to_pickle("movies_with_embeddings.pkl")

from sklearn.metrics.pairwise import cosine_similarity

overview_vecs = np.vstack(movies['overview_emb'])
tag_vecs = np.vstack(movies['tag_emb'])

sim_overview = cosine_similarity(overview_vecs)
sim_tags = cosine_similarity(tag_vecs)

np.save("sim_overview.npy", sim_overview)
np.save("sim_tags.npy", sim_tags)
