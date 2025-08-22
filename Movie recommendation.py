import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = [
    {
        "title": "The Matrix",
        "genres": ["Action", "Sci-Fi"],
        "description": "A hacker learns about the true nature of reality and leads a rebellion against intelligent machines."
    },
    {
        "title": "Inception",
        "genres": ["Action", "Sci-Fi", "Thriller"],
        "description": "A thief enters dreams to plant ideas, navigating layered subconscious worlds and complex heists."
    },
    {
        "title": "Interstellar",
        "genres": ["Sci-Fi", "Drama"],
        "description": "Explorers travel through a wormhole to find a new home for humanity while a father fights against time."
    },
    {
        "title": "The Dark Knight",
        "genres": ["Action", "Crime", "Drama"],
        "description": "Batman faces the Joker in a gritty struggle for Gotham, testing morality, chaos, and justice."
    },
    {
        "title": "Pulp Fiction",
        "genres": ["Crime", "Drama"],
        "description": "Intersecting stories of criminals in Los Angeles with dark humor, nonlinear narrative, and sharp dialogue."
    },
    {
        "title": "The Shawshank Redemption",
        "genres": ["Drama"],
        "description": "Two imprisoned men forge a lasting friendship and find hope through perseverance and small acts of defiance."
    },
    {
        "title": "Avengers: Endgame",
        "genres": ["Action", "Adventure", "Sci-Fi"],
        "description": "Earth's mightiest heroes assemble for a final battle across time to undo catastrophic loss."
    },
    {
        "title": "Parasite",
        "genres": ["Drama", "Thriller"],
        "description": "A poor family infiltrates a wealthy household, exposing class tensions through suspense and dark comedy."
    },
    {
        "title": "Blade Runner 2049",
        "genres": ["Sci-Fi", "Drama"],
        "description": "A young blade runner uncovers a buried secret that leads him to a former blade runner and questions of identity."
    },
    {
        "title": "The Social Network",
        "genres": ["Drama", "Biography"],
        "description": "The controversial founding of a social media empire and the costs of ambition and friendship."
    },
    {
        "title": "Mad Max: Fury Road",
        "genres": ["Action", "Adventure"],
        "description": "In a desert apocalypse, a fierce driver and a rebel fighter flee a tyrant in a relentless chase."
    },
    {
        "title": "Whiplash",
        "genres": ["Drama", "Music"],
        "description": "An ambitious drummer faces a ruthless instructor, pushing limits of discipline, talent, and obsession."
    },
]

ratings_data = {
    "User":  ["A","A","A","B","B","B","C","C","D","D","E","E","F","F","G"],
    "Movie": [
        "The Matrix","Inception","Interstellar",
        "The Matrix","The Dark Knight","Pulp Fiction",
        "Inception","Interstellar",
        "Pulp Fiction","The Shawshank Redemption",
        "The Matrix","Pulp Fiction",
        "Blade Runner 2049","Interstellar",
        "Mad Max: Fury Road"
    ],
    "Rating":[5,4,5,5,4,4,4,5,4,5,5,3,5,4,5]
}

movies_df  = pd.DataFrame(movies)
ratings_df = pd.DataFrame(ratings_data)

descriptions = movies_df["description"].fillna("")
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),      
    min_df=1
)
tfidf_matrix = vectorizer.fit_transform(descriptions)  

title_to_idx = {t:i for i,t in enumerate(movies_df["title"])}

def content_scores_from_seed_title(seed_title: str) -> np.ndarray:
    """Return similarity scores to all movies given a seed title."""
    if seed_title not in title_to_idx:
        return np.zeros((len(movies_df),), dtype=float)
    idx = title_to_idx[seed_title]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()
    return sims

def content_scores_from_keywords(query: str) -> np.ndarray:
    """Return similarity scores to all movies given free-text keywords."""
    if not query.strip():
        return np.zeros((len(movies_df),), dtype=float)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    return sims

user_item = ratings_df.pivot_table(index="User", columns="Movie", values="Rating").fillna(0.0)
users = user_item.index.tolist()
movies_in_cf = user_item.columns.tolist()

user_sim = cosine_similarity(user_item)  
user_sim_df = pd.DataFrame(user_sim, index=users, columns=users)

def collaborative_scores_for_user(user_id: str) -> Dict[str, float]:
    """
    Predict a preference score per movie for the given user using
    similarity-weighted average of neighbors' ratings.
    Returns dict: {movie_title: score}
    """
    if user_id not in user_item.index:
        return {m: 0.0 for m in movies_df["title"]}
    
    user_ratings = user_item.loc[user_id]
    watched = set(user_ratings[user_ratings > 0].index)

    sims = user_sim_df.loc[user_id].copy()
    sims.drop(index=user_id, inplace=True)

    scores = {m: 0.0 for m in movies_df["title"]}

    for m in movies_df["title"]:
        if m in user_item.columns:
            neighbor_ratings = user_item[m]
            mask = neighbor_ratings > 0
            if mask.any():
                sim_vals = sims[mask]
                rate_vals = neighbor_ratings[mask]

                denom = sim_vals.abs().sum()
                if denom > 1e-9:
                    pred = float((sim_vals * rate_vals).sum() / denom)
                else:
                    pred = 0.0
            else:
                pred = 0.0
        else:
            pred = 0.0
        if m in watched:
            pred *= 0.5

        scores[m] = pred

    return scores

def safe_minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = np.min(x), np.max(x)
    if math.isclose(lo, hi):
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def hybrid_recommend(
    user_id: str,
    *,
    seed_title: str = "",
    keywords: str = "",
    top_k: int = 10,
    alpha_content: float = 0.6 
) -> List[Tuple[str, float, float, float]]:
    """
    Returns a ranked list of (title, hybrid_score, content_score_norm, collab_score_norm)
    """

    if seed_title and seed_title in title_to_idx:
        content_scores = content_scores_from_seed_title(seed_title)
    else:
        content_scores = content_scores_from_keywords(keywords)

    collab_dict = collaborative_scores_for_user(user_id)
    collab_scores = np.array([collab_dict[t] for t in movies_df["title"]], dtype=float)

    c_norm = safe_minmax(content_scores)
    k_norm = safe_minmax(collab_scores)

    hybrid = alpha_content * c_norm + (1 - alpha_content) * k_norm

    results = []
    for i, t in enumerate(movies_df["title"]):
        results.append((t, float(hybrid[i]), float(c_norm[i]), float(k_norm[i])))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    print("=== Hybrid Recommender (TF-IDF + Collaborative) ===")
    user = input("Enter user ID (e.g., A, B, C, D, E, F, G or new): ").strip()

    mode = input("Use a seed movie title (T) or free-text keywords (K)? [T/K]: ").strip().upper()
    seed = ""
    kw = ""
    if mode == "T":
        print("Available titles:")
        for t in movies_df['title']:
            print(" -", t)
        seed = input("Type a movie title exactly as above: ").strip()
    else:
        kw = input("Enter keywords/genres/themes (e.g., 'space travel, identity, noir'): ").strip()

    topn = input("How many recommendations? [default 10]: ").strip()
    topn = int(topn) if topn.isdigit() else 10

    alpha_in = input("Weight for content (0..1)? [default 0.6]: ").strip()
    try:
        alpha = float(alpha_in)
        if not (0 <= alpha <= 1):
            alpha = 0.6
    except:
        alpha = 0.6

    recs = hybrid_recommend(user, seed_title=seed, keywords=kw, top_k=topn, alpha_content=alpha)

    print("\nRecommendations (title | hybrid | content | collaborative):")
    for title, h, c, k in recs:
        print(f"{title:25s} | {h:0.3f} | {c:0.3f} | {k:0.3f}")
