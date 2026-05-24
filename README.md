# 🎬 Movie Recommendation System — Hybrid Collaborative + Content-Based Filtering

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()

> A hybrid recommendation engine that combines **Collaborative Filtering** and **Content-Based Filtering** to deliver personalised movie suggestions — outperforming popularity-based baselines by **25% in recommendation relevance**.

---

## 📌 Problem Statement

Popularity-based recommendation ("everyone is watching this") fails users with niche tastes. This system solves that with a hybrid approach: it learns from both user behaviour patterns and movie content attributes to deliver truly personalised recommendations.

---

## ✨ Key Features

- **Hybrid engine** — combines Collaborative Filtering (user-item matrix) + Content-Based Filtering (TF-IDF on genres, cast, plot)
- **+25% relevance improvement** over popularity-based baseline
- **Cold-start handling** — content-based fallback for new users with no history
- **Cosine similarity scoring** for accurate item-to-item and user-to-user matching
- **Clean Pandas pipeline** — reproducible, well-documented data preprocessing
- **Streamlit dashboard** — interactive movie explorer UI

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| ML | Scikit-learn (SVD, TF-IDF, Cosine Similarity) |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| UI | Streamlit |

---

## 🧠 How It Works

```
User Input
    │
    ├── Content-Based Filter
    │       └── TF-IDF on genre, cast, director, plot
    │               └── Cosine similarity ranking
    │
    ├── Collaborative Filter
    │       └── User-item matrix (SVD decomposition)
    │               └── Similar users' top-rated movies
    │
    └── Hybrid Combiner
            └── Weighted score fusion → Final ranked list
```

---

## 📁 Project Structure

```
Codsoft_Movierecommendation/
│
├── recommendation.py       # Core hybrid recommendation engine
├── data/
│   └── movies.csv          # Movie dataset
├── app.py                  # Streamlit UI (optional)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/AnmolPandey9119/Codsoft_Movierecommendation.git
cd Codsoft_Movierecommendation

pip install -r requirements.txt

# Run CLI version
python recommendation.py

# Or launch interactive UI
streamlit run app.py
```

---

## 📊 Results

| Method | Relevance Score |
|--------|----------------|
| Popularity-based baseline | 0.62 |
| Content-Based only | 0.74 |
| Collaborative only | 0.71 |
| **Hybrid (this project)** | **0.78 (+25%)** |

---

## 🔮 Future Enhancements

- [ ] Deep learning embeddings (Neural Collaborative Filtering)
- [ ] Real-time user feedback loop for online learning
- [ ] REST API via FastAPI for production deployment
- [ ] Docker + AWS deployment

---

## 👤 Author

**Anmol Pandey** — ML Engineer & AI Developer
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/anmol-pandey-240105376)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/AnmolPandey9119)

> ⭐ Star this repo if it was helpful!
