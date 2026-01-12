## 🎬 Genre-Aware Hybrid Movie Recommendation System

A production-style movie recommendation system built using ** Collaborative Filtering + Genre-Aware Content Filtering + IMDB-style ** confidence weighting, designed to deliver relevant, personalized, and reliable movie recommendations.

This project addresses real-world challenges such as cold start, data sparsity, and genre mismatch, making it suitable for internship and entry-level data science roles.

## 🚀 Project Overview

Traditional recommendation systems often suffer from:

`` recommending popular but irrelevant movies

`` failing for new or cold users

`` trusting movies with few ratings

This project solves those problems by combining:

`` User behavior similarity (Collaborative Filtering)

`` Genre similarity (Content-based filtering using TF-IDF)

`` Rating reliability (IMDB weighted rating formula)

The result is a hybrid recommender system similar to those used by ** Netflix and Amazon.**

## 🧠 Recommendation Strategy

The final recommendation score is computed as:

### Final Score = α × Collaborative Similarity
###           + β × Genre Similarity
###           + γ × IMDB Weighted Rating

Where:

``Collaborative Similarity captures user taste.

``Genre Similarity ensures contextual relevance.

``IMDB Weighted Rating ensures quality & trust.

``α, β, γ are tunable weights.

## ✨ Key Features

``✅ Multi-movie cold user recommendations

``✅ Genre-aware filtering to avoid mismatched suggestions

``✅ IMDB-style Bayesian confidence scoring

``✅ Sparse matrix optimization for scalability

``✅ Robust fallbacks for edge cases

``✅ Logging for traceability and debugging

## 🧰 Technologies Used

`` ** Python **

Pandas & NumPy

Scikit-Learn

SciPy (CSR Sparse Matrix)

TF-IDF Vectorization

Cosine Similarity

Logging

## 📂 Dataset

MovieLens Dataset

Columns include:

userId

movieId

rating

title

genres

Genres are preprocessed into a clean, TF-IDF-ready format.

## 🏗️ System Architecture
### 1️⃣ Collaborative Filtering

User-movie rating matrix (sparse)

Cosine similarity between movies

Captures user behavior patterns

### 2️⃣ Genre-Based Similarity

TF-IDF vectorization of movie genres

Cosine similarity between genre vectors

Ensures genre-consistent recommendations

### 3️⃣ IMDB Weighted Rating

Used to avoid unreliable movies with few ratings:

Weighted Rating = (v / (v + m)) × R + (m / (v + m)) × C


Where:

R = average rating of the movie

v = number of ratings

C = global average rating

m = minimum votes threshold

### 🧪 Example Usage
watched_movies = [109487, 79132, 134130]
recommendations = recommend_genre_aware_hybrid(watched_movies)

print(recommendations)

This returns:

#### Movie Title
#### Weighted Rating
#### Final Hybrid Score

🎯 What Makes This Project Stand Out

🔥 Goes beyond basic collaborative filtering

🔥 Handles cold-start intelligently

🔥 Uses industry-inspired ranking logic

🔥 Designed with scalability in mind

🔥 Resume-ready real-world system

## 👤 Author

### Anshuman Gupta
### Aspiring Data Scientist
### Passionate about Machine Learning & Recommendation Systems
