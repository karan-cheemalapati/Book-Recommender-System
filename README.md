# Book Recommender System

A content-based and collaborative filtering book recommendation engine 
built with Python and Streamlit, deployed as an interactive web app.

🔗 **Live Demo:** [Click Here](https://huggingface.co/spaces/KaranCheemalapati/Book-Recommender-System)

---

## How It Works

- **Popularity-Based Filtering** — surfaces top-rated books based on 
  aggregated user ratings (minimum threshold: 50 ratings)
- **Collaborative Filtering** — uses cosine similarity on a user-book 
  pivot matrix to recommend titles liked by similar readers
- **Search** — real-time title/author search across the full catalog 
  with instant recommendation lookup

---

## Tech Stack

| Layer        | Tools                              |
|--------------|------------------------------------|
| Data         | pandas, NumPy                      |
| ML           | scikit-learn (cosine similarity)   |
| Frontend     | Streamlit                          |
| Deployment   | Hugging Face Spaces                |

---

## Dataset

Using the Book-Crossing Dataset:
- **Books.csv** — 271K book records (title, author, ISBN, cover URL)
- **Users.csv** — 278K anonymized users
- **Ratings.csv** — 1.1M ratings (explicit + implicit)

Filtering applied: users with >200 ratings, books with ≥50 ratings, 
to build a dense, high-quality similarity matrix.

---

## Run Locally

```bash
git clone https://github.com/karan-cheemalapati/Book-Recommender-System.git
cd Book-Recommender-System
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure
```
├── app.py                     # Streamlit app (UI + logic)
├── book_recommender.ipynb     # EDA and model development notebook
├── Books.csv                  # Book metadata
├── Users.csv                  # User data
├── Ratings.csv                # Rating data
└── requirements.txt
```

---

## Future Improvements

- Add content-based filtering using book genres/descriptions (NLP/TF-IDF)
- Integrate a user login system for personalized history
- Swap cosine similarity for matrix factorization (SVD/ALS)
- Add genre/author filters to the search UI
