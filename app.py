import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        body, .stApp, .main, .css-18e3th9, .css-1outpf7 {
            background: #090909;
            color: #f5f5f5;
        }
        .css-1aumxhk, .css-1d391kg, .css-10trblm, .css-ffhzg2,
        .stMarkdown, .stText, .stSelectbox, .stSlider {
            color: #f5f5f5;
        }
        .stSidebar {
            background-color: #111111;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
        .book-card {
            background: #121212;
            padding: 1rem;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            color: #f5f5f5;
            text-align: left;
            transition: transform 0.2s ease;
            margin-bottom: 1rem;
        }
        .book-card:hover {
            transform: translateY(-3px);
        }
        .book-title {
            font-size: 1rem;
            font-weight: 700;
            margin: 0.5rem 0 0.25rem;
            color: #ffffff;
        }
        .book-author {
            font-size: 0.95rem;
            color: #b3c7d6;
            margin-bottom: 0.5rem;
        }
        .book-meta {
            color: #d6d6d6;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .stButton>button {
            background-color: #3a7ca5;
            color: white;
            border-radius: 0.75rem;
        }
        .css-1n76uvr, .css-1n76uvr span, .css-1n76uvr dt,
        .css-1n76uvr dd {
            color: #f5f5f5;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data"""
    books = pd.read_csv('Books.csv')
    users = pd.read_csv('Users.csv')
    ratings = pd.read_csv('Ratings.csv')

    all_books_df = books[['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-M']].drop_duplicates('Book-Title')
    all_books_df['Cover-URL'] = all_books_df['ISBN'].apply(
        lambda isbn: f'https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg' if pd.notna(isbn) else None
    )
    all_books_df['Search-Text'] = (
        all_books_df['Book-Title'].str.lower() + ' ' + all_books_df['Book-Author'].str.lower()
    )

    # Merge ratings and books
    ratings_books_merged = ratings.merge(books, on='ISBN')

    # Count and average ratings
    num_rating = ratings_books_merged.groupby('Book-Title')['Book-Rating'].count().reset_index()
    num_rating.rename(columns={'Book-Rating': 'Num-Ratings'}, inplace=True)

    avg_rating = ratings_books_merged.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating.rename(columns={'Book-Rating': 'Avg-Rating'}, inplace=True)

    # Get popular books
    popular_books = num_rating.merge(avg_rating, on='Book-Title')
    popular_df = popular_books[popular_books['Num-Ratings'] >= 50].sort_values('Avg-Rating', ascending=False).head(706)
    popular_books_df = popular_df.merge(ratings_books_merged, on='Book-Title').drop_duplicates('Book-Title')[
        ['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-M', 'Num-Ratings', 'Avg-Rating']
    ]
    popular_books_df['Cover-URL'] = popular_books_df['ISBN'].apply(
        lambda isbn: f'https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg' if pd.notna(isbn) else None
    )

    # Filter for collaborative filtering
    x = ratings_books_merged.groupby('User-ID').count()['Book-Rating'] > 200
    famous_users = x[x].index

    filtered_ratings = ratings_books_merged[ratings_books_merged['User-ID'].isin(famous_users)]
    y = filtered_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index

    final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]
    final_ratings = final_ratings.drop_duplicates()

    # Create pivot table and similarity matrix
    pivot_table = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pivot_table.fillna(0, inplace=True)

    sim_score = cosine_similarity(pivot_table)

    return popular_books_df, all_books_df, pivot_table, sim_score

@st.cache_data
def get_recommendations(book_name, pivot_table, sim_score, n=6):
    """Get book recommendations based on cosine similarity"""
    try:
        book_index = np.where(pivot_table.index == book_name)[0][0]
        similar_items = sorted(list(enumerate(sim_score[book_index])), key=lambda x: x[1], reverse=True)[1:n+1]
        recommendations = [pivot_table.index[i[0]] for i in similar_items]
        return recommendations
    except IndexError:
        return []

def render_book_card(book, show_rating=True):
    st.markdown('<div class="book-card">', unsafe_allow_html=True)
    cols = st.columns([1, 2], gap='small')
    with cols[0]:
        cover_url = book.get('Cover-URL') or book.get('Image-URL-M')
        if cover_url:
            st.image(cover_url, width='stretch')
        else:
            st.image('https://via.placeholder.com/150x200?text=No+Image', width='stretch')
    with cols[1]:
        st.markdown(f"<div class='book-title'>{book['Book-Title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='book-author'>by {book['Book-Author']}</div>", unsafe_allow_html=True)
        if show_rating and 'Avg-Rating' in book:
            st.markdown(f"<div class='book-meta'>⭐ {book['Avg-Rating']:.2f} · {int(book['Num-Ratings'])} ratings</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    popular_books_df, all_books_df, pivot_table, sim_score = load_data()

    st.title('📚 Book Recommender System')
    st.write('Find your next great read!')

    home_tab, search_tab = st.tabs(['Home', 'Search'])

    with home_tab:
        st.subheader('Top 50 Books')
        st.write('Browse the top-rated books.')
        top_50 = popular_books_df.nlargest(50, 'Avg-Rating').reset_index(drop=True)
        for i in range(0, len(top_50), 3):
            cols = st.columns(3, gap='small')
            for col, (_, book) in zip(cols, top_50.iloc[i:i+3].iterrows()):
                with col:
                    render_book_card(book)

    with search_tab:
        st.subheader('Search Books & Recommendations')
        st.write('Search by title or author, then get recommendations from similar books.')

        query = st.text_input('Search by title or author', '')
        num_recommendations = st.slider('Recommendations to show', 1, 10, 5)

        if query:
            query_lower = query.strip().lower()
            search_results = all_books_df[
                all_books_df['Search-Text'].str.contains(query_lower, na=False, regex=False)
            ].reset_index(drop=True)
        else:
            search_results = all_books_df.head(12).reset_index(drop=True)

        display_results = search_results.head(12).reset_index(drop=True)

        st.markdown('### Search Results')
        if search_results.empty:
            st.info('No books matched your search. Try a different title or author.')
        else:
            st.write(f'Found {len(search_results)} book(s). Showing top {len(display_results)} results.')
            for i in range(0, len(display_results), 3):
                cols = st.columns(3, gap='small')
                for col, (_, book) in zip(cols, display_results.iloc[i:i+3].iterrows()):
                    with col:
                        render_book_card(book, show_rating=False)

        recommended_search_books = search_results[search_results['Book-Title'].isin(pivot_table.index)].copy()
        available_books = sorted(recommended_search_books['Book-Title'].unique())
        if len(available_books) > 50:
            st.info('Showing the first 50 matching titles for selection to keep the search responsive.')
            available_books = available_books[:50]

        if available_books:
            st.markdown('### Choose a book for recommendations')
            selected_book = st.selectbox('Select a book', available_books)
            selected_book_data = all_books_df[all_books_df['Book-Title'] == selected_book].iloc[0]
            st.markdown('### Selected Book')
            render_book_card(selected_book_data, show_rating=False)
        else:
            st.warning('No searchable books with recommendation data were found. Showing all available recommendation titles.')
            available_books = sorted(pivot_table.index.tolist())
            selected_book = st.selectbox('Select a book', available_books)
            selected_book_data = popular_books_df[popular_books_df['Book-Title'] == selected_book]
            if not selected_book_data.empty:
                st.markdown('### Selected Book')
                render_book_card(selected_book_data.iloc[0], show_rating=False)

        if st.button('Show Recommendations', type='primary'):
            rec_titles = get_recommendations(selected_book, pivot_table, sim_score, num_recommendations)
            if rec_titles:
                rec_books = popular_books_df[popular_books_df['Book-Title'].isin(rec_titles)].reset_index(drop=True)
                st.markdown('### Recommended for you')
                for i in range(0, len(rec_books), 2):
                    cols = st.columns(2, gap='small')
                    for col, (_, book) in zip(cols, rec_books.iloc[i:i+2].iterrows()):
                        with col:
                            render_book_card(book)
            else:
                st.warning('No recommendations available for this book. Please select a different title.')

if __name__ == '__main__':
    main()
