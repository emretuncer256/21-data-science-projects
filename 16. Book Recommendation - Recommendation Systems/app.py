import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('darkgrid')

# Page configuration
st.set_page_config(
    page_title="Most Popular Books",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .book-card {
        background-color: #f8f9fa;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .book-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .book-author {
        color: #7f8c8d;
        font-style: italic;
        margin-bottom: 0.3rem;
    }
    .book-rating {
        color: #f39c12;
        font-weight: bold;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .popular-book {
        background-color: #fff3cd;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the books dataset"""
    try:
        df = pd.read_csv('books.csv', index_col=0, on_bad_lines='skip')
        
        # Clean the data
        df = df.dropna(subset=['title', 'authors', 'average_rating'])
        df = df[df['average_rating'] > 0]
        df = df[df['ratings_count'] > 0]
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess data similar to notebook approach"""
    # Group books by title and aggregate ratings (same as notebook)
    bg = df.groupby('title').agg({
        'ratings_count': ['size', 'sum', 'mean'],
        'average_rating': 'mean',
        'authors': 'first',
        'num_pages': 'mean',
        'language_code': 'first'
    })
    
    # Flatten column names
    bg.columns = ['_'.join(col).strip() for col in bg.columns.values]
    bg = bg.reset_index()
    
    # Calculate percentage of total ratings (same as notebook)
    total_ratings = bg['ratings_count_sum'].sum()
    bg['percentage'] = bg['ratings_count_sum'].div(total_ratings) * 100
    
    # Create popularity rank (same as notebook)
    bg['rank'] = bg['percentage'].rank(ascending=False)
    bg = bg.sort_values(by='rank', ascending=True)
    
    return bg

def display_popular_book(book, index):
    """Display a popular book card with information"""
    st.markdown(f"""
    <div class="popular-book">
        <div class="book-title">{index + 1}. {book['title']}</div>
        <div class="book-author">by {book['authors_first']}</div>
        <div class="book-rating">⭐ {book['average_rating_mean']:.2f} ({book['ratings_count_sum']:,} ratings)</div>
        <div>📖 {book['num_pages_mean']:.0f} pages | 🌐 {book['language_code_first']} | 📊 {book['percentage']:.2f}% of total ratings</div>
    </div>
    """, unsafe_allow_html=True)

def display_book_info(book):
    """Display detailed book information"""
    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">{book['title']}</div>
        <div class="book-author">by {book['authors_first']}</div>
        <div class="book-rating">⭐ {book['average_rating_mean']:.2f} ({book['ratings_count_sum']:,} ratings)</div>
        <div>📖 {book['num_pages_mean']:.0f} pages | 🌐 {book['language_code_first']}</div>
        <div>📊 Popularity: #{int(book['rank'])} | {book['percentage']:.2f}% of total ratings</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Most Popular Books</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading books data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if 'books.csv' exists in the current directory.")
        return
    
    # Preprocess data
    with st.spinner("Processing data..."):
        processed_df = preprocess_data(df)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Number of books to display
    n_books = st.sidebar.slider("Number of books to display:", 5, 50, 10)
    
    # Main content - Most Popular Books
    st.subheader(f"🏆 Top {n_books} Most Popular Books (Based on Total Ratings)")
    st.info("💡 These are the most popular books ranked by the percentage of total ratings they received.")
    
    # Display top N most popular books
    top_books = processed_df.head(n_books)
    
    for idx, (_, book) in enumerate(top_books.iterrows()):
        display_popular_book(book, idx)
    
    # Statistics section
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", f"{len(processed_df):,}")
    with col2:
        st.metric("Total Ratings", f"{processed_df['ratings_count_sum'].sum():,}")
    with col3:
        st.metric("Avg Rating", f"{processed_df['average_rating_mean'].mean():.2f}")
    with col4:
        st.metric("Avg Pages", f"{processed_df['num_pages_mean'].mean():.0f}")
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Data Visualizations")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Rating distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['average_rating'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Number of Books')
        ax.set_title('Distribution of Book Ratings')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col6:
        # Rating vs Ratings Count
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['average_rating'], df['ratings_count'], 
                           alpha=0.6, s=30, c=df['ratings_count'], cmap='viridis')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Number of Ratings')
        ax.set_title('Rating vs Popularity')
        ax.set_yscale('log')
        plt.colorbar(scatter, ax=ax, label='Ratings Count')
        st.pyplot(fig)
    
    # Top authors and languages
    col7, col8 = st.columns(2)
    
    with col7:
        st.subheader("👥 Top Authors")
        top_authors = df.groupby('authors')['title'].count().sort_values(ascending=False).head(10)
        for author, count in top_authors.items():
            st.write(f"• {author}: {count} books")
    
    with col8:
        st.subheader("🌐 Languages")
        lang_dist = df['language_code'].value_counts().head(10)
        for lang, count in lang_dist.items():
            st.write(f"• {lang}: {count} books")
    
    # Book information lookup
    st.markdown("---")
    st.subheader("🔍 Book Information Lookup")
    st.write("Select a book to view detailed information:")
    
    # Book selection dropdown
    book_titles = sorted(processed_df['title'].tolist())
    selected_book = st.selectbox(
        "Choose a book:",
        book_titles,
        index=0,
        key="book_selector"
    )
    
    # Display selected book information
    if selected_book:
        selected_book_data = processed_df[processed_df['title'] == selected_book].iloc[0]
        display_book_info(selected_book_data)

if __name__ == "__main__":
    main()
