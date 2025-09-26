import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


st.set_page_config(page_title="Amazon Recommender", layout="wide")
st.title("Amazon Recommendation System")
st.caption("Simple item-based and user-based recommendations built from ratings data.")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), 'ratings_Electronics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    # File has no header; assign names explicitly to match the notebook
    df = pd.read_csv(csv_path, header=None, names=['user_id', 'product_id', 'rating', 'timestamp'])

    # Ensure types
    df['user_id'] = df['user_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating']).reset_index(drop=True)

    # Keep only users with > 100 ratings (same as notebook)
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].map(user_counts) > 100].reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def build_models(df: pd.DataFrame):
    # Encode IDs to categorical integer indices
    user_cats = df['user_id'].astype('category')
    product_cats = df['product_id'].astype('category')
    user_index = user_cats.cat.codes.values
    product_index = product_cats.cat.codes.values
    ratings_values = df['rating'].astype(np.float32).values

    num_users = user_cats.cat.categories.size
    num_items = product_cats.cat.categories.size

    uim_sparse = csr_matrix((ratings_values, (user_index, product_index)), shape=(num_users, num_items))

    # Fit KNN models
    item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    item_knn.fit(uim_sparse.T)

    user_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    user_knn.fit(uim_sparse)

    # Mappings
    product_id_to_col = {pid: idx for idx, pid in enumerate(product_cats.cat.categories)}
    col_to_product_id = {idx: pid for pid, idx in product_id_to_col.items()}
    user_id_to_row = {uid: idx for idx, uid in enumerate(user_cats.cat.categories)}
    row_to_user_id = {idx: uid for uid, idx in user_id_to_row.items()}

    # Density for info
    density = (uim_sparse.nnz / (uim_sparse.shape[0] * uim_sparse.shape[1])) * 100

    return {
        'uim': uim_sparse,
        'item_knn': item_knn,
        'user_knn': user_knn,
        'product_id_to_col': product_id_to_col,
        'col_to_product_id': col_to_product_id,
        'user_id_to_row': user_id_to_row,
        'row_to_user_id': row_to_user_id,
        'num_users': num_users,
        'num_items': num_items,
        'density': density,
    }


def recommend_similar_products(models, query_product_id, top_k=10) -> pd.DataFrame:
    item_knn = models['item_knn']
    uim = models['uim']
    product_id_to_col = models['product_id_to_col']
    col_to_product_id = models['col_to_product_id']

    if query_product_id not in product_id_to_col:
        return pd.DataFrame(columns=['product_id', 'similarity'])

    qcol = product_id_to_col[query_product_id]
    distances, indices = item_knn.kneighbors(uim.T[qcol], n_neighbors=top_k + 1)
    indices = indices.ravel()[1:]
    distances = distances.ravel()[1:]
    return pd.DataFrame({
        'product_id': [col_to_product_id[i] for i in indices],
        'similarity': 1.0 - distances
    })


def recommend_products_for_user(models, query_user_id, top_k=10, neighbor_k=50) -> pd.DataFrame:
    user_knn = models['user_knn']
    uim = models['uim']
    user_id_to_row = models['user_id_to_row']
    col_to_product_id = models['col_to_product_id']

    if query_user_id not in user_id_to_row:
        return pd.DataFrame(columns=['product_id', 'score'])

    qrow = user_id_to_row[query_user_id]
    n_neighbors = min(neighbor_k + 1, uim.shape[0])
    distances, indices = user_knn.kneighbors(uim[qrow], n_neighbors=n_neighbors)
    indices = indices.ravel()
    distances = distances.ravel()

    mask = indices != qrow
    neighbor_indices = indices[mask][:neighbor_k]
    neighbor_distances = distances[mask][:neighbor_k]
    similarities = 1.0 - neighbor_distances

    if neighbor_indices.size == 0:
        return pd.DataFrame(columns=['product_id', 'score'])

    neighbor_ratings = uim[neighbor_indices]
    weighted_scores = similarities @ neighbor_ratings.toarray()

    user_rated_mask = uim[qrow].toarray().ravel() > 0
    weighted_scores[user_rated_mask] = -np.inf

    if np.all(~np.isfinite(weighted_scores)):
        return pd.DataFrame(columns=['product_id', 'score'])

    k = min(top_k, len(weighted_scores) - 1)
    top_cols = np.argpartition(-weighted_scores, kth=k)[:top_k]
    top_cols = top_cols[np.argsort(-weighted_scores[top_cols])]

    return pd.DataFrame({
        'product_id': [col_to_product_id[c] for c in top_cols],
        'score': weighted_scores[top_cols]
    })


with st.sidebar:
    st.subheader("Configuration")
    top_k = st.number_input("Results (top_k)", min_value=1, max_value=50, value=10, step=1)
    neighbor_k = st.number_input("Neighbors (user-based)", min_value=5, max_value=200, value=50, step=5)


try:
    df = load_data()
    models = build_models(df)
    st.success(f"Loaded {len(df)} ratings | Users: {models['num_users']} | Items: {models['num_items']} | Density: {models['density']:.4f}%")

    tab1, tab2 = st.tabs(["Recommend by Product", "Recommend for User"])

    with tab1:
        st.subheader("Product → Similar Products")
        product_ids = df['product_id'].drop_duplicates().head(5000).tolist()
        pid = st.selectbox("Select a product_id", product_ids)
        if st.button("Find similar products", key="by_product"):
            recs = recommend_similar_products(models, pid, top_k=int(top_k))
            st.dataframe(recs, width='stretch')

    with tab2:
        st.subheader("User → Recommended Products")
        user_ids = df['user_id'].drop_duplicates().head(5000).tolist()
        uid = st.selectbox("Select a user_id", user_ids)
        if st.button("Recommend for user", key="by_user"):
            recs = recommend_products_for_user(models, uid, top_k=int(top_k), neighbor_k=int(neighbor_k))
            st.dataframe(recs, width='stretch')

except Exception as e:
    st.error(str(e))
