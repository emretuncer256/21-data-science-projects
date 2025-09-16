import io
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


def load_data(file_buffer: io.BytesIO | None) -> pd.DataFrame:
    if file_buffer is None:
        df = pd.read_csv("rfm_data.csv")
    else:
        df = pd.read_csv(file_buffer)
    return df


def compute_rfm(transactions: pd.DataFrame, analysis_date: datetime) -> pd.DataFrame:
    data = transactions.copy()
    data["PurchaseDate"] = pd.to_datetime(data["PurchaseDate"], errors="coerce")

    rfm = (
        data.groupby("CustomerID").agg(
            {
                "PurchaseDate": lambda x: (analysis_date - x.max()).days,
                "OrderID": "nunique",
                "TransactionAmount": "sum",
            }
        )
    )
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Scores (quantiles). For Frequency use rank to avoid qcut ties issue
    rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["FrequencyScore"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    rfm["MonetaryScore"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    # RF segmentation map
    segmentation_map = {
        r"[1-2][1-2]": "Hibernating",
        r"[1-2][3-4]": "At_Risk",
        r"[1-2]5": "Cant_Lose",
        r"3[1-2]": "About_to_Sleep",
        r"33": "Need_Attention",
        r"[3-4][4-5]": "Loyal_Customers",
        r"41": "Promising",
        r"51": "New_Customers",
        r"[4-5][2-3]": "Potential_Loyalists",
        r"5[4-5]": "Champions",
    }

    rfm["RF_Score"] = rfm["RecencyScore"].astype(str) + rfm["FrequencyScore"].astype(str)
    rfm["Segment"] = rfm["RF_Score"].replace(segmentation_map, regex=True)
    return rfm


def render_dashboard(rfm: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{len(rfm):,}")
    with col2:
        st.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.1f}")
    with col3:
        st.metric("Avg Frequency", f"{rfm['Frequency'].mean():.2f}")
    with col4:
        st.metric("Total Monetary", f"{rfm['Monetary'].sum():,.0f}")

    st.markdown("---")

    # Segment count bar chart
    counts = (
        rfm["Segment"].value_counts().rename_axis("Segment").reset_index(name="Count")
    )
    fig_bar = px.bar(
        counts,
        x="Segment",
        y="Count",
        color="Segment",
        title="Number of Customers by Segment",
    )
    fig_bar.update_layout(xaxis_tickangle=-30, legend_title_text="Segment")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: Recency vs Monetary colored by Segment
    fig_scatter = px.scatter(
        rfm.reset_index(),
        x="Recency",
        y="Monetary",
        color="Segment",
        size="Frequency",
        hover_data=[rfm.index, "Frequency"],
        title="Customer Segments by Recency and Monetary"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Treemap
    fig_tree = px.treemap(
        rfm.reset_index(),
        path=["Segment", "RecencyScore", "FrequencyScore"],
        values="Monetary",
        color="Segment",
        title="Monetary Distribution by Segment and RF Scores",
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

    # Segment filter and table
    segments = ["All"] + sorted(rfm["Segment"].dropna().unique().tolist())
    sel = st.selectbox("Filter by segment", options=segments, index=0)
    view = rfm if sel == "All" else rfm[rfm["Segment"] == sel]
    st.dataframe(view.reset_index().rename(columns={"index": "CustomerID"}))

    # Download
    csv_bytes = view.reset_index().rename(columns={"index": "CustomerID"}).to_csv(
        index=False
    ).encode("utf-8")
    st.download_button(
        "Download current view (CSV)",
        data=csv_bytes,
        file_name=f"rfm_segments_{sel.lower()}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Customer RFM Analysis", layout="wide")
    st.title("Customer RFM Analysis — Clustering")
    st.caption(
        "Compute Recency, Frequency, Monetary features and segment customers via RF scoring and visual analytics."
    )

    with st.sidebar:
        st.header("Inputs")
        uploaded = st.file_uploader("Upload transactions CSV", type=["csv"]) 
        st.markdown(
            "Expected columns: `CustomerID`, `OrderID`, `TransactionAmount`, `PurchaseDate`."
        )

        use_latest = st.checkbox(
            "Use 1 day after latest purchase as analysis date", value=True
        )
        if use_latest:
            # Will compute after loading data
            analysis_date = None
        else:
            selected_date = st.date_input("Analysis date", value=datetime.now().date())
            analysis_date = datetime.combine(selected_date, datetime.min.time())

    # Load
    df = load_data(uploaded)
    if analysis_date is None:
        # infer from data
        max_date = pd.to_datetime(df["PurchaseDate"]).max()
        analysis_date = (max_date + timedelta(days=1)).to_pydatetime()

    st.subheader("Preview")
    st.write(df.head())

    # Compute RFM
    rfm = compute_rfm(df, analysis_date)

    st.subheader("RFM Summary")
    st.write(
        pd.DataFrame(
            {
                "Analysis Date": [analysis_date.strftime("%Y-%m-%d")],
                "Customers": [len(rfm)],
                "Avg Recency": [rfm["Recency"].mean()],
                "Avg Frequency": [rfm["Frequency"].mean()],
                "Total Monetary": [rfm["Monetary"].sum()],
            }
        )
    )

    render_dashboard(rfm)


if __name__ == "__main__":
    main()


