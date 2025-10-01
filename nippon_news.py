import streamlit as st
import feedparser
from transformers import pipeline
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Finance News Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# Config
# --------------------------
SEARCH_TERMS = [
    "NSE", "BSE", "Sensex", "Nifty", "stock market", "IPO", "RBI",
    "banking", "earnings", "quarterly results", "dividend", "merger",
    "acquisition", "sales", "revenue", "inflation", "rate"
]
MAX_TOTAL_RESULTS = 25

# --------------------------
# Cache FinBERT model
# --------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

finbert = load_model()

# --------------------------
# Functions
# --------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_news():
    all_articles = []
    for term in SEARCH_TERMS:
        try:
            url = f"https://news.google.com/rss/search?q={term}+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            all_articles.extend(feed.entries)
        except Exception:
            continue
        if len(all_articles) >= MAX_TOTAL_RESULTS:
            all_articles = all_articles[:MAX_TOTAL_RESULTS]
            break
    return all_articles

def process_articles(articles):
    seen_titles = set()
    records = []
    
    for art in articles:
        title = art.title
        if title in seen_titles:
            continue
        seen_titles.add(title)
        
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        url = art.link
        
        # Get sentiment
        sentiment = finbert(title[:512])[0]["label"]
        
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Link": url
        })
    
    return pd.DataFrame(records)

# --------------------------
# Streamlit App
# --------------------------
st.title("📈 Finance News Dashboard")
st.markdown("---")

# Refresh button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Refresh News", type="primary"):
        st.cache_data.clear()
        st.rerun()

# Fetch and process news
with st.spinner("Fetching latest finance news..."):
    articles = fetch_news()
    df = process_articles(articles)

if df.empty:
    st.warning("No new articles found.")
else:
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(df))
    
    with col2:
        positive_count = len(df[df['Sentiment'] == 'positive'])
        st.metric("Positive", positive_count)
    
    with col3:
        neutral_count = len(df[df['Sentiment'] == 'neutral'])
        st.metric("Neutral", neutral_count)
    
    with col4:
        negative_count = len(df[df['Sentiment'] == 'negative'])
        st.metric("Negative", negative_count)
    
    st.markdown("---")
    
    # Sentiment chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={
            "positive": "green",
            "neutral": "gray",
            "negative": "red"
        },
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # News table with clickable links
    st.subheader("📰 Latest News Articles")
    
    # Add filter
    sentiment_filter = st.multiselect(
        "Filter by Sentiment",
        options=["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"]
    )
    
    filtered_df = df[df['Sentiment'].isin(sentiment_filter)]
    
    # Display articles
    for idx, row in filtered_df.iterrows():
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"**[{row['Title']}]({row['Link']})**")
                st.caption(f"Source: {row['Source']}")
            
            with col2:
                sentiment_color = {
                    "positive": "🟢",
                    "neutral": "⚪",
                    "negative": "🔴"
                }
                st.markdown(f"### {sentiment_color.get(row['Sentiment'], '⚪')} {row['Sentiment'].title()}")
            
            st.markdown("---")
    
    # Download data
    st.subheader("💾 Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="finance_news.csv",
        mime="text/csv"
    )
