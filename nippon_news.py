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
ARTICLES_PER_REFRESH = 10

# --------------------------
# Initialize session state
# --------------------------
if 'all_articles' not in st.session_state:
    st.session_state.all_articles = []

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
def fetch_news(num_articles=10):
    all_articles = []
    seen_titles = {article['Title'] for article in st.session_state.all_articles}
    
    for term in SEARCH_TERMS:
        try:
            url = f"https://news.google.com/rss/search?q={term}+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                if entry.title not in seen_titles:
                    all_articles.append(entry)
                    seen_titles.add(entry.title)
                
                if len(all_articles) >= num_articles:
                    break
        except Exception:
            continue
        
        if len(all_articles) >= num_articles:
            break
    
    return all_articles[:num_articles]

def process_articles(articles):
    records = []
    
    for art in articles:
        title = art.title
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
    
    return records

# --------------------------
# Streamlit App
# --------------------------
st.title("📈 Finance News Dashboard")
st.markdown("---")

# Refresh button
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Refresh News", type="primary"):
        with st.spinner("Fetching latest finance news..."):
            new_articles = fetch_news(ARTICLES_PER_REFRESH)
            if new_articles:
                processed = process_articles(new_articles)
                # Add new articles to the top
                st.session_state.all_articles = processed + st.session_state.all_articles
                st.success(f"Added {len(processed)} new articles!")
            else:
                st.info("No new articles found.")

# Load initial articles if empty
if not st.session_state.all_articles:
    with st.spinner("Loading initial articles..."):
        initial_articles = fetch_news(ARTICLES_PER_REFRESH)
        if initial_articles:
            st.session_state.all_articles = process_articles(initial_articles)

# Display content
if st.session_state.all_articles:
    df = pd.DataFrame(st.session_state.all_articles)
    
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
        title="Overall Sentiment Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # News articles
    st.subheader("📰 Latest News Articles")
    
    # Display articles
    for idx, row in df.iterrows():
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

else:
    st.info("Click 'Refresh News' to load articles.")
