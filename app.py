import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# --- CONFIGURE PAGE ---
st.set_page_config(page_title="VoC Dashboard (Air France Ed.)", page_icon="✈️", layout="wide")

# --- 1. SETUP & CACHING ---
@st.cache_resource
def load_vader():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_vader()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Functions to detect conversation topic
def categorize_review(text):
    text = str(text).lower()
    
    # Keyword Dictionary
    keywords = {
        'Food & Drinks': ['food', 'meal', 'drink', 'beverage', 'snack', 'catering', 'water', 'coffee', 'tea'],
        'Seat & Comfort': ['seat', 'legroom', 'space', 'comfort', 'cabin', 'sleep', 'chair'],
        'Service & Staff': ['crew', 'staff', 'attendant', 'service', 'hostess', 'rude', 'polite', 'helpful'],
        'Punctuality': ['delay', 'late', 'time', 'schedule', 'cancel', 'wait', 'hour'],
        'Baggage': ['bag', 'luggage', 'suitcase', 'lost', 'claim', 'damaged'],
        'Entertainment': ['movie', 'screen', 'entertainment', 'wifi', 'film', 'music']
    }
    
    # Check every categories
    for category, key_list in keywords.items():
        if any(word in text for word in key_list):
            return category
            
    return 'General/Others' # If no matching category found
# --- 2. SIDEBAR (Business Parameter & Configuration) ---
with st.sidebar:
    st.header("💰 Business Parameter")
    st.info("Set this parameter to estimate revenue loss.")
    ticket_price = st.number_input("Product Price (€)", min_value=0, value=250, step=10) # Adjusted for Flight Ticket
    churn_rate = st.slider("Churn Estimation (%)", 0, 100, 20) / 100
    
    st.divider()
    
    # --- DYNAMIC FILTERING ---
    st.subheader("⚙️ Analysis Configuration")
    # Default keywords that can be adjusted
    default_ignore = "flight, plane, airline, air france, trip, fly, review, travel, passengers"
    
    ignore_text = st.text_area(
        "List of Ignored Words/Phrases (comma-separated)", 
        value=default_ignore,
        help="These words/phrases will be removed from the Root Cause Analysis graph."
    )
    custom_ignore = [x.strip() for x in ignore_text.split(',')]
    
    st.divider()
    st.caption("ℹ️ File upload & Analyze button is on the main page.")

# --- 3. MAIN APP LOGIC ---
st.title("✈️ Air France VoC: Analytics Dashboard")
st.markdown("This dashboard uses **Hybrid Method (Title + Text)** to obtain customer sentiment more accurately.")

setup_container = st.container()

with setup_container:
    uploaded_file = st.file_uploader("📂 Step 1: Upload a .csv file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        with st.expander("⚙️ Step 2: Configure the Columns", expanded=True):
            st.info("Select the TITLE column dan REVIEW TEXT column to combine both.")
            
            all_cols = df.columns.tolist()
            
            # Automatically finds default index if it as a match with the column name
            idx_title = all_cols.index('title') if 'title' in all_cols else 0
            idx_text = all_cols.index('text') if 'text' in all_cols else (1 if len(all_cols) > 1 else 0)
            idx_date = all_cols.index('published date') if 'published date' in all_cols else (2 if len(all_cols) > 2 else 0)

            c1, c2, c3 = st.columns(3)
            
            with c1:
                col_title = st.selectbox("Select the TITLE column", all_cols, index=idx_title)
            with c2:
                col_text = st.selectbox("Select the REVIEW TEXT column", all_cols, index=idx_text)
            with c3:
                col_date = st.selectbox("Select the DATE/TIME column", all_cols, index=idx_date)
            
            st.write("") 
            run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner('In process of Feature Engineering & Analysis...'):
                
                # 1. Feature Engineering: Combines Title + Text with separator " . "
                df['Combined_Raw'] = df[col_title].astype(str) + " . " + df[col_text].astype(str)
                
                # 2. Cleaning Data (Using the combined column)
                df = df.dropna(subset=['Combined_Raw'])
                df['Cleaned_Text'] = df['Combined_Raw'].apply(clean_text)
                
                # Handling Date/Time
                df['Fixed_Date'] = pd.to_datetime(df[col_date], errors='coerce')
                df = df.dropna(subset=['Fixed_Date']).sort_values('Fixed_Date')
                df['Date_Only'] = df['Fixed_Date'].dt.date

                # 3. Counting Sentiments of the Combined Text with VADER
                def get_sentiment(text):
                    s = sia.polarity_scores(text)['compound']
                    if s >= 0.05: return 'Positive'
                    elif s <= -0.05: return 'Negative'
                    else: return 'Neutral'
                
                df['Sentiment'] = df['Cleaned_Text'].apply(get_sentiment)

                #Apply Categorization
                df['Category'] = df['Cleaned_Text'].apply(categorize_review)

                # --- 4. DASHBOARD VISUALIZATION ---
                st.divider()
                st.toast("Analysis Completed! Title & Text have been combined.", icon="✅")

                # ROW 1: METRICS
                total_neg = len(df[df['Sentiment'] == 'Negative'])
                potential_loss = total_neg * ticket_price * churn_rate
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews", len(df))
                col2.metric("Negative Reviews", f"{total_neg} user", delta_color="inverse")
                col3.metric("⚠️ Risk of Revenue Loss", f"€ {potential_loss:,.0f}", help="Negative x Price x Churn")

                st.divider()

                # ROW 2: TIME SERIES ANALYSIS
                st.subheader("📈 Daily Sentiment Trend")
                daily_trend = df.groupby(['Date_Only', 'Sentiment']).size().unstack(fill_value=0)
                
                color_map = {'Negative': '#e74c3c', 'Neutral': '#95a5a6', 'Positive': '#2ecc71'}
                existing_cols = daily_trend.columns.tolist()
                final_colors = [color_map[col] for col in existing_cols if col in color_map]
                
                st.line_chart(daily_trend, color=final_colors)

                # ROW 3: ROOT CAUSE ANALYSIS
                st.subheader("🔍 Root Cause Analysis")
                st.caption("Analyzed from the combined Title + Text data")
                
                col_left, col_right = st.columns(2)

                def plot_bigram_improved(data, title, color_palette, ignore_list):
                    if len(data) < 1: return None
                    
                    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(data)
                    bag_of_words = vec.transform(data)
                    sum_words = bag_of_words.sum(axis=0) 
                    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                    
                    # FILTER TRASH
                    words_freq = [w for w in words_freq if w[0] not in ignore_list]
                    
                    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:10]
                    if not words_freq: return None

                    x, y = zip(*words_freq)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.barplot(x=list(y), y=list(x), palette=color_palette, ax=ax)
                    ax.set_title(title)
                    return fig

                with col_left:
                    neg_reviews = df[df['Sentiment'] == 'Negative']['Cleaned_Text']
                    if not neg_reviews.empty:
                        fig_neg = plot_bigram_improved(neg_reviews, "Top Negative Issues", "Reds_r", custom_ignore)
                        if fig_neg: st.pyplot(fig_neg)
                    else:
                        st.info("No negative reviews.")

                with col_right:
                    pos_reviews = df[df['Sentiment'] == 'Positive']['Cleaned_Text']
                    if not pos_reviews.empty:
                        fig_pos = plot_bigram_improved(pos_reviews, "Top Liked Features", "Greens_r", custom_ignore)
                        if fig_pos: st.pyplot(fig_pos)
                    else:
                        st.info("No positive reviews.")

                # ROW 4: CATEGORIZATION BY TOPIC
                st.subheader("📊 Sentiment Distribution by Topic")
                st.caption("Lihat aspek mana yang paling banyak mengecewakan pelanggan.")

                # Hitung data untuk grafik
                category_sentiment = df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')

                chart = alt.Chart(category_sentiment).mark_bar().encode(
                    x=alt.X('Category', sort='-y'), # Urutkan dari yang terbanyak
                    y='Count',
                    color=alt.Color('Sentiment', scale={'domain': ['Negative', 'Neutral', 'Positive'], 'range': ['#e74c3c', '#95a5a6', '#2ecc71']}),
                    tooltip=['Category', 'Sentiment', 'Count']
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
                
                # ROW 5: RAW DATA
                with st.expander("See raw data (including the combined text column)"):
                    st.dataframe(df[[col_date, col_title, col_text, 'Combined_Raw', 'Sentiment']].head(50))

    except Exception as e:
        st.error(f"There was an error while processing the file: {e}")

else:

    st.info("👆 Upload your .csv file to begin.")


