import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# --- KONFIGURASI HALAMAN (Wajib di baris pertama) ---
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

# --- 2. SIDEBAR (Parameter Bisnis & Konfigurasi) ---
with st.sidebar:
    st.header("💰 Simulasi Bisnis")
    st.info("Atur parameter ini untuk estimasi kerugian.")
    ticket_price = st.number_input("Harga Produk (Rp)", min_value=0, value=5000000, step=100000) # Adjusted for Flight Ticket
    churn_rate = st.slider("Estimasi Churn (%)", 0, 100, 20) / 100
    
    st.divider()
    
    # --- DYNAMIC FILTERING ---
    st.subheader("⚙️ Konfigurasi Analisis")
    # Default keywords disesuaikan untuk penerbangan
    default_ignore = "flight, plane, airline, air france, trip, fly, review, travel, passengers"
    
    ignore_text = st.text_area(
        "Daftar Kata/Frasa Diabaikan (pisahkan koma)", 
        value=default_ignore,
        help="Kata-kata ini akan dihapus dari grafik Root Cause Analysis."
    )
    custom_ignore = [x.strip() for x in ignore_text.split(',')]
    
    st.divider()
    st.caption("ℹ️ Upload file & Tombol Analisis ada di halaman utama.")

# --- 3. MAIN APP LOGIC ---
st.title("✈️ Air France VoC: Analytics Dashboard")
st.markdown("Dashboard ini menggunakan **Metode Hybrid (Judul + Teks)** untuk menangkap sentimen pelanggan dengan lebih akurat.")

setup_container = st.container()

with setup_container:
    uploaded_file = st.file_uploader("📂 Langkah 1: Upload Data CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        with st.expander("⚙️ Langkah 2: Konfigurasi Kolom (Strategi Anchor)", expanded=True):
            st.info("Pilih kolom JUDUL dan ISI REVIEW untuk digabungkan.")
            
            all_cols = df.columns.tolist()
            
            # Mencari index default secara otomatis jika nama kolom cocok
            idx_title = all_cols.index('title') if 'title' in all_cols else 0
            idx_text = all_cols.index('text') if 'text' in all_cols else (1 if len(all_cols) > 1 else 0)
            idx_date = all_cols.index('published date') if 'published date' in all_cols else (2 if len(all_cols) > 2 else 0)

            c1, c2, c3 = st.columns(3)
            
            with c1:
                col_title = st.selectbox("Pilih Kolom Judul (Title)", all_cols, index=idx_title)
            with c2:
                col_text = st.selectbox("Pilih Kolom Isi (Text)", all_cols, index=idx_text)
            with c3:
                col_date = st.selectbox("Pilih Kolom Tanggal", all_cols, index=idx_date)
            
            st.write("") 
            run_btn = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner('Sedang melakukan Feature Engineering & Analisis...'):
                
                # --- [PERUBAHAN UTAMA DI SINI] ---
                # 1. Feature Engineering: Concatenation Strategy
                # Kita gabungkan Title + Text dengan separator " . "
                df['Combined_Raw'] = df[col_title].astype(str) + " . " + df[col_text].astype(str)
                
                # 2. Cleaning Data (Menggunakan kolom gabungan tadi)
                df = df.dropna(subset=['Combined_Raw'])
                df['Cleaned_Text'] = df['Combined_Raw'].apply(clean_text)
                
                # Handling Tanggal
                df['Tanggal_Fix'] = pd.to_datetime(df[col_date], errors='coerce')
                df = df.dropna(subset=['Tanggal_Fix']).sort_values('Tanggal_Fix')
                df['Tanggal_Only'] = df['Tanggal_Fix'].dt.date

                # 3. Hitung Sentimen (VADER pada teks gabungan)
                def get_sentiment(text):
                    s = sia.polarity_scores(text)['compound']
                    if s >= 0.05: return 'Positif'
                    elif s <= -0.05: return 'Negatif'
                    else: return 'Netral'
                
                df['Sentiment'] = df['Cleaned_Text'].apply(get_sentiment)

                # --- 4. DASHBOARD VISUALIZATION (Logic Visualisasi Tetap Sama) ---
                st.divider()
                st.toast("Analisis Selesai! Data Judul & Teks telah digabungkan.", icon="✅")

                # ROW 1: METRICS
                total_neg = len(df[df['Sentiment'] == 'Negatif'])
                potential_loss = total_neg * ticket_price * churn_rate
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Ulasan", len(df))
                col2.metric("User Kecewa (Negatif)", f"{total_neg} user", delta_color="inverse")
                col3.metric("⚠️ Potensi Kerugian (Risk)", f"Rp {potential_loss:,.0f}", help="Negatif x Harga x Churn")

                st.divider()

                # ROW 2: TIME SERIES
                st.subheader("📈 Tren Sentimen Harian")
                daily_trend = df.groupby(['Tanggal_Only', 'Sentiment']).size().unstack(fill_value=0)
                
                color_map = {'Negatif': '#e74c3c', 'Netral': '#95a5a6', 'Positif': '#2ecc71'}
                existing_cols = daily_trend.columns.tolist()
                final_colors = [color_map[col] for col in existing_cols if col in color_map]
                
                st.line_chart(daily_trend, color=final_colors)

                # ROW 3: ROOT CAUSE (Updated Logic)
                st.subheader("🔍 Akar Masalah (Root Cause Analysis)")
                st.caption("Dianalisis dari gabungan Judul Review + Isi Review")
                
                col_left, col_right = st.columns(2)

                def plot_bigram_improved(data, title, color_palette, ignore_list):
                    if len(data) < 1: return None
                    
                    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(data)
                    bag_of_words = vec.transform(data)
                    sum_words = bag_of_words.sum(axis=0) 
                    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                    
                    # FILTER SAMPAH
                    words_freq = [w for w in words_freq if w[0] not in ignore_list]
                    
                    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:10]
                    if not words_freq: return None

                    x, y = zip(*words_freq)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.barplot(x=list(y), y=list(x), palette=color_palette, ax=ax)
                    ax.set_title(title)
                    return fig

                with col_left:
                    neg_reviews = df[df['Sentiment'] == 'Negatif']['Cleaned_Text']
                    if not neg_reviews.empty:
                        fig_neg = plot_bigram_improved(neg_reviews, "Top Isu Negatif", "Reds_r", custom_ignore)
                        if fig_neg: st.pyplot(fig_neg)
                    else:
                        st.info("Tidak ada ulasan negatif.")

                with col_right:
                    pos_reviews = df[df['Sentiment'] == 'Positif']['Cleaned_Text']
                    if not pos_reviews.empty:
                        fig_pos = plot_bigram_improved(pos_reviews, "Top Fitur Disukai", "Greens_r", custom_ignore)
                        if fig_pos: st.pyplot(fig_pos)
                    else:
                        st.info("Tidak ada ulasan positif.")

                # ROW 4: RAW DATA (Menampilkan kolom gabungan untuk verifikasi)
                with st.expander("Lihat Data Mentah (Termasuk Kolom Gabungan)"):
                    st.dataframe(df[[col_date, col_title, col_text, 'Combined_Raw', 'Sentiment']].head(50))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

else:
    st.info("👆 Silakan upload file CSV Air France di kotak di atas untuk memulai.")