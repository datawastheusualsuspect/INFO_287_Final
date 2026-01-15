#!/usr/bin/env python3
"""
Generate key visualizations for the README from the Nutella analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from textblob import TextBlob
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from scipy.stats import entropy
import collections
from math import pi
import warnings
warnings.filterwarnings("ignore")

# Set up the plotting style
sns.set_theme(style="whitegrid")
plt.style.use('default')

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def clean_tweet_text(text):
    """Clean tweet text by removing byte strings and artifacts"""
    if pd.isna(text): 
        return ""
    
    text = str(text)
    text = re.sub(r"^b['\"]", "", text)
    text = re.sub(r"['\"]$", "", text)
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    text = re.sub(r'^RT @\w+:', '', text)
    
    return text

def get_entropy(text):
    """Calculate Shannon entropy of text"""
    words = text.split()
    if not words: 
        return 0
    
    counts = collections.Counter(words)
    probs = [c / len(words) for c in counts.values()]
    return entropy(probs, base=2)

def save_figure(filename, title, dpi=300, bbox_inches='tight'):
    """Save current figure with consistent formatting"""
    plt.savefig(f'images/{filename}', dpi=dpi, bbox_inches=bbox_inches, 
                facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close()

# Load and process data
print("Loading and processing data...")
filename = 'result_Nutella.csv'
if os.path.exists(filename):
    df = pd.read_csv(filename, on_bad_lines='skip')
    df['cleaned_text'] = df['text'].apply(clean_tweet_text)
    
    # Feature engineering
    df['entropy'] = df['cleaned_text'].apply(get_entropy)
    df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['infiltration_ratio'] = df['friends'] / (df['followers'] + 1)
    df['log_retwc'] = np.log1p(df['retwc'])
    
    # Clustering
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    text_vectors = tfidf.fit_transform(df['cleaned_text']).toarray()
    
    metadata_features = ['entropy', 'sentiment', 'infiltration_ratio', 'log_retwc']
    scaler = MinMaxScaler()
    X_meta = scaler.fit_transform(df[metadata_features].fillna(0))
    X_combined = np.hstack((X_meta, text_vectors))
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_combined)
    
    # Temporal processing
    df['created'] = pd.to_datetime(df['created'], errors='coerce')
    df = df.dropna(subset=['created'])
    df['minute'] = df['created'].dt.floor('T')
    df['hour'] = df['created'].dt.hour
    df['min_int'] = df['created'].dt.minute
    
    print(f"Data loaded successfully. Total tweets: {len(df)}")
    
    # 1. Feature Distributions
    print("Generating feature distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    sns.histplot(df['entropy'], kde=True, ax=axes[0,0], color='skyblue')
    axes[0,0].set_title("Distribution of Linguistic Entropy")
    axes[0,0].set_xlabel("Entropy (bits)")
    axes[0,0].set_ylabel("Frequency")
    
    sns.histplot(df['sentiment'], kde=True, ax=axes[0,1], color='orange')
    axes[0,1].set_title("Distribution of Sentiment Polarity")
    axes[0,1].set_xlabel("Sentiment Score (-1 to +1)")
    axes[0,1].set_ylabel("Frequency")
    
    sns.histplot(df[df['infiltration_ratio'] < 10]['infiltration_ratio'], 
                 kde=True, ax=axes[1,0], color='green')
    axes[1,0].set_title("Distribution of Infiltration Ratio (Zoomed < 10)")
    axes[1,0].set_xlabel("Friends/Followers Ratio")
    axes[1,0].set_ylabel("Frequency")
    
    sns.histplot(df['log_retwc'], kde=True, ax=axes[1,1], color='red')
    axes[1,1].set_title("Distribution of Impact (Log Retweets)")
    axes[1,1].set_xlabel("Log(Retweets + 1)")
    axes[1,1].set_ylabel("Frequency")
    
    plt.tight_layout()
    save_figure('feature_distributions.png', 'Feature Distributions')
    
    # 2. Cluster Distribution and Timeline
    print("Generating cluster distribution and timeline...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    cluster_counts = df['cluster'].value_counts()
    bars = axes[0].bar(['Cluster 0', 'Cluster 1'], 
                      [cluster_counts.get(0, 0), cluster_counts.get(1, 0)], 
                      color=['#3498db', '#e74c3c'])
    axes[0].set_title("User Group Distribution: Scale Analysis")
    axes[0].set_ylabel("Number of Tweets")
    
    total_tweets = len(df)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / total_tweets) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
    
    time_data = df.groupby('minute').size()
    time_data.plot(kind='line', ax=axes[1], color='#c0392b', lw=2, marker='o', markersize=3)
    axes[1].set_title("Temporal Activity Pattern: Conversation Timeline")
    axes[1].set_xlabel("Time (Minute Intervals)")
    axes[1].set_ylabel("Tweets per Minute")
    axes[1].grid(True, alpha=0.3)
    
    mean_activity = time_data.mean()
    max_activity = time_data.max()
    axes[1].axhline(y=mean_activity, color='green', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_activity:.1f}')
    axes[1].axhline(y=max_activity, color='red', linestyle='--', alpha=0.7, 
                   label=f'Peak: {max_activity}')
    axes[1].legend()
    
    plt.tight_layout()
    save_figure('cluster_distribution_timeline.png', 'Cluster Distribution and Timeline')
    
    # 3. Temporal Heatmap
    print("Generating temporal heatmap...")
    pivot_table = df.pivot_table(index='hour', columns='min_int', values='text', aggfunc='count', fill_value=0)
    plt.figure(figsize=(18, 6))
    sns.heatmap(pivot_table, cmap='YlGnBu', cbar_kws={'label': 'Frequency'})
    plt.title("Temporal Heatmap (Hour vs Minute) - Detection of Cron Jobs")
    plt.xlabel("Minute of Hour")
    plt.ylabel("Hour of Day")
    save_figure('temporal_heatmap.png', 'Temporal Heatmap')
    
    # 4. PCA Visualization
    print("Generating PCA visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['cluster'], palette='viridis', alpha=0.6, s=50)
    plt.title("PCA Projection: Automated User Group Discovery")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster ID")
    plt.grid(True, alpha=0.3)
    save_figure('pca_clusters.png', 'PCA Clusters')
    
    # 5. Behavioral DNA - Radar and Scatter
    print("Generating behavioral DNA analysis...")
    fig = plt.figure(figsize=(18, 8))
    
    # Radar Chart
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    cluster_means = df.groupby('cluster')[metadata_features].mean()
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    angles = [n / float(len(metadata_features)) * 2 * pi for n in range(len(metadata_features))]
    angles += [angles[0]]
    
    for i in range(2):
        vals = cluster_means_norm.loc[i].tolist()
        vals += [vals[0]]
        ax1.plot(angles, vals, linewidth=2, label=f'Cluster {i}')
        ax1.fill(angles, vals, alpha=0.1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metadata_features)
    ax1.set_title("Cluster Profiles (Radar)")
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 0.1))
    
    # Scatter Plot
    ax2 = fig.add_subplot(1, 2, 2)
    sns.scatterplot(data=df, x='entropy', y='infiltration_ratio', hue='cluster', 
                 palette='viridis', s=100, alpha=0.7, ax=ax2)
    ax2.set_title("Anomaly Detection: Entropy vs. Infiltration")
    ax2.set_ylabel("Infiltration Ratio (Friends/Followers)")
    ax2.set_ylim(0, 10)
    
    # Add annotations
    ax2.text(1, 8, "BOT REGION\nLow Entropy\nHigh Infiltration", 
             bbox=dict(facecolor='red', alpha=0.3), fontsize=10, ha='center')
    ax2.text(4, 1, "HUMAN REGION\nHigh Entropy\nLow Infiltration", 
             bbox=dict(facecolor='blue', alpha=0.3), fontsize=10, ha='center')
    
    plt.tight_layout()
    save_figure('behavioral_dna.png', 'Behavioral DNA Analysis')
    
    # 6. Word Clouds
    print("Generating word clouds...")
    feature_names = tfidf.get_feature_names_out()
    c0_idx = df.index[df['cluster'] == 0].tolist()
    c1_idx = df.index[df['cluster'] == 1].tolist()
    
    c0_freqs = dict(zip(feature_names, text_vectors[c0_idx].sum(axis=0)))
    c1_freqs = dict(zip(feature_names, text_vectors[c1_idx].sum(axis=0)))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    wc0 = WordCloud(background_color='white', colormap='Blues', 
                  width=400, height=300).generate_from_frequencies(c0_freqs)
    axes[0].imshow(wc0, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title("Cluster 0 Narrative (TF-IDF Weighted)")
    
    wc1 = WordCloud(background_color='black', colormap='Reds', 
                  width=400, height=300).generate_from_frequencies(c1_freqs)
    axes[1].imshow(wc1, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title("Cluster 1 Narrative (TF-IDF Weighted)")
    
    plt.tight_layout()
    save_figure('word_clouds.png', 'Word Clouds')
    
    # 7. Top Terms
    print("Generating top terms analysis...")
    top10_c0 = sorted(c0_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_c1 = sorted(c1_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    df_c0 = pd.DataFrame(top10_c0, columns=['term', 'score'])
    df_c1 = pd.DataFrame(top10_c1, columns=['term', 'score'])
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.barplot(x='score', y='term', data=df_c0, ax=axes[0], palette='Blues_r')
    axes[0].set_title("Top 10 Terms: Cluster 0 (Organic)")
    axes[0].set_xlabel("Cumulative TF-IDF Score")
    
    sns.barplot(x='score', y='term', data=df_c1, ax=axes[1], palette='Reds_r')
    axes[1].set_title("Top 10 Terms: Cluster 1 (Inauthentic/Bot)")
    axes[1].set_xlabel("Cumulative TF-IDF Score")
    
    plt.tight_layout()
    save_figure('top_terms.png', 'Top Terms Analysis')
    
    # 8. Conclusion Composite
    print("Generating conclusion composite...")
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle("CONCLUSION: The 'Smoking Gun' Evidence", fontsize=16, weight='bold')
    
    # The Smoking Gun: Scatter Plot
    ax1 = fig.add_subplot(2, 1, 1)
    sns.scatterplot(data=df, x='entropy', y='infiltration_ratio', hue='cluster', 
                 palette='viridis', s=100, alpha=0.7, ax=ax1)
    ax1.set_title("The Separation: Bots (Top-Left) vs. Humans (Bottom-Right)", fontsize=14)
    ax1.set_xlabel("Linguistic Entropy (Complexity)")
    ax1.set_ylabel("Infiltration Ratio")
    ax1.set_ylim(0, 10)
    
    # The Content Proof: Top Terms Comparison
    ax2 = fig.add_subplot(2, 2, 3)
    sns.barplot(x='score', y='term', data=df_c0, ax=ax2, palette='Blues_r')
    ax2.set_title("Cluster 0: Organic Vocabulary")
    
    ax3 = fig.add_subplot(2, 2, 4)
    sns.barplot(x='score', y='term', data=df_c1, ax=ax3, palette='Reds_r')
    ax3.set_title("Cluster 1: Inauthentic Vocabulary")
    
    plt.tight_layout()
    save_figure('conclusion_composite.png', 'Conclusion Composite')
    
    print("\nAll visualizations generated successfully!")
    print("Files saved in 'images/' directory:")
    
    # List generated files
    image_files = [f for f in os.listdir('images') if f.endswith('.png')]
    for file in sorted(image_files):
        print(f"  - {file}")
        
else:
    print(f"Error: '{filename}' not found. Please ensure the data file is in the current directory.")
