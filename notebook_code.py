#!/usr/bin/env python3
"""
The Mystery of the Nutella Bots: A Forensic Data Investigation

This script analyzes Twitter data about Nutella to detect bot activity using
unsupervised machine learning techniques including clustering, NLP, and
behavioral analysis.

Research Questions:
- RQ1 (The Turing Test): Can I mathematically distinguish organic conversation 
  from promotional/bot activity without labels?
- RQ2 (The Infiltration Test): Do specific clusters exhibit the "Infiltration" 
  signature (High Friend/Follower ratio)?
"""

# Import essential libraries for data analysis, visualization, and machine learning
import pandas as pd          # Data manipulation and analysis
import numpy as np           # Numerical operations and array handling
import matplotlib.pyplot as plt  # Basic plotting and visualization
import seaborn as sns        # Advanced statistical data visualization
import re                    # Regular expressions for text cleaning
import os                    # Operating system interface for file operations
from textblob import TextBlob   # Sentiment analysis and text processing
from sklearn.decomposition import PCA  # Dimensionality reduction for visualization
from sklearn.preprocessing import MinMaxScaler  # Feature scaling for machine learning
from sklearn.cluster import KMeans  # Unsupervised clustering algorithm
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization for NLP
from wordcloud import WordCloud  # Word cloud generation for text visualization
from scipy.stats import entropy  # Mathematical entropy calculation for text complexity
import collections           # Additional data structures and utilities
from math import pi          # Mathematical constant for radar charts
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.preprocessing import StandardScaler
import warnings

# Configure seaborn for professional-looking plots
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

print("Digital Detective Toolkit Loaded.")


def clean_tweet_text(text):
    """
    Clean and normalize tweet text by removing various artifacts.
    
    Args:
        text: Raw tweet text that may contain byte strings, unicode, etc.
        
    Returns:
        Cleaned text ready for NLP analysis
    """
    if pd.isna(text): 
        return ""  # Handle missing values
    
    # Convert to string and remove byte string wrapper (b'...')
    text = str(text)
    text = re.sub(r"^b['\"]", "", text)  # Remove leading b' or b"
    text = re.sub(r"['\"]$", "", text)   # Remove trailing ' or "
    
    # Remove hexadecimal and unicode escape sequences
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)  # Remove \xNN hex bytes
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)  # Remove \uNNNN unicode
    
    # Remove retweet headers to focus on original content
    text = re.sub(r'^RT @\w+:', '', text)  # Remove "RT @username:" prefix
    
    return text


def get_entropy(text):
    """
    Calculate Shannon entropy of text to measure linguistic complexity.
    
    Higher entropy = more diverse vocabulary and complex language patterns
    Lower entropy = repetitive, template-based language (potential bot behavior)
    
    Args:
        text: Input text string to analyze
        
    Returns:
        Shannon entropy value in bits (base-2 logarithm)
    """
    words = text.split()
    if not words: 
        return 0  # Handle empty text
    
    # Count word frequencies
    counts = collections.Counter(words)
    
    # Calculate probability distribution of words
    probs = [c / len(words) for c in counts.values()]
    
    # Calculate Shannon entropy using base-2 logarithm (measured in bits)
    return entropy(probs, base=2)


def categorize_content(row):
    """Categorize tweet content based on structural features."""
    if row['url_count'] > 0:
        return 'Has Links'
    elif row['hashtag_count'] > 2:
        return 'Heavy Hashtags'
    elif row['mention_count'] > 2:
        return 'Heavy Mentions'
    elif row['exclamation_count'] > 1:
        return 'Excited'
    else:
        return 'Simple Text'


def load_and_clean_data(filename='result_Nutella.csv'):
    """Load and clean the tweet dataset."""
    print("=" * 60)
    print("DATA LOADING AND CLEANING")
    print("=" * 60)
    
    # Check if the dataset file exists in the current directory
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found in current directory: {os.getcwd()}")
        print("Please ensure the .csv file is in the same folder as this script.")
        return None
    
    try:
        # Load the CSV file, skipping problematic lines that might cause parsing errors
        df = pd.read_csv(filename, on_bad_lines='skip')
        print(f"Success: Dataset loaded. Total Tweets: {len(df)}")
        
        # --- RAW DATA INSPECTION ---
        # Display first 10 rows of the text column to understand data format
        print("\n--- RAW DATA (First 10 Rows) ---")
        print(df[['text']].head(10)) 
        
        # Apply the cleaning function to all tweets in the dataset
        df['cleaned_text'] = df['text'].apply(clean_tweet_text)
        
        # --- CLEANED DATA VERIFICATION ---
        # Compare raw vs cleaned text to verify the cleaning process worked correctly
        print("\n--- CLEANED DATA (First 10 Rows) ---")
        print(df[['text', 'cleaned_text']].head(10))
        print("\nData cleaning complete.")
        
        return df
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def feature_engineering(df):
    """Calculate behavioral metrics for bot detection."""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Calculate behavioral metrics for each tweet to serve as bot detection indicators
    if 'cleaned_text' in df.columns:
        # 1. LINGUISTIC ENTROPY: Measures text complexity and vocabulary diversity
        df['entropy'] = df['cleaned_text'].apply(get_entropy)
        
        # 2. SENTIMENT ANALYSIS: Measures emotional tone (-1=negative, +1=positive, 0=neutral)
        df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        
        # 3. INFILTRATION RATIO: Key bot detection metric
        # High ratio (friends >> followers) suggests aggressive following behavior typical of bots
        # Formula: friends / (followers + 1) to avoid division by zero
        df['infiltration_ratio'] = df['friends'] / (df['followers'] + 1)
        
        # 4. IMPACT METRIC: Log-transformed retweet count to handle skewness
        # Uses log1p (log(1+x)) to handle zero retweets gracefully
        df['log_retwc'] = np.log1p(df['retwc'])

        # Display summary statistics for the engineered features
        print("Feature Engineering Complete. Summary Statistics:")
        print(df[['entropy', 'sentiment', 'infiltration_ratio']].describe())

        # --- EXPLORATORY DATA ANALYSIS: FEATURE DISTRIBUTIONS ---
        # Visualize the distribution of each feature to identify patterns and anomalies
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. ENTROPY DISTRIBUTION: Look for bimodal patterns (human vs bot clusters)
        sns.histplot(df['entropy'], kde=True, ax=axes[0,0], color='skyblue')
        axes[0,0].set_title("Distribution of Linguistic Entropy")
        axes[0,0].set_xlabel("Entropy (bits)")
        axes[0,0].set_ylabel("Frequency")
        
        # 2. SENTIMENT DISTRIBUTION: Check emotional tone patterns
        sns.histplot(df['sentiment'], kde=True, ax=axes[0,1], color='orange')
        axes[0,1].set_title("Distribution of Sentiment Polarity")
        axes[0,1].set_xlabel("Sentiment Score (-1 to +1)")
        axes[0,1].set_ylabel("Frequency")
        
        # 3. INFILTRATION RATIO DISTRIBUTION (Zoomed view for better visibility)
        # Filter to ratio < 10 to focus on typical range and exclude extreme outliers
        sns.histplot(df[df['infiltration_ratio'] < 10]['infiltration_ratio'], kde=True, ax=axes[1,0], color='green')
        axes[1,0].set_title("Distribution of Infiltration Ratio (Zoomed < 10)")
        axes[1,0].set_xlabel("Friends/Followers Ratio")
        axes[1,0].set_ylabel("Frequency")
        
        # 4. IMPACT DISTRIBUTION: Log-transformed retweet counts
        sns.histplot(df['log_retwc'], kde=True, ax=axes[1,1], color='red')
        axes[1,1].set_title("Distribution of Impact (Log Retweets)")
        axes[1,1].set_xlabel("Log(Retweets + 1)")
        axes[1,1].set_ylabel("Frequency")
        
        # Adjust layout to prevent overlapping labels and titles
        plt.tight_layout()
        plt.show()
        
        return df
    else:
        print("Error: cleaned_text column not found.")
        return None


def perform_clustering(df):
    """Apply unsupervised clustering to detect user groups."""
    print("=" * 60)
    print("UNSUPERVISED CLUSTERING ANALYSIS")
    print("=" * 60)
    
    if 'cleaned_text' not in df.columns:
        print("Error: cleaned_text column not found.")
        return None
    
    # 1. TEXT VECTORIZATION WITH TF-IDF
    # Convert text to numerical vectors using Term Frequency-Inverse Document Frequency
    # This method:
    # - Ignores common words like "Nutella" that appear in all tweets
    # - Highlights unique, discriminative words that distinguish user groups
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    text_vectors = tfidf.fit_transform(df['cleaned_text']).toarray()

    # 2. METADATA FEATURE SCALING
    # Normalize numerical features to ensure equal weighting in clustering
    # Features: entropy, sentiment, infiltration_ratio, log_retwc
    metadata_features = ['entropy', 'sentiment', 'infiltration_ratio', 'log_retwc']
    scaler = MinMaxScaler()  # Scale features to [0,1] range
    X_meta = scaler.fit_transform(df[metadata_features].fillna(0))

    # 3. MULTI-MODAL FEATURE COMBINATION
    # Combine text vectors (500 dimensions) with metadata features (4 dimensions)
    # This creates a comprehensive 504-dimensional feature space for clustering
    X_combined = np.hstack((X_meta, text_vectors))
    
    # 4. K-MEANS CLUSTERING
    # Partition data into 2 clusters (expected: humans vs bots)
    # K-Means minimizes within-cluster variance to find natural groupings
    kmeans = KMeans(n_clusters=2, random_state=42)  # Fixed seed for reproducible results
    df['cluster'] = kmeans.fit_predict(X_combined)

    print(f"Clustering Complete. Found {len(df['cluster'].unique())} groups.")
    
    # 5. DIMENSIONALITY REDUCTION FOR VISUALIZATION
    # Use PCA to project 504-dimensional data to 2D for human interpretation
    # This preserves as much variance as possible while making it visualizable
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    # 6. CLUSTER VISUALIZATION
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['cluster'], palette='viridis', alpha=0.6)
    plt.title("PCA Projection of Clusters (2D Visualization of 504 Dimensions)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()
    
    return df, tfidf, text_vectors


def temporal_analysis(df):
    """Analyze temporal patterns and distribution of clusters."""
    print("=" * 60)
    print("TEMPORAL AND DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    if 'cluster' not in df.columns:
        print("Error: cluster column not found.")
        return
    
    try:
        # 1. TIMESTAMP PROCESSING
        # Convert string timestamps to datetime objects for temporal analysis
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
        
        # Remove rows with invalid timestamps to avoid analysis errors
        df = df.dropna(subset=['created'])
        
        # Create minute-level time buckets for temporal aggregation
        df['minute'] = df['created'].dt.floor('T')

        # 2. COMPARATIVE VISUALIZATION SETUP
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # CHART 1: CLUSTER SIZE DISTRIBUTION
        # Shows the relative size of each detected group
        # Important for understanding the scale of bot activity vs human activity
        sns.countplot(x='cluster', data=df, palette='viridis', ax=axes[0])
        axes[0].set_title("Group Sizes (How many in each cluster?)")
        axes[0].set_xlabel("Cluster ID")
        axes[0].set_ylabel("Number of Tweets")
        axes[0].set_xticklabels(['Cluster 0', 'Cluster 1'])

        # CHART 2: TEMPORAL ACTIVITY PATTERNS
        # Analyze tweet frequency over time to identify patterns
        # Bursty patterns may indicate coordinated bot activity
        # Steady patterns suggest organic human activity
        time_data = df.groupby('minute').size()  # Count tweets per minute
        time_data.plot(kind='line', ax=axes[1], color='#c0392b', lw=2)
        axes[1].set_title("Timeline of the Conversation")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Tweets per Minute")
        axes[1].grid(True, alpha=0.3)
        
        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not draw timeline due to error: {e}")
        print("This may be due to missing timestamp data or format issues.")


def cron_job_detection(df):
    """Detect automated scheduling patterns through temporal analysis."""
    print("=" * 60)
    print("TEMPORAL FORENSIC ANALYSIS: CRON JOB DETECTION")
    print("=" * 60)
    
    if 'cluster' not in df.columns:
        print("Error: cluster column not found.")
        return
    
    # Extract temporal components from timestamps
    df['hour'] = df['created'].dt.hour      # Hour of day (0-23)
    df['min_int'] = df['created'].dt.minute # Minute of hour (0-59)

    # Create a pivot table for temporal heatmap analysis
    # Rows: Hours of day, Columns: Minutes of hour
    # Values: Count of tweets at that specific time
    pivot_table = df.pivot_table(
        index='hour',           # Group by hour
        columns='min_int',      # Group by minute
        values='text',          # Count tweets
        aggfunc='count',        # Aggregation function
        fill_value=0            # Fill empty cells with 0
    )
    
    # Generate heatmap to visualize temporal patterns
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        pivot_table, 
        cmap='YlGnBu',           # Yellow-Green-Blue color scheme
        cbar_kws={'label': 'Activity Level'}  # Label for color bar
    )
    plt.title("The Stakeout: Are they tweeting on a schedule?")
    plt.xlabel("Minute of the Hour (0-59)")
    plt.ylabel("Hour of Day (0-23)")
    
    # Add annotations to help interpretation
    plt.text(0, -1, "Look for vertical stripes at :00, :15, :30, :45", 
             fontsize=10, color='red', style='italic')
    plt.text(0, -1.5, "These indicate automated cron job scheduling", 
             fontsize=10, color='red', style='italic')
    
    plt.show()


def behavioral_dna_analysis(df, metadata_features):
    """Compare clusters across multiple dimensions to identify bot vs human signatures."""
    print("=" * 60)
    print("BEHAVIORAL DNA ANALYSIS: COMPREHENSIVE CLUSTER PROFILING")
    print("=" * 60)
    
    if 'cluster' not in df.columns:
        print("Error: cluster column not found.")
        return
    
    # Create figure for dual visualization
    fig = plt.figure(figsize=(18, 8))

    # 1. RADAR CHART: MULTI-DIMENSIONAL CLUSTER COMPARISON
    # Shows the average behavioral profile of each cluster across all features
    ax1 = fig.add_subplot(1, 2, 1, polar=True)  # Polar coordinates for radar chart
    
    # Calculate mean values for each feature by cluster
    cluster_means = df.groupby('cluster')[metadata_features].mean()
    
    # Normalize features to [0,1] range for fair comparison on radar chart
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Calculate angles for radar chart (equally spaced around circle)
    angles = [n / float(len(metadata_features)) * 2 * pi for n in range(len(metadata_features))]
    angles += [angles[0]]  # Close the circle by repeating first angle

    # Plot each cluster as a separate line on the radar chart
    for i in range(2):  # Two clusters (0 and 1)
        vals = cluster_means_norm.loc[i].tolist()
        vals += [vals[0]]  # Close the radar shape
        
        # Plot line and fill area for each cluster
        ax1.plot(angles, vals, linewidth=2, label=f'Cluster {i}')
        ax1.fill(angles, vals, alpha=0.1)  # Semi-transparent fill

    # Configure radar chart labels and appearance
    ax1.set_xticks(angles[:-1])  # Set tick positions
    ax1.set_xticklabels(metadata_features)  # Set feature names as labels
    ax1.set_title("Digital DNA Comparison", pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 0.1))

    # 2. SCATTER PLOT: THE SMOKING GUN - ENTROPY VS INFILTRATION
    # This is the key visualization for bot detection
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Create scatter plot with clusters color-coded
    sns.scatterplot(
        data=df, 
        x='entropy', 
        y='infiltration_ratio', 
        hue='cluster', 
        palette='viridis', 
        s=100,  # Point size
        alpha=0.7,  # Transparency
        ax=ax2
    )
    
    # Configure scatter plot
    ax2.set_title("The Smoking Gun: Complexity vs. Infiltration")
    ax2.set_xlabel("Complexity of Language (Entropy)")
    ax2.set_ylabel("Infiltration Ratio (Friends/Followers)")
    ax2.set_ylim(0, 10)  # Limit y-axis for better visibility of typical range
    
    # Add annotations to highlight key regions
    ax2.text(1, 8, "BOT REGION\nLow Entropy\nHigh Infiltration", 
             bbox=dict(facecolor='red', alpha=0.3), fontsize=10, ha='center')
    ax2.text(4, 1, "HUMAN REGION\nHigh Entropy\nLow Infiltration", 
             bbox=dict(facecolor='blue', alpha=0.3), fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.show()


def content_forensics(df, tfidf, text_vectors):
    """Analyze the actual vocabulary used by each cluster to validate bot detection."""
    print("=" * 60)
    print("CONTENT FORENSICS: TF-IDF WEIGHTED WORD CLOUDS")
    print("=" * 60)
    
    if 'cluster' not in df.columns:
        print("Error: cluster column not found.")
        return
    
    # Extract feature names (words) from TF-IDF vectorizer
    feature_names = tfidf.get_feature_names_out()
    
    # Get indices of tweets belonging to each cluster
    c0_idx = df.index[df['cluster'] == 0].tolist()
    c1_idx = df.index[df['cluster'] == 1].tolist()

    # Calculate TF-IDF score sums for each word by cluster
    # This identifies the most important/discriminative words for each group
    c0_freqs = dict(zip(feature_names, text_vectors[c0_idx].sum(axis=0)))
    c1_freqs = dict(zip(feature_names, text_vectors[c1_idx].sum(axis=0)))

    # Create side-by-side word cloud comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # CLUSTER 0 WORD CLOUD
    # Generate word cloud from TF-IDF weighted frequencies
    wc0 = WordCloud(
        background_color='white', 
        colormap='Blues',  # Blue color scheme
        width=400, 
        height=300
    ).generate_from_frequencies(c0_freqs)
    
    axes[0].imshow(wc0, interpolation='bilinear')
    axes[0].axis('off')  # Hide axes
    axes[0].set_title("Cluster 0: What are they saying?", fontsize=14, pad=20)

    # CLUSTER 1 WORD CLOUD
    # Use contrasting color scheme for easy visual distinction
    wc1 = WordCloud(
        background_color='black', 
        colormap='Reds',  # Red color scheme
        width=400, 
        height=300
    ).generate_from_frequencies(c1_freqs)
    
    axes[1].imshow(wc1, interpolation='bilinear')
    axes[1].axis('off')  # Hide axes
    axes[1].set_title("Cluster 1: What are they saying?", fontsize=14, pad=20)
    
    # Add interpretation guide
    fig.suptitle("Content Analysis: TF-IDF Weighted Vocabulary Comparison", 
                 fontsize=16, y=0.95)
    fig.text(0.5, 0.02, "Larger words = More discriminative for that cluster", 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    return c0_freqs, c1_freqs


def quantitative_vocabulary_analysis(c0_freqs, c1_freqs):
    """Extract and display the most important words for each cluster."""
    print("=" * 60)
    print("QUANTITATIVE VOCABULARY ANALYSIS: TOP 10 DISCRIMINATIVE TERMS")
    print("=" * 60)
    
    # Sort TF-IDF frequency dictionaries by score (descending) to find top terms
    # This provides a precise, ranked list rather than visual word clouds
    top10_c0 = sorted(c0_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_c1 = sorted(c1_freqs.items(), key=lambda x: x[1], reverse=True)[:10]

    # Convert sorted lists to DataFrames for easy plotting with seaborn
    df_c0 = pd.DataFrame(top10_c0, columns=['term', 'score'])
    df_c1 = pd.DataFrame(top10_c1, columns=['term', 'score'])

    # Create side-by-side bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # CLUSTER 0 TOP TERMS BAR CHART
    sns.barplot(
        x='score', 
        y='term', 
        data=df_c0, 
        ax=axes[0], 
        palette='Blues_r',  # Reverse blue colors (darker = higher score)
        orient='h'  # Horizontal bars
    )
    axes[0].set_title("Cluster 0: Top 10 Discriminative Terms", fontsize=14)
    axes[0].set_xlabel("Cumulative TF-IDF Score")
    axes[0].set_ylabel("Terms")
    
    # Add value labels on bars for precise reading
    for i, v in enumerate(df_c0['score']):
        axes[0].text(v + 0.01, i, f'{v:.2f}', va='center')

    # CLUSTER 1 TOP TERMS BAR CHART
    sns.barplot(
        x='score', 
        y='term', 
        data=df_c1, 
        ax=axes[1], 
        palette='Reds_r',  # Reverse red colors (darker = higher score)
        orient='h'  # Horizontal bars
    )
    axes[1].set_title("Cluster 1: Top 10 Discriminative Terms", fontsize=14)
    axes[1].set_xlabel("Cumulative TF-IDF Score")
    axes[1].set_ylabel("Terms")
    
    # Add value labels on bars for precise reading
    for i, v in enumerate(df_c1['score']):
        axes[1].text(v + 0.01, i, f'{v:.2f}', va='center')

    # Add overall title and interpretation guide
    fig.suptitle("Quantitative Vocabulary Analysis: Most Discriminative Terms by Cluster", 
                 fontsize=16, y=0.95)
    fig.text(0.5, 0.02, "Higher scores = More unique/important to that cluster's vocabulary", 
             ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.show()


def comprehensive_relationship_analysis(df):
    """Systematically examine ALL possible relationships to ensure no bot detection patterns are missed."""
    print("=" * 60)
    print("COMPREHENSIVE RELATIONSHIP ANALYSIS")
    print("=" * 60)
    
    # --- ADDITIONAL FEATURE ENGINEERING ---
    # Calculate content structure features for relationship analysis
    
    # Text structure metrics
    df['text_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df['hashtag_count'] = df['cleaned_text'].str.count(r'#\w+')
    df['mention_count'] = df['cleaned_text'].str.count(r'@\w+')
    df['url_count'] = df['cleaned_text'].str.count(r'http\S+')
    df['question_count'] = df['cleaned_text'].str.count(r'\?')
    df['exclamation_count'] = df['cleaned_text'].str.count(r'!')
    
    # Engagement ratios
    df['follower_friend_ratio'] = df['followers'] / (df['friends'] + 1)
    df['retweet_per_follower'] = df['retwc'] / (df['followers'] + 1)
    
    # Temporal features
    df['day_of_week'] = df['created'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Content categorization
    df['content_type'] = df.apply(categorize_content, axis=1)
    
    # Engagement categorization
    df['engagement_level'] = pd.cut(df['retwc'], 
                                    bins=[-1, 0, 1, 5, float('inf')],
                                    labels=['No RT', 'Low RT', 'Medium RT', 'High RT'])
    
    print("Additional features calculated:")
    print(f"- Content structure: {['text_length', 'word_count', 'hashtag_count', 'mention_count', 'url_count', 'question_count', 'exclamation_count']}")
    print(f"- Engagement ratios: {['follower_friend_ratio', 'retweet_per_follower']}")
    print(f"- Temporal: {['day_of_week', 'is_weekend']}")
    print(f"- Content categories: {df['content_type'].nunique()} types")
    print(f"- Engagement levels: {df['engagement_level'].nunique()} levels")


def correlation_matrix_analysis(df):
    """Systematically examine ALL relationships between numerical features."""
    print("=" * 60)
    print("COMPREHENSIVE CORRELATION MATRIX ANALYSIS")
    print("=" * 60)
    
    # Select all numerical features for correlation analysis
    numerical_features = [
        'entropy', 'sentiment', 'infiltration_ratio', 'log_retwc',
        'text_length', 'word_count', 'hashtag_count', 'mention_count',
        'url_count', 'question_count', 'exclamation_count',
        'followers', 'friends', 'retwc', 'follower_friend_ratio',
        'retweet_per_follower', 'hour'
    ]
    
    # Filter to available features
    available_features = [f for f in numerical_features if f in df.columns]
    df_num = df[available_features].fillna(0)
    
    # Calculate both Pearson (linear) and Spearman (monotonic) correlations
    pearson_corr = df_num.corr(method='pearson')
    spearman_corr = df_num.corr(method='spearman')
    
    # Create comprehensive correlation visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Pearson correlation (linear relationships)
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', ax=ax1, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8})
    ax1.set_title('Pearson Correlation (Linear Relationships)', fontsize=14, fontweight='bold')
    
    # Spearman correlation (monotonic relationships)
    sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', ax=ax2, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8})
    ax2.set_title('Spearman Correlation (Monotonic Relationships)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Identify and report strongest correlations
    print("\nSTRONGEST CORRELATIONS (|r| > 0.5):")
    print("-" * 50)
    
    strong_correlations = []
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            feat1, feat2 = available_features[i], available_features[j]
            pearson_r = pearson_corr.loc[feat1, feat2]
            spearman_r = spearman_corr.loc[feat1, feat2]
            
            if abs(pearson_r) > 0.5 or abs(spearman_r) > 0.5:
                strong_correlations.append({
                    'Feature 1': feat1,
                    'Feature 2': feat2,
                    'Pearson r': round(pearson_r, 3),
                    'Spearman r': round(spearman_r, 3)
                })
    
    if strong_correlations:
        corr_df = pd.DataFrame(strong_correlations)
        print(corr_df.to_string(index=False))
    else:
        print("No strong correlations found (|r| > 0.5)")


def content_structure_analysis(df):
    """Examine how content structure relates to bot-like behavior."""
    print("=" * 60)
    print("CONTENT STRUCTURE RELATIONSHIP ANALYSIS")
    print("=" * 60)
    
    # Analyze bot indicators by content type
    content_analysis = df.groupby('content_type').agg({
        'infiltration_ratio': ['mean', 'median', 'std', 'count'],
        'entropy': ['mean', 'median', 'std'],
        'sentiment': 'mean',
        'retwc': 'mean',
        'text_length': 'mean',
        'cluster': lambda x: (x == 1).mean()  # Proportion in cluster 1
    }).round(3)
    
    print("BOT INDICATORS BY CONTENT TYPE:")
    print("=" * 50)
    print(content_analysis)
    
    # Create comprehensive content structure visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Content Structure vs. Bot Behavior Analysis', fontsize=16, fontweight='bold')
    
    # 1. Infiltration ratio by content type
    sns.boxplot(data=df, x='content_type', y='infiltration_ratio', ax=axes[0,0])
    axes[0,0].set_title('Infiltration Ratio by Content Type')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].set_ylim(0, df['infiltration_ratio'].quantile(0.95))
    
    # 2. Entropy by content type
    sns.boxplot(data=df, x='content_type', y='entropy', ax=axes[0,1])
    axes[0,1].set_title('Entropy by Content Type')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Sentiment by content type
    sns.boxplot(data=df, x='content_type', y='sentiment', ax=axes[0,2])
    axes[0,2].set_title('Sentiment by Content Type')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # 4. Text length by content type
    sns.boxplot(data=df, x='content_type', y='text_length', ax=axes[1,0])
    axes[1,0].set_title('Text Length by Content Type')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Hashtag count analysis
    hashtag_analysis = df.groupby('content_type')['hashtag_count'].agg(['mean', 'median']).round(2)
    sns.barplot(data=hashtag_analysis.reset_index(), x='content_type', y='mean', ax=axes[1,1])
    axes[1,1].set_title('Average Hashtag Count by Content Type')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Cluster distribution by content type
    cluster_content = pd.crosstab(df['content_type'], df['cluster'], normalize='index') * 100
    cluster_content.plot(kind='bar', stacked=True, ax=axes[1,2], color=['blue', 'red'])
    axes[1,2].set_title('Cluster Distribution by Content Type')
    axes[1,2].set_ylabel('Percentage')
    axes[1,2].tick_params(axis='x', rotation=45)
    axes[1,2].legend(['Cluster 0', 'Cluster 1'])
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function for the Nutella bot detection analysis."""
    print("=" * 80)
    print("THE MYSTERY OF THE NUTELLA BOTS: A FORENSIC DATA INVESTIGATION")
    print("=" * 80)
    print("\nResearch Questions:")
    print("- RQ1 (The Turing Test): Can I mathematically distinguish organic conversation")
    print("  from promotional/bot activity without labels?")
    print("- RQ2 (The Infiltration Test): Do specific clusters exhibit the 'Infiltration'")
    print("  signature (High Friend/Follower ratio)?")
    print("=" * 80)
    
    # Step 1: Load and clean data
    df = load_and_clean_data()
    if df is None:
        return
    
    # Step 2: Feature engineering
    df = feature_engineering(df)
    if df is None:
        return
    
    # Step 3: Perform clustering
    result = perform_clustering(df)
    if result is None:
        return
    
    df, tfidf, text_vectors = result
    metadata_features = ['entropy', 'sentiment', 'infiltration_ratio', 'log_retwc']
    
    # Step 4: Temporal analysis
    temporal_analysis(df)
    
    # Step 5: Cron job detection
    cron_job_detection(df)
    
    # Step 6: Behavioral DNA analysis
    behavioral_dna_analysis(df, metadata_features)
    
    # Step 7: Content forensics
    word_freqs = content_forensics(df, tfidf, text_vectors)
    if word_freqs is None:
        return
    
    c0_freqs, c1_freqs = word_freqs
    
    # Step 8: Quantitative vocabulary analysis
    quantitative_vocabulary_analysis(c0_freqs, c1_freqs)
    
    # Step 9: Comprehensive relationship analysis
    comprehensive_relationship_analysis(df)
    
    # Step 10: Correlation matrix analysis
    correlation_matrix_analysis(df)
    
    # Step 11: Content structure analysis
    content_structure_analysis(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("The investigation has concluded. Review the visualizations and")
    print("statistical summaries above to identify bot patterns in the Nutella dataset.")


if __name__ == "__main__":
    main()
