# Comprehensive Relationship Analysis for Bot Detection Project
# This script checks relationships that may be missing from the current analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filename='result_Nutella.csv'):
    """Load and clean the main dataset"""
    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
        
        # Clean text
        def clean_tweet_text(text):
            if pd.isna(text): return ""
            text = str(text)
            text = re.sub(r"^b['\"]", "", text)
            text = re.sub(r"['\"]$", "", text)
            text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            text = re.sub(r'^RT @\w+:', '', text)
            return text
        
        df['cleaned_text'] = df['text'].apply(clean_tweet_text)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_additional_features(df):
    """Calculate features for comprehensive relationship analysis"""
    from textblob import TextBlob
    from scipy.stats import entropy
    import collections
    
    # Basic features (from original analysis)
    def get_entropy(text):
        words = text.split()
        if not words: return 0
        counts = collections.Counter(words)
        probs = [c / len(words) for c in counts.values()]
        return entropy(probs, base=2)
    
    df['entropy'] = df['cleaned_text'].apply(get_entropy)
    df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['infiltration_ratio'] = df['friends'] / (df['followers'] + 1)
    df['log_retwc'] = np.log1p(df['retwc'])
    
    # Additional features for relationship analysis
    df['text_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df['hashtag_count'] = df['cleaned_text'].str.count(r'#\w+')
    df['mention_count'] = df['cleaned_text'].str.count(r'@\w+')
    df['url_count'] = df['cleaned_text'].str.count(r'http\S+')
    df['question_count'] = df['cleaned_text'].str.count(r'\?')
    df['exclamation_count'] = df['cleaned_text'].str.count(r'!')
    
    # Time-based features
    df['created'] = pd.to_datetime(df['created'], errors='coerce')
    df['hour'] = df['created'].dt.hour
    df['day_of_week'] = df['created'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Engagement ratios
    df['follower_friend_ratio'] = df['followers'] / (df['friends'] + 1)
    df['retweet_per_follower'] = df['retwc'] / (df['followers'] + 1)
    
    return df

def correlation_analysis(df):
    """Comprehensive correlation analysis between all numerical features"""
    print("=" * 60)
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Select numerical features
    numerical_features = [
        'entropy', 'sentiment', 'infiltration_ratio', 'log_retwc',
        'text_length', 'word_count', 'hashtag_count', 'mention_count',
        'url_count', 'question_count', 'exclamation_count',
        'followers', 'friends', 'retwc', 'follower_friend_ratio',
        'retweet_per_follower', 'hour'
    ]
    
    # Filter available features
    available_features = [f for f in numerical_features if f in df.columns]
    df_num = df[available_features].fillna(0)
    
    # Pearson correlation
    pearson_corr = df_num.corr(method='pearson')
    
    # Spearman correlation (for non-linear relationships)
    spearman_corr = df_num.corr(method='spearman')
    
    # Create comprehensive correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pearson correlation
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Pearson Correlation (Linear Relationships)', fontsize=14)
    
    # Spearman correlation
    sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Spearman Correlation (Monotonic Relationships)', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Find strongest correlations
    print("\nSTRONGEST CORRELATIONS (|r| > 0.5):")
    print("-" * 40)
    
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
                    'Pearson r': pearson_r,
                    'Spearman r': spearman_r
                })
    
    if strong_correlations:
        corr_df = pd.DataFrame(strong_correlations)
        print(corr_df.to_string(index=False))
    else:
        print("No strong correlations found (|r| > 0.5)")
    
    return pearson_corr, spearman_corr

def temporal_relationship_analysis(df):
    """Analyze temporal relationships and patterns"""
    print("\n" + "=" * 60)
    print("TEMPORAL RELATIONSHIP ANALYSIS")
    print("=" * 60)
    
    # Hourly patterns
    hourly_stats = df.groupby('hour').agg({
        'entropy': 'mean',
        'sentiment': 'mean',
        'infiltration_ratio': 'mean',
        'retwc': 'sum',
        'text_length': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot hourly patterns
    sns.lineplot(data=hourly_stats, x='hour', y='entropy', ax=axes[0,0], marker='o')
    axes[0,0].set_title('Average Entropy by Hour')
    axes[0,0].set_xlabel('Hour of Day')
    
    sns.lineplot(data=hourly_stats, x='hour', y='sentiment', ax=axes[0,1], marker='o', color='orange')
    axes[0,1].set_title('Average Sentiment by Hour')
    axes[0,1].set_xlabel('Hour of Day')
    
    sns.lineplot(data=hourly_stats, x='hour', y='infiltration_ratio', ax=axes[0,2], marker='o', color='green')
    axes[0,2].set_title('Average Infiltration Ratio by Hour')
    axes[0,2].set_xlabel('Hour of Day')
    
    sns.lineplot(data=hourly_stats, x='hour', y='retwc', ax=axes[1,0], marker='o', color='red')
    axes[1,0].set_title('Total Retweets by Hour')
    axes[1,0].set_xlabel('Hour of Day')
    
    sns.lineplot(data=hourly_stats, x='hour', y='text_length', ax=axes[1,1], marker='o', color='purple')
    axes[1,1].set_title('Average Text Length by Hour')
    axes[1,1].set_xlabel('Hour of Day')
    
    # Activity volume
    hourly_volume = df.groupby('hour').size().reset_index(name='tweet_count')
    sns.lineplot(data=hourly_volume, x='hour', y='tweet_count', ax=axes[1,2], marker='o', color='black')
    axes[1,2].set_title('Tweet Volume by Hour')
    axes[1,2].set_xlabel('Hour of Day')
    
    plt.tight_layout()
    plt.show()
    
    # Weekend vs Weekday analysis
    if 'is_weekend' in df.columns:
        weekend_comparison = df.groupby('is_weekend').agg({
            'entropy': 'mean',
            'sentiment': 'mean',
            'infiltration_ratio': 'mean',
            'retwc': 'mean',
            'text_length': 'mean'
        }).round(3)
        
        print("\nWEEKEND vs WEEKDAY COMPARISON:")
        print("(0 = Weekday, 1 = Weekend)")
        print(weekend_comparison)

def content_structure_relationships(df):
    """Analyze relationships between content structure and bot-like behavior"""
    print("\n" + "=" * 60)
    print("CONTENT STRUCTURE RELATIONSHIPS")
    print("=" * 60)
    
    # Create content structure categories
    def categorize_content(row):
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
    
    df['content_type'] = df.apply(categorize_content, axis=1)
    
    # Analyze bot indicators by content type
    content_analysis = df.groupby('content_type').agg({
        'infiltration_ratio': ['mean', 'median', 'std'],
        'entropy': ['mean', 'median', 'std'],
        'sentiment': 'mean',
        'retwc': 'mean',
        'text_length': 'mean'
    }).round(3)
    
    print("BOT INDICATORS BY CONTENT TYPE:")
    print(content_analysis)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Infiltration ratio by content type
    sns.boxplot(data=df, x='content_type', y='infiltration_ratio', ax=axes[0,0])
    axes[0,0].set_title('Infiltration Ratio by Content Type')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Entropy by content type
    sns.boxplot(data=df, x='content_type', y='entropy', ax=axes[0,1])
    axes[0,1].set_title('Entropy by Content Type')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Sentiment by content type
    sns.boxplot(data=df, x='content_type', y='sentiment', ax=axes[1,0])
    axes[1,0].set_title('Sentiment by Content Type')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Text length by content type
    sns.boxplot(data=df, x='content_type', y='text_length', ax=axes[1,1])
    axes[1,1].set_title('Text Length by Content Type')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def engagement_relationship_analysis(df):
    """Analyze relationships between engagement metrics and user behavior"""
    print("\n" + "=" * 60)
    print("ENGAGEMENT RELATIONSHIP ANALYSIS")
    print("=" * 60)
    
    # Create engagement categories
    df['engagement_level'] = pd.cut(df['retwc'], 
                                    bins=[-1, 0, 1, 5, float('inf')],
                                    labels=['No RT', 'Low RT', 'Medium RT', 'High RT'])
    
    # Analyze behavior by engagement level
    engagement_analysis = df.groupby('engagement_level').agg({
        'infiltration_ratio': 'mean',
        'entropy': 'mean',
        'sentiment': 'mean',
        'followers': 'median',
        'friends': 'median',
        'text_length': 'mean'
    }).round(3)
    
    print("USER BEHAVIOR BY ENGAGEMENT LEVEL:")
    print(engagement_analysis)
    
    # Correlation between engagement and bot indicators
    engagement_corr = df[['retwc', 'infiltration_ratio', 'entropy', 'sentiment']].corr()
    
    print("\nENGAGEMENT CORRELATION MATRIX:")
    print(engagement_corr.round(3))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Retweets vs Infiltration
    sns.scatterplot(data=df[df['infiltration_ratio'] < 20], x='infiltration_ratio', 
                   y='retwc', alpha=0.6, ax=axes[0,0])
    axes[0,0].set_title('Retweets vs Infiltration Ratio')
    
    # Retweets vs Entropy
    sns.scatterplot(data=df, x='entropy', y='retwc', alpha=0.6, ax=axes[0,1])
    axes[0,1].set_title('Retweets vs Entropy')
    
    # Engagement by content type
    sns.boxplot(data=df, x='content_type', y='retwc', ax=axes[1,0])
    axes[1,0].set_title('Retweets by Content Type')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].set_ylim(0, df['retwc'].quantile(0.95))  # Limit outliers
    
    # Engagement heatmap
    engagement_pivot = df.groupby(['content_type', 'engagement_level']).size().unstack(fill_value=0)
    sns.heatmap(engagement_pivot, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title('Content Type vs Engagement Level')
    
    plt.tight_layout()
    plt.show()

def missing_relationships_summary():
    """Identify relationships not checked in the original analysis"""
    print("\n" + "=" * 60)
    print("MISSING RELATIONSHIPS FROM ORIGINAL ANALYSIS")
    print("=" * 60)
    
    missing_relationships = [
        "Text length vs. Bot indicators",
        "Hashtag frequency vs. Infiltration ratio", 
        "Mention patterns vs. User type",
        "URL sharing behavior vs. Engagement",
        "Question/Exclamation usage vs. Sentiment",
        "Weekend vs. Weekday behavioral patterns",
        "Hour-of-day vs. Content complexity",
        "Engagement level vs. Network metrics",
        "Content structure vs. Temporal patterns",
        "Multi-variate interaction effects",
        "Non-linear relationships (beyond linear correlation)",
        "Categorical relationships (Chi-square tests)",
        "Cluster stability over time",
        "Cross-feature interaction terms"
    ]
    
    for i, relationship in enumerate(missing_relationships, 1):
        print(f"{i:2d}. {relationship}")

def main():
    """Run comprehensive relationship analysis"""
    print("COMPREHENSIVE RELATIONSHIP ANALYSIS FOR BOT DETECTION")
    print("Analyzing relationships not covered in the original project...")
    
    # Load and prepare data
    df = load_and_clean_data()
    if df is None:
        return
    
    df = calculate_additional_features(df)
    
    # Run all relationship analyses
    correlation_analysis(df)
    temporal_relationship_analysis(df)
    content_structure_relationships(df)
    engagement_relationship_analysis(df)
    missing_relationships_summary()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("This analysis reveals relationships that should be checked")
    print("to ensure comprehensive bot detection coverage.")

if __name__ == "__main__":
    main()
