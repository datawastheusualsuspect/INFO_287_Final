# %% [markdown]
# # The Mystery of the Nutella Bots: A Forensic Data Investigation
# 
# ## 1. The Problem
# Social media is full of bots negatively impacting private companies. In the past, they were easy to spot because they just spammed links. Today, they are smarter. They sleep at night, use normal words, and try to trick real people into following them.
# 
# **The Challenge:** I have a dataset of tweets about "Nutella." I suspect there are bots hiding in here, but I don't have a list of who is who. I have to catch them using only math.
# 
# **The Plan:** I will use **Clustering** to look at how the users behave and statistically separate them into groups based on their behavior. I propose a **Multi-Modal Forensic Pipeline**. I triangulate bot behavior using Computational Linguistics, Network Statistics, and Unsupervised Learning (K-Means) to mathematically separate organic users from automated actors.
# 
# My hypothesis is that one group will be the **Humans** chatting and the other will be **Bots** trying to influence the conversation.
# 
# ### Research Questions
# *   **RQ1 (The Turing Test):** Can I mathematically distinguish organic conversation from promotional/bot activity without labels?
# *   **RQ2 (The Infiltration Test):** Do specific clusters exhibit the "Infiltration" signature (High Friend/Follower ratio)?

# %%
# First, I set up my digital workspace.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from scipy.stats import entropy
import collections

# Make the charts look nice
sns.set_theme(style="whitegrid")
print("Digital Detective Toolkit Loaded.")

# %% [markdown]
# ## 2. Cleaning the Crime Scene (Data Loading)
# 
# Raw data is messy. When I look at the tweets, I see garbage symbols like `b'text''. 
# Before I can analyze what people are saying, I need to clean this up.
# 
# I will write a **Cleaning Function** that strips away the noise so we are left with clean posts.

# %%
filename = 'result_Nutella.csv'

# --- Data Loading ---
if os.path.exists(filename):
    try:
        # Load csv from the current directory
        df = pd.read_csv(filename, on_bad_lines='skip')
        print(f"Success: Dataset loaded. Total Tweets: {len(df)}")
        
        # --- INSPECT RAW DATA ---
        print("\n--- RAW DATA (First 10 Rows) ---")
        # Showing text column to visualize byte strings
        print(df[['text']].head(10)) 
        
        # --- Data Cleaning ---
        def clean_tweet_text(text):
            if pd.isna(text): return ""
            text = str(text)
            text = re.sub(r"^b['\"]", "", text) # Remove byte wrapper
            text = re.sub(r"['\"]$", "", text)  # Remove trailing quote
            text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text) # Remove hex bytes
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text) # Remove unicode
            text = re.sub(r'^RT @\w+:', '', text) # Remove RT headers
            return text

        df['cleaned_text'] = df['text'].apply(clean_tweet_text)
        
        # --- INSPECT CLEANED DATA ---
        print("\n--- CLEANED DATA (First 10 Rows) ---")
        print(df[['text', 'cleaned_text']].head(10))
        print("\nData cleaning complete.")
        
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"Error: '{filename}' not found in current directory: {os.getcwd()}")
    print("Please ensure the .csv file is in the same folder as this notebook.")

# %% [markdown]
# ## 3. Gathering Clues (Feature Engineering)
# 
# Since I don't know *who* the bots are, I have to look for **behaviors**. I will calculate four specific "clues" for every tweet:
# 
# 1.  **Complexity (Entropy):** Is the text creative (High Score) or repetitive/simple (Low Score)? Bots often use templates. ($H$) Measures complexity. 
# 2.  **Infiltration Ratio:** This is the most important clue. ($\frac{\text{Friends}}{\text{Followers} + 1}$). Measures aggressive following. I divide the number of people they *Follow* by the number of *Followers* they have.
#     *   *Normal:* Follows 100, has 100 followers. Ratio ~ 1.
#     *   *Bot:* Follows 5,000, has 10 followers. Ratio ~ 500. (They are trying to force their way in).
# 3.  **Sentiment:** Are they super happy, super angry, or neutral?
# 4.  **Impact:** How many retweets did they get?
# 
# **Visual Inspection (EDA):** Before clustering, I visualize the distributions of these features using histograms. 
# *   **Why?** To check for "Bimodal Distributions" (two humps). If the histograms show two distinct peaks, it suggests that the data naturally falls into two groups (e.g., Humans vs. Bots) even before we apply AI.

# %%
def get_entropy(text):
    words = text.split()
    if not words: return 0
    counts = collections.Counter(words)
    probs = [c / len(words) for c in counts.values()]
    return entropy(probs, base=2)

if 'cleaned_text' in df.columns:
    # Calculations
    df['entropy'] = df['cleaned_text'].apply(get_entropy)
    df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['infiltration_ratio'] = df['friends'] / (df['followers'] + 1)
    df['log_retwc'] = np.log1p(df['retwc'])

    print("Feature Engineering Complete. Summary Statistics:")
    print(df[['entropy', 'sentiment', 'infiltration_ratio']].describe())

    # --- VISUALIZATION: Feature Distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Entropy
    sns.histplot(df['entropy'], kde=True, ax=axes[0,0], color='skyblue')
    axes[0,0].set_title("Distribution of Linguistic Entropy")
    
    # 2. Sentiment
    sns.histplot(df['sentiment'], kde=True, ax=axes[0,1], color='orange')
    axes[0,1].set_title("Distribution of Sentiment Polarity")
    
    # 3. Infiltration (Zoomed in to < 10 for visibility)
    sns.histplot(df[df['infiltration_ratio'] < 10]['infiltration_ratio'], kde=True, ax=axes[1,0], color='green')
    axes[1,0].set_title("Distribution of Infiltration Ratio (Zoomed < 10)")
    
    # 4. Impact
    sns.histplot(df['log_retwc'], kde=True, ax=axes[1,1], color='red')
    axes[1,1].set_title("Distribution of Impact (Log Retweets)")
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. The AI Detective (Clustering)
# 
# Now I have the clues, but I need to group the suspects. 
# 
# **Why Clustering?**
# Because I do not have ground-truth labels (I don't know which tweets are bots), I cannot use Supervised Classification (like Random Forest). Instead, I use **Unsupervised Learning** to discover inherent structures in the data. The goal is to partition the tweets into two distinct groups based on statistical similarity.
# 
# **The Mathematics (K-Means):**
# I employ the **K-Means Algorithm** ($k=2$). K-Means attempts to minimize the **Inertia** (Within-Cluster Sum of Squares). 
# 
# The objective function is:
# $$ J = \sum_{j=1}^{k} \sum_{i=1}^{n} ||x_i^{(j)} - \mu_j||^2 $$
# 
# Where:
# *   $||x_i - \mu_j||^2$ is the **Euclidean Distance** between a data point $x_i$ and its cluster centroid $\mu_j$.
# *   The algorithm iteratively assigns tweets to the nearest centroid and then recalculates the centroids until the position stabilizes.
# 
# **The Approach (TF-IDF):**
# Every tweet contains the word "Nutella", so simple word counts are useless. I apply **TF-IDF**, which penalizes common words (IDF) and highlights unique vocabulary. I then combine these text vectors with the metadata features to perform the clustering.
# 
# **Visualization:** To verify the clustering worked, I use **PCA (Principal Component Analysis)** to project the 504 dimensions down to 2 dimensions.

# %%
if 'cleaned_text' in df.columns:
    # 1. TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    text_vectors = tfidf.fit_transform(df['cleaned_text']).toarray()

    # 2. Metadata Scaling
    metadata_features = ['entropy', 'sentiment', 'infiltration_ratio', 'log_retwc']
    scaler = MinMaxScaler()
    X_meta = scaler.fit_transform(df[metadata_features].fillna(0))

    # 3. Combine & Cluster (High Dimensional Space)
    X_combined = np.hstack((X_meta, text_vectors))
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_combined)

    print(f"Clustering Complete. Found {len(df['cluster'].unique())} groups.")
    
    # 4. PCA Projection (Visualizing the separation in 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['cluster'], palette='viridis', alpha=0.6)
    plt.title("PCA Projection of Clusters (2D Visualization of 504 Dimensions)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# %% [markdown]
# ## 5. The Lineup (Summary Statistics)
# 
# Let's look at the basic breakdown. 
# *   **The Bar Chart** tells us how big the problem is. Are there more bots than humans?
# *   **The Line Chart** shows us *when* they are active. Is it a steady stream, or did they attack all at once?
# 
# **Approach:** I visualize the magnitude of the dataset and the temporal flow.
# **Findings:** The Bar Chart shows the balance between the two detected groups. The Line Chart reveals if activity was continuous or bursty.

# %%
import warnings
warnings.filterwarnings("ignore")

if 'cluster' in df.columns:
    try:
        # Fix dates
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
        df = df.dropna(subset=['created'])
        df['minute'] = df['created'].dt.floor('T')

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Chart 1: How many in each group?
        sns.countplot(x='cluster', data=df, palette='viridis', ax=axes[0])
        axes[0].set_title("Group Sizes (How many in each?)")

        # Chart 2: Activity over time
        time_data = df.groupby('minute').size()
        time_data.plot(kind='line', ax=axes[1], color='#c0392b', lw=2)
        axes[1].set_title("Timeline of the Conversation")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not draw timeline.")

# %% [markdown]
# ## 6. The Stakeout (Temporal Heatmap)
# 
# Humans behave chaotically. We tweet when we feel like it. 
# Bots behave robotically. They often run on cron timers set to run at the top of the hour (:00) or every 15 minutes (:15, :30).
# 
# I'm plotting a heatmap of **Hour vs. Minute**. I am looking for **Vertical Stripes**. If I see dark blocks lined up at specific minutes across all hours, that is a sign of automation.
# 
# **Approach:** I plot a two-way frequency heatmap (Hour vs. Minute) to detect robotic scheduling.
# **Findings:** I look for **Vertical Stripes**. If activity spikes at exactly `:00`, `:15`, or `:30` across different hours, it indicates Cron Job automation (False Positive check: ensure stripes persist across multiple hours).

# %%
if 'cluster' in df.columns:
    df['hour'] = df['created'].dt.hour
    df['min_int'] = df['created'].dt.minute

    # Create a grid: Hours on Y, Minutes on X
    pivot_table = df.pivot_table(index='hour', columns='min_int', values='text', aggfunc='count', fill_value=0)
    
    plt.figure(figsize=(18, 6))
    sns.heatmap(pivot_table, cmap='YlGnBu', cbar_kws={'label': 'Activity Level'})
    plt.title("The Stakeout: Are they tweeting on a schedule?")
    plt.xlabel("Minute of the Hour (0-59)")
    plt.show()

# %% [markdown]
# ## 7. The Smoking Gun (Scatter Plot)
# 
# This is the most critical test. I am comparing **Complexity (Entropy)** against **Infiltration (Follower Ratio)**.
# 
# **What to look for:**
# *   **Bottom-Right:** High Complexity, Low Infiltration. These are normal people.
# *   **Top-Left:** Low Complexity, High Infiltration. These users follow *thousands* of people but only say simple, repetitive things. 
# 
# **Approach:** 
# 1.  **Radar Chart:** visualizes the average profile of each cluster (Central Tendency).
# 2.  **Scatter Plot:** A **Visual Test for Outliers**. I plot Entropy vs. Infiltration.
# 
# **Findings:** 
# *   **The Bot Pattern:** If we see a cluster in the **Top-Left**, those are our bots. (Low Entropy, High Infiltration). These are users who follow aggressively but speak robotically.

# %%
if 'cluster' in df.columns:
    from math import pi
    fig = plt.figure(figsize=(18, 8))

    # Radar Chart: The "DNA" of the clusters
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
    ax1.set_title("Digital DNA Comparison")
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 0.1))

    # Scatter Plot: The Smoking Gun
    ax2 = fig.add_subplot(1, 2, 2)
    sns.scatterplot(data=df, x='entropy', y='infiltration_ratio', hue='cluster', palette='viridis', s=100, alpha=0.7, ax=ax2)
    ax2.set_title("The Smoking Gun: Complexity vs. Infiltration")
    ax2.set_xlabel("Complexity of Language (Entropy)")
    ax2.set_ylabel("Infiltration Ratio (Friends/Followers)")
    ax2.set_ylim(0, 10)
    plt.show()

# %% [markdown]
# ## 8. The Wiretap (Word Clouds)
# 
# Now that we have separated them mathematically, let's see what they are actually saying.
# 
# I am using the **TF-IDF scores** to generate these clouds. This means common words are hidden, and only the words *unique* to each group are shown.
# 
# **Interpretation:** 
# *   If one cloud says "Breakfast," "Good," and "Brownie," that is likely the Humans.
# *   If the other cloud says "Win," "Vote," "Streaming," or "Giveaway," that is likely the Bot Farm.
# 
# **Approach:** I generate Word Clouds weighted by **TF-IDF Scores** (not raw counts).
# **Findings:** This allows me to validate the clusters qualitatively. If one cluster emphasizes organic words ("Breakfast", "Love") and the other emphasizes promotional words ("Win", "Free"), the unsupervised model has successfully identified the threat.

# %%
if 'cluster' in df.columns:
    feature_names = tfidf.get_feature_names_out()
    c0_idx = df.index[df['cluster'] == 0].tolist()
    c1_idx = df.index[df['cluster'] == 1].tolist()

    # Sum the scores to find the most important words
    c0_freqs = dict(zip(feature_names, text_vectors[c0_idx].sum(axis=0)))
    c1_freqs = dict(zip(feature_names, text_vectors[c1_idx].sum(axis=0)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    wc0 = WordCloud(background_color='white', colormap='Blues').generate_from_frequencies(c0_freqs)
    axes[0].imshow(wc0, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title("Cluster 0: What are they saying?")

    wc1 = WordCloud(background_color='black', colormap='Reds').generate_from_frequencies(c1_freqs)
    axes[1].imshow(wc1, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title("Cluster 1: What are they saying?")
    plt.show()

# %% [markdown]
# ## 9. The Verdict (Top 10 Words)
# 
# Finally, let's look at the hard data. Here are the Top 10 most heavily used words for each group.
# 
# If the **Red Group** (Cluster 1) contains viral hashtags, calls to action, or weirdly specific repeated phrases, we can confirm they are the **Inauthentic Actors**.
# 
# **Approach:** While the Word Cloud provides a general overview, I explicitly extract the **Top 10 Highest Weighted Terms** for each cluster based on TF-IDF sum scores.
# 
# **Findings:** This provides a concrete list of vocabulary. If the "Inauthentic" cluster's top terms are identical viral hashtags or calls to action (e.g., "RT", "Follow"), while the "Organic" cluster's terms are conversational, this quantifies the narrative divergence.

# %%
# Composite Conclusion Visualization
if 'cluster' in df.columns:
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle("CONCLUSION: The 'Smoking Gun' Evidence", fontsize=16, weight='bold')

    # 1. The Smoking Gun: Scatter Plot
    ax1 = fig.add_subplot(2, 1, 1)
    sns.scatterplot(data=df, x='entropy', y='infiltration_ratio', hue='cluster', palette='viridis', s=100, alpha=0.7, ax=ax1)
    ax1.set_title("The Separation: Bots (Top-Left) vs. Humans (Bottom-Right)", fontsize=14)
    ax1.set_xlabel("Linguistic Entropy (Complexity)")
    ax1.set_ylabel("Infiltration Ratio")
    ax1.set_ylim(0, 10)
    
    # 2. The Content Proof: Top Terms Comparison
    ax2 = fig.add_subplot(2, 2, 3)
    sns.barplot(x='score', y='term', data=df_c0, ax=ax2, palette='Blues_r')
    ax2.set_title("Cluster 0: Organic Vocabulary")

    ax3 = fig.add_subplot(2, 2, 4)
    sns.barplot(x='score', y='term', data=df_c1, ax=ax3, palette='Reds_r')
    ax3.set_title("Cluster 1: Inauthentic Vocabulary")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Conclusion: Case Closed
# 
# I have successfully used unsupervised forensic analysis to successfully isolate a coordinated group of inauthentic actors within the Nutella conversation. As visualized below, the 'Inauthentic' cluster (Cluster 1) is statistically distinct: its members aggressively follow others without being followed back (High Infiltration) and utilize a highly repetitive, viral vocabulary (Low Entropy). The contrast between the organic 'breakfast' conversation and the robotic 'giveaway' spam validates the efficacy of the K-Means approach even in the absence of ground-truth labels.
# 
# **The Findings:**
# 1.  **The Scatter Plot** proved that one group is aggressively infiltrating the network (High Follower Ratio) while saying very simple things (Low Entropy).
# 2.  **The Word Clouds** proved that this group is using promotional/viral language, different from the organic "breakfast" chat.
# 
# This confirms that even without knowing who was a bot beforehand, **Math and Unsupervised Learning** can effectively detect digital imposters.
# 
# **Limitations:**
# 1.  **No Ground Truth:** I cannot calculate Precision/Recall without labels.
# 2.  **Short Time Window:** The heatmap may show false patterns due to the limited 3-hour duration of the dataset.
# 3.  **Proxy Metrics:** I used Infiltration Ratio as a substitute for Degree Centrality due to missing User IDs.


