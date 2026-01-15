# 🍫 Social Bot Detection in Nutella Twitter Stream

## 📋 Research Problem & Proposed Solution

### Problem Statement
The integrity of social media discourse is increasingly compromised by **"Social Bots"**—algorithmic accounts designed to mimic human behavior to manipulate public opinion, inflate influence, or spread spam. While early automation was easily identifiable through high-volume spamming, modern bots have evolved into sophisticated "cyborgs" (Ferrara et al., 2016). These actors employ complex strategies such as organic sleep cycles, varied vocabulary, and **"infiltration" tactics**—following human users to solicit reciprocal follows.

A critical challenge in detecting these actors within the specific "Nutella" dataset provided is the lack of **"Ground Truth"**; there are no pre-existing labels indicating which users are bots and which are humans, rendering traditional supervised classification methods impossible.

### Proposed Solution: Multi-Modal Forensic Pipeline
I aim to solve this problem by implementing a **Multi-Modal Forensic Pipeline**. Instead of relying on a single metric, I triangulate bot behavior using three theoretical frameworks:

1. **Computational Linguistics** - to measure information density
2. **Network Statistics** - to detect structural infiltration  
3. **Unsupervised Learning** - to mathematically partition the dataset

By applying **K-Means Clustering**, I aim to mathematically partition the dataset into distinct behavioral groups, isolating the "inauthentic" actors based on their statistical distance from organic users.

### Research Questions
- **RQ1 (The Turing Test):** Can quantifiable linguistic features and temporal signatures reliably distinguish an automated account from a human in an unlabeled dataset?
- **RQ2 (The Infiltration Test):** Do specific clusters exhibit the "Infiltration" signature (a high ratio of friends to followers) indicative of spam behavior?

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/datawastheusualsuspect/INFO_287_Final.git
   cd INFO_287_Final
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install pandas numpy matplotlib seaborn textblob scikit-learn wordcloud scipy
   ```

### Running the Analysis

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook Nutella_Forensics.ipynb
```

#### Option 2: Python Script
```bash
python Final.py
```

#### Option 3: Comprehensive Analysis
```bash
python comprehensive_relationship_analysis.py
```

### Data Requirements
- Place `result_Nutella.csv` in the project root directory
- The dataset should contain Twitter data with columns: `text`, `created`, `friends`, `followers`, `retwc`

---

## 🔍 Key Findings & Analysis

### Methodology Overview
I employed **K-Means Clustering** as the primary method of analysis. This Unsupervised Learning technique was chosen specifically because the data is unlabeled. K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)** to find natural groupings in the data.

#### Mathematical Foundation
The K-Means objective function:

$$J = \sum_{j=1}^{k} \sum_{i=1}^{n} ||x_i^{(j)} - \mu_j||^2$$

Where $||x_i - \mu_j||^2$ is the **Euclidean Distance** between a data point $x_i$ and its cluster centroid $\mu_j$.

#### Feature Engineering
I engineered four specific predictors to drive the clustering:

1. **Linguistic Entropy (H):** Measures the randomness/complexity of the text
2. **Infiltration Ratio:** Calculated as $\frac{\text{Friends}}{\text{Followers} + 1}$
3. **Sentiment Polarity:** Measures emotional tone (-1 to +1)
4. **Log Retweet Count:** Measures content impact

### 🔬 Forensic Evidence

#### 1. Feature Distribution Analysis
![Feature Distributions](images/feature_distributions.png)

The feature engineering revealed distinct behavioral patterns:
- **Linguistic Entropy:** Bimodal distribution suggesting two user types
- **Infiltration Ratio:** Skewed distribution with extreme outliers (potential bots)
- **Sentiment:** Varied emotional tones across the dataset
- **Impact:** Log-transformed retweet counts showing engagement patterns

#### 2. Cluster Distribution & Temporal Analysis
![Cluster Distribution and Timeline](images/cluster_distribution_timeline.png)

The analysis revealed two distinct user groups:
- **Cluster 0:** 65.6% of tweets (presumed organic users)
- **Cluster 1:** 34.4% of tweets (suspected automated accounts)

**Temporal Pattern:** Activity shows both bursty and steady patterns, suggesting mixed human and bot behavior.

#### 3. Temporal Forensics: Cron Job Detection
![Temporal Heatmap](images/temporal_heatmap.png)

**Critical Finding:** The heatmap revealed **vertical stripes** at minutes :00, :15, and :30, indicating robotic scheduling (Cron jobs) rather than organic biological rhythms.

#### 4. PCA Cluster Visualization
![PCA Clusters](images/pca_clusters.png)

The 2D projection of the 504-dimensional feature space shows clear separation between the two clusters, validating the effectiveness of the unsupervised approach.

#### 5. Behavioral DNA: The "Smoking Gun" Evidence
![Behavioral DNA Analysis](images/behavioral_dna.png)

**The Bot Pattern:** The scatter plot revealed a distinct cluster in the **top-left quadrant** (High Infiltration, Low Entropy). This confirms the presence of **"Follow-Back Spam Bots"** who aggressively follow others but use repetitive templates.

**Radar Chart Insights:**
- **Cluster 0:** Balanced profile across all metrics
- **Cluster 1:** High infiltration ratio, low entropy signature

#### 6. Content Forensics: Vocabulary Analysis
![Word Clouds](images/word_clouds.png)

**Narrative Divergence:**
- **Cluster 0 (Organic):** Vocabulary focused on consumption ("Eat", "Toast", "Breakfast")
- **Cluster 1 (Inauthentic):** Vocabulary related to promotions ("Win", "Free", "Giveaway")

#### 7. Top Terms Quantitative Analysis
![Top Terms Analysis](images/top_terms.png)

**Cluster 0 (Organic) Top Terms:**
- nutella, love, breakfast, eat, toast, good, like, morning, coffee, time

**Cluster 1 (Inauthentic) Top Terms:**
- rt, follow, win, free, giveaway, nutella, contest, chance, enter, prize

#### 8. Conclusion: Composite Evidence
![Conclusion Composite](images/conclusion_composite.png)

**The Complete Picture:** The composite visualization shows the definitive separation between organic users and automated accounts, with clear behavioral and linguistic differences validating the bot detection methodology.

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **No Ground Truth:** Cannot calculate Precision/Recall without labels
2. **Short Time Window:** The heatmap may show false patterns due to the limited 3-hour duration
3. **Proxy Metrics:** Used Infiltration Ratio as a substitute for Degree Centrality due to missing User IDs

### Algorithm Evaluation
Since standard metrics like Accuracy or Recall cannot be calculated without ground truth, I evaluated the algorithm using **Qualitative Interpretation and Silhouette Logic**. The distinct separation of the clusters in the Scatter Plot and the coherent, distinct vocabularies observed in the comparative Word Clouds serve as validation that the algorithm successfully distinguished between two fundamentally different types of user behavior.

### Future Improvements
- Incorporate network analysis with complete user graphs
- Apply temporal clustering to detect coordinated campaigns
- Develop hybrid supervised-unsupervised approaches when partial labels become available
- Extend analysis to longer time periods for pattern validation

---

## 📚 References

- Ferrara, E., Varol, O., Davis, C., Menczer, F., & Flammini, A. (2016). The rise of social bots. Communications of the ACM, 59(7), 96-104.
- Varol, O., Ferrara, E., Davis, C. A., Menczer, F., & Flammini, A. (2017). Online human-bot interactions: Detection, estimation, and characterization. Proceedings of the International AAAI Conference on Web and Social Media, 11(1).

---

## 📊 Project Structure

```
INFO_287_Final/
├── README.md                          # This file
├── Final.py                           # Main analysis script
├── Nutella_Forensics.ipynb           # Interactive analysis notebook
├── comprehensive_relationship_analysis.py  # Extended analysis
├── Research Problem and Proposed Solution.txt  # Research methodology
├── result_Nutella.csv                 # Dataset (not tracked in git)
└── .gitignore                         # Git ignore configuration
```

---

## 🤝 Contributing

This project serves as a forensic analysis framework for social bot detection. Feel free to:
- Fork the repository
- Create issues for bugs or feature requests
- Submit pull requests for improvements
- Adapt the methodology to other datasets

---

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

---

**🔍 Conclusion:** This unsupervised forensic analysis successfully isolated a coordinated group of inauthentic actors within the Nutella conversation. The 'Inauthentic' cluster is statistically distinct: its members aggressively follow others without being followed back (High Infiltration) and utilize a highly repetitive, viral vocabulary (Low Entropy). The contrast between the organic 'breakfast' conversation and the robotic 'giveaway' spam validates the efficacy of the K-Means approach even in the absence of ground-truth labels.
