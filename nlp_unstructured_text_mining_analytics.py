# ============================================================================
# UNSTRUCTURED TEXT MINING & SENTIMENT ANALYSIS PIPELINE
# Python / Google Colab
# Focus: Lexicon-based sentiment scoring and topic modeling
# ============================================================================

# ============================================================================
# 1. DEPENDENCIES & ENVIRONMENT INITIALIZATION
# ============================================================================
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import os

# Download required text analysis packs
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define relative pathing to target dataset
DATA_PATH = "restaurant_reviews.csv"

# Verify file existence and initialize the main data frame structure
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Data ingested successfully from local path: '{DATA_PATH}'")
    print(f"Total dataset size: {len(df)} customer reviews.\n")
    print("Execution Preview:")
    print(df.head())
else:
    raise FileNotFoundError(f"Target data resource not detected at path: '{DATA_PATH}'")


# ============================================================================
# 2. SENTIMENT PROFILING VIA VALENCE DICTIONARY MATCHING
# ============================================================================
# Initialize the VADER sentiment tool
analyzer = SentimentIntensityAnalyzer()

# Calculate the continuous compound score for each text review row
df['compound_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Sort the continuous scores into distinct Positive, Negative, and Neutral categories
df['sentiment_category'] = df['compound_score'].apply(
    lambda x: 'Negative' if x <= -0.05 else ('Positive' if x >= 0.05 else 'Neutral')
)

# Compute the average sentiment scores across different restaurants
print("--- Average Sentiment Score by Restaurant ---")
print(df.groupby('restaurant')['compound_score'].mean().sort_values(ascending=False))

# Count review volumes grouped by restaurant and sentiment class
print("\n--- Total Review Count by Sentiment Category ---")
print(df.groupby(['restaurant', 'sentiment_category']).size().unstack(fill_value=0)) 


# ============================================================================
# 3. LINGUISTIC REGULARIZATION & STOPWORD FILTRATION
# ============================================================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Exclude common, unhelpful words to keep the vocabulary clean and focused
custom_stops = ['poke', 'bowl', 'food', 'restaurant', 'aloha', 'paradise', 'island', 'fresh']

def clean_text(text):
    # Enforce lowercase uniformity
    text = str(text).lower()
    # Strip out punctuation marks and special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Break text sentences down into individual words
    tokens = word_tokenize(text)
    # Reduce words to their base dictionary form and filter out stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t not in custom_stops]
    return " ".join(tokens)

# Run raw data text through the cleaning pipeline
df['clean_review'] = df['review'].apply(clean_text)
print("Text pre-processing complete. Ready for modeling.") 


# ============================================================================
# 4. FILTERING SPECIFIC NEGATIVE COMPLAINTS
# ============================================================================
print("--- Exact Negative Reviews for Aloha Bowl ---")

# Isolate the raw text for reviews that match specific filtering criteria
aloha_neg_raw = df[(df['restaurant'] == 'Aloha Bowl') & (df['sentiment_category'] == 'Negative')]['review']

if not aloha_neg_raw.empty:
    for idx, review in enumerate(aloha_neg_raw):
         print(f"Complaint {idx + 1}: {review}\n")
else:
    print("No negative reviews found.") 


# ============================================================================
# 5. TOPIC MODELING FOR MENU TRENDS (LDA)
# ============================================================================
print("--- LDA Topics for All Positive Reviews (Menu Inspiration) ---")

# Pull out a subset of reviews that match the positive sentiment category
positive_reviews = df[df['sentiment_category'] == 'Positive']['clean_review'].tolist()

if positive_reviews:
    # Convert the clean text into a matrix of word frequencies
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(positive_reviews)

    # Train the unsupervised LDA model to sort words into 3 main themes
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()

    # Extract the top 8 words associated with each underlying topic
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-8:]]
        print(f"Topic {idx + 1}: {', '.join(top_words)}")
else:
    print("Not enough positive data to extract topics.") 


# ============================================================================
# 6. ASPECT-BASED FREQUENCY MATRIX COMPILATION
# ============================================================================
print("--- Competitive Advantage: Keyword Mentions by Aspect ---")

# Map tracking keywords to specific operational business categories
aspects = {
    'Service / Logistics': ['staff', 'friendly', 'wait', 'service', 'time', 'line', 'quick', 'fast', 'slow'],
    'Food Quality / Menu': ['flavor', 'spicy', 'fresh', 'portion', 'sauce', 'tuna', 'rice', 'delicious', 'sweet'],
    'Value / Price': ['price', 'expensive', 'worth', 'cost', 'deal', 'cheap', 'money']
}

def count_aspect(text, keywords):
    tokens = str(text).split()
    # Count how many target keywords appear in a single review string
    return sum(1 for word in tokens if word in keywords)

# Generate new dataset attributes for each category count vector
for aspect, words in aspects.items():
    df[aspect] = df['clean_review'].apply(lambda x: count_aspect(x, words))

# Aggregate keyword frequencies across different restaurant groupings
aspect_summary = df.groupby('restaurant')[['Service / Logistics', 'Food Quality / Menu', 'Value / Price']].sum()
print(aspect_summary) 


# ============================================================================
# 7. PLOT GENERATION: VALENCE DISTRIBUTIONS
# ============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate average compound metrics to prepare coordinates for plotting
sentiment_avg = df.groupby('restaurant')['compound_score'].mean().reset_index()
sentiment_avg = sentiment_avg.sort_values(by='compound_score', ascending=False)

# Configure chart style and canvas limits
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# Render data categories as distinct bar columns
ax = sns.barplot(x='restaurant', y='compound_score', data=sentiment_avg, palette=['#2ecc71', '#f1c40f', '#e74c3c'])

# Apply structural axis labels and chart titles
plt.title('Average Customer Sentiment by Restaurant', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Average Compound VADER Score', fontsize=12)
plt.xlabel('')
plt.ylim(0, 0.6) 

# Add summary data numbers directly on top of visual elements
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 8),
                textcoords='offset points')

plt.tight_layout()
plt.show()


# ============================================================================
# 8. PLOT GENERATION: MULTIVARIATE INTERSECTION TRAILS
# ============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Flatten dataset summary layouts from tables into long data frames
aspect_summary_reset = aspect_summary.reset_index()
aspect_melted = aspect_summary_reset.melt(id_vars='restaurant',
                                          var_name='Business Aspect',
                                          value_name='Total Mentions')

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Build a clustered chart comparing volume categories side-by-side
ax = sns.barplot(x='Business Aspect', y='Total Mentions', hue='restaurant',
                 data=aspect_melted, palette='viridis')

# Format plot titles, axis parameters, and placement legends
plt.title('Customer Engagement: Keyword Mentions by Aspect', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Total Number of Keyword Mentions', fontsize=12)
plt.xlabel('')
plt.legend(title='Restaurant', fontsize=11, title_fontsize=12)

# Label elements with specific values where metrics track above zero
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{int(height)}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 6),
                    textcoords='offset points')

plt.tight_layout()
plt.show()
