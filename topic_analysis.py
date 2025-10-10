"""
LLM Pandemic Policy Response Analysis
--------------------------------------
Performs topic modeling and sentiment analysis on LLM responses
to understand what factors and considerations models emphasize
in their pandemic policy decision-making.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from textblob import TextBlob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# === 1. LOAD DATA ===
def load_combined_data(filepath="combined_results/all_models_combined.csv"):
    """Load the combined dataset with response texts."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} responses from {len(df['model'].unique())} models")
    return df


# === 2. TEXT PREPROCESSING ===
def preprocess_text(text):
    """
    Clean and preprocess text for analysis.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep important terms
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def get_custom_stopwords():
    """
    Define custom stopwords including common words that don't add meaning.
    Uses a built-in list instead of NLTK to avoid SSL issues.
    """
    # Common English stopwords
    base_stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
        'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
        'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
        "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
        'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
        'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }
    
    # Add domain-specific and common filler words
    custom_stops = {
        'option', 'choose', 'select', 'would', 'however', 'although',
        'prefer', 'preferable', 'chosen', 'choice', 'selecting',
        'also', 'one', 'two', 'may', 'might', 'could', 'think',
        'believe', 'seems', 'appears', 'likely', 'even', 'though',
        'per', '100', 'people', 'next', 'month', 'businesses', 'business',
        'closed', 'infection', 'infected', 'rate', 'a', 'b', 'c', 'd',
        'e', 'f', 'g', 'h', 'i', 'option_a', 'option_b', 'letter',
        'scenarios', 'scenario', 'options', 'considering', 'given',
        'therefore', 'thus', 'hence', 'since', 'due', 'make', 'makes',
        'making', 'provide', 'provides', 'providing', 'way', 'ways'
    }
    
    return base_stopwords.union(custom_stops)


# === 3. KEYWORD EXTRACTION ===
def extract_key_themes(df, n_keywords=30):
    """
    Extract key themes and keywords from all responses using TF-IDF.
    """
    responses = df['response_text'].fillna('').apply(preprocess_text)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=list(get_custom_stopwords()),
        ngram_range=(1, 3),  # Include 1-3 word phrases
        min_df=3,  # Must appear in at least 3 documents
        max_df=0.7  # Can't appear in more than 70% of documents
    )
    
    tfidf_matrix = vectorizer.fit_transform(responses)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get average TF-IDF scores
    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    top_indices = avg_scores.argsort()[-n_keywords:][::-1]
    
    top_keywords = [(feature_names[i], avg_scores[i]) for i in top_indices]
    
    return top_keywords, vectorizer, tfidf_matrix


# === 4. TOPIC MODELING ===
def perform_topic_modeling(df, n_topics=5, n_words=10):
    """
    Perform LDA topic modeling on responses.
    """
    responses = df['response_text'].fillna('').apply(preprocess_text)
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_features=500,
        stop_words=list(get_custom_stopwords()),
        ngram_range=(1, 2),
        min_df=3
    )
    
    doc_term_matrix = vectorizer.fit_transform(responses)
    feature_names = vectorizer.get_feature_names_out()
    
    # LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method='batch'
    )
    
    lda.fit(doc_term_matrix)
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_num': topic_idx + 1,
            'keywords': top_words,
            'weights': [topic[i] for i in top_words_idx]
        })
    
    # Get document-topic distribution
    doc_topics = lda.transform(doc_term_matrix)
    
    return topics, doc_topics, lda


# === 5. KEYWORD FREQUENCY BY MODEL ===
def analyze_keywords_by_model(df, top_n=20):
    """
    Analyze which keywords/themes each model emphasizes.
    """
    models = df['model'].unique()
    model_keywords = {}
    
    for model in models:
        model_df = df[df['model'] == model]
        responses = model_df['response_text'].fillna('').apply(preprocess_text)
        
        # TF-IDF for this model
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words=list(get_custom_stopwords()),
            ngram_range=(1, 3),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(responses)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average scores
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            
            keywords = [(feature_names[i], avg_scores[i]) for i in top_indices]
            model_keywords[model] = keywords
        except:
            model_keywords[model] = []
    
    return model_keywords


# === 6. SENTIMENT ANALYSIS ===
def analyze_sentiment(df):
    """
    Analyze sentiment (polarity and subjectivity) of responses.
    """
    def get_sentiment(text):
        if pd.isna(text):
            return 0, 0
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0, 0
    
    sentiments = df['response_text'].apply(get_sentiment)
    df['polarity'] = sentiments.apply(lambda x: x[0])
    df['subjectivity'] = sentiments.apply(lambda x: x[1])
    
    return df


# === 7. CONCERN/FACTOR DETECTION ===
def detect_concerns(df):
    """
    Detect specific concerns and factors mentioned in responses.
    """
    concern_patterns = {
        'economic': r'\b(econom|business|financial|livelihood|job|income|revenue|work)\w*\b',
        'health': r'\b(health|safe|protect|prevent|spread|disease|infect)\w*\b',
        'mortality': r'\b(death|mortalit|fatal|die|lives|casualt)\w*\b',
        'mental_health': r'\b(mental|stress|anxiety|wellbeing|well-being|depression|isolation|psycholog)\w*\b',
        'freedom': r'\b(freedom|libert|rights|restriction|mandate|autonomy)\w*\b',
        'public_good': r'\b(public|community|collective|societal|common good|society)\w*\b',
        'balance': r'\b(balanc|tradeoff|trade-off|trade off|weigh|compromise|middle ground)\w*\b',
        'risk': r'\b(risk|danger|threat|hazard|vulnerable|peril)\w*\b',
        'capacity': r'\b(hospital|healthcare|health care|capacity|overwhelm|resource|medical)\w*\b',
        'vulnerable': r'\b(elderly|senior|vulnerable|immunocompromised|at-risk|at risk|frail|weak)\w*\b',
        'transmission': r'\b(transmit|transmiss|contagio|infect|spread)\w*\b',
        'severity': r'\b(sever|serious|critical|acute|mild|moderate)\w*\b'
    }
    
    for concern, pattern in concern_patterns.items():
        df[f'mentions_{concern}'] = df['response_text'].fillna('').str.contains(
            pattern, case=False, regex=True
        ).astype(int)
    
    return df


# === 8. VISUALIZATIONS ===
def create_topic_visualizations(df, topics, doc_topics, model_keywords, output_dir="analysis_output/topics"):
    """
    Create visualizations for topic and sentiment analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Word Cloud of Overall Themes
    all_text = ' '.join(df['response_text'].fillna('').apply(preprocess_text))
    
    if len(all_text.strip()) > 0:
        try:
            wordcloud = WordCloud(
                width=1600, 
                height=800,
                background_color='white',
                stopwords=get_custom_stopwords(),
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Overall Key Themes in LLM Responses', fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'overall_wordcloud.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir}/overall_wordcloud.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create overall wordcloud: {e}")
    
    # Figure 2: Topics Bar Chart
    fig, axes = plt.subplots(len(topics), 1, figsize=(14, 4*len(topics)))
    if len(topics) == 1:
        axes = [axes]
    
    for idx, topic in enumerate(topics):
        words = topic['keywords'][:10]
        weights = topic['weights'][:10]
        
        axes[idx].barh(words, weights, color=plt.cm.viridis(idx/len(topics)))
        axes[idx].set_xlabel('Weight', fontsize=11)
        axes[idx].set_title(f'Topic {topic["topic_num"]}: {", ".join(words[:5])}', 
                           fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
    
    plt.suptitle('Discovered Topics in LLM Responses (LDA)', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topics_lda.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/topics_lda.png")
    plt.close()
    
    # Figure 3: Model-Specific Word Clouds
    models = df['model'].unique()
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        model_text = ' '.join(
            df[df['model'] == model]['response_text'].fillna('').apply(preprocess_text)
        )
        
        if len(model_text.strip()) > 0:
            try:
                wc = WordCloud(
                    width=800, 
                    height=600,
                    background_color='white',
                    stopwords=get_custom_stopwords(),
                    colormap='plasma',
                    max_words=50,
                    relative_scaling=0.5,
                    min_font_size=8
                ).generate(model_text)
                
                axes[idx].imshow(wc, interpolation='bilinear')
                axes[idx].axis('off')
                axes[idx].set_title(f'{model}', fontsize=13, fontweight='bold')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error creating wordcloud\nfor {model}', 
                             ha='center', va='center')
                axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Key Themes by Model', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_wordclouds.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_wordclouds.png")
    plt.close()
    
    # Figure 4: Top Keywords by Model (Bar Chart)
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        if model in model_keywords and len(model_keywords[model]) > 0:
            keywords = model_keywords[model][:15]
            words = [k[0] for k in keywords]
            scores = [k[1] for k in keywords]
            
            axes[idx].barh(words, scores, color=plt.cm.plasma(idx/n_models))
            axes[idx].set_xlabel('TF-IDF Score', fontsize=10)
            axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Top Keywords by Model (TF-IDF)', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_keywords.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_keywords.png")
    plt.close()


def create_sentiment_visualizations(df, output_dir="analysis_output/topics"):
    """
    Create sentiment analysis visualizations.
    """
    # Figure 1: Sentiment Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Polarity
    axes[0].hist(df['polarity'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['polarity'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {df["polarity"].mean():.3f}')
    axes[0].set_xlabel('Polarity (Negative ← → Positive)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Sentiment Polarity Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    
    # Subjectivity
    axes[1].hist(df['subjectivity'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].axvline(df['subjectivity'].mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Mean: {df["subjectivity"].mean():.3f}')
    axes[1].set_xlabel('Subjectivity (Objective ← → Subjective)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Sentiment Subjectivity Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    
    plt.suptitle('Overall Sentiment Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/sentiment_distribution.png")
    plt.close()
    
    # Figure 2: Sentiment by Model
    models = df['model'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Polarity by model
    model_polarity = df.groupby('model')['polarity'].mean().sort_values()
    axes[0].barh(model_polarity.index, model_polarity.values, color='skyblue', edgecolor='black')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel('Mean Polarity', fontsize=11)
    axes[0].set_title('Average Sentiment Polarity by Model', fontsize=13, fontweight='bold')
    
    for i, v in enumerate(model_polarity.values):
        axes[0].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Subjectivity by model
    model_subj = df.groupby('model')['subjectivity'].mean().sort_values()
    axes[1].barh(model_subj.index, model_subj.values, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Mean Subjectivity', fontsize=11)
    axes[1].set_title('Average Subjectivity by Model', fontsize=13, fontweight='bold')
    
    for i, v in enumerate(model_subj.values):
        axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_model.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/sentiment_by_model.png")
    plt.close()


def create_concern_visualizations(df, output_dir="analysis_output/topics"):
    """
    Visualize which concerns/factors models mention.
    """
    concern_cols = [col for col in df.columns if col.startswith('mentions_')]
    concern_names = [col.replace('mentions_', '').replace('_', ' ').title() 
                    for col in concern_cols]
    
    # Overall concern frequencies
    overall_concerns = df[concern_cols].sum().sort_values(ascending=False)
    overall_concerns.index = [name.replace('mentions_', '').replace('_', ' ').title() 
                             for name in overall_concerns.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(overall_concerns.index, overall_concerns.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(overall_concerns))))
    ax.set_xlabel('Number of Mentions', fontsize=12, fontweight='bold')
    ax.set_title('Frequency of Concerns/Factors Mentioned Across All Models', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, v) in enumerate(zip(bars, overall_concerns.values)):
        pct = (v / len(df)) * 100
        ax.text(v + 5, i, f'{int(v)} ({pct:.1f}%)', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_concerns.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/overall_concerns.png")
    plt.close()
    
    # Concerns by model (heatmap)
    models = df['model'].unique()
    concern_matrix = []
    
    for model in models:
        model_df = df[df['model'] == model]
        model_concerns = model_df[concern_cols].sum()
        # Convert to percentage
        model_concerns_pct = (model_concerns / len(model_df)) * 100
        concern_matrix.append(model_concerns_pct.values)
    
    concern_df = pd.DataFrame(
        concern_matrix,
        index=models,
        columns=concern_names
    )
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.5)))
    sns.heatmap(concern_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Percentage of Responses (%)'}, ax=ax)
    ax.set_title('Concerns/Factors Mentioned by Model', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Concern Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concerns_by_model_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/concerns_by_model_heatmap.png")
    plt.close()


# === 9. SUMMARY REPORTS ===
def print_topic_summary(topics, model_keywords):
    """
    Print summary of discovered topics and model-specific keywords.
    """
    print("\n" + "="*80)
    print("TOPIC ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n--- Discovered Topics (LDA) ---")
    for topic in topics:
        print(f"\nTopic {topic['topic_num']}:")
        print(f"  Keywords: {', '.join(topic['keywords'][:10])}")
    
    print("\n--- Top Keywords by Model ---")
    for model, keywords in model_keywords.items():
        if len(keywords) > 0:
            print(f"\n{model}:")
            top_10 = keywords[:10]
            print(f"  {', '.join([k[0] for k in top_10])}")
    
    print("\n" + "="*80)


def print_sentiment_summary(df):
    """
    Print sentiment analysis summary.
    """
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n--- Overall Sentiment ---")
    print(f"Mean Polarity: {df['polarity'].mean():.4f} (range: -1 to 1)")
    print(f"Mean Subjectivity: {df['subjectivity'].mean():.4f} (range: 0 to 1)")
    
    print("\n--- Sentiment by Model ---")
    print(f"{'Model':<30} {'Polarity':<15} {'Subjectivity':<15}")
    print("-" * 60)
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        pol = model_df['polarity'].mean()
        subj = model_df['subjectivity'].mean()
        print(f"{model:<30} {pol:<15.4f} {subj:<15.4f}")
    
    print("\n" + "="*80)


def print_concern_summary(df):
    """
    Print summary of concerns mentioned.
    """
    concern_cols = [col for col in df.columns if col.startswith('mentions_')]
    
    print("\n" + "="*80)
    print("CONCERN/FACTOR ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n--- Overall Concern Frequencies ---")
    print(f"{'Concern':<30} {'Mentions':<15} {'Percentage':<15}")
    print("-" * 60)
    
    for col in concern_cols:
        concern_name = col.replace('mentions_', '').replace('_', ' ').title()
        count = df[col].sum()
        pct = (count / len(df)) * 100
        print(f"{concern_name:<30} {int(count):<15} {pct:<15.1f}%")
    
    print("\n" + "="*80)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("="*80)
    print("LLM RESPONSE TOPIC & SENTIMENT ANALYSIS")
    print("="*80)
    
    # Step 1: Load data
    print("\nStep 1: Loading combined data...")
    df = load_combined_data("combined_results/all_models_combined.csv")
    
    # Step 2: Extract keywords and themes
    print("\nStep 2: Extracting key themes and keywords...")
    top_keywords, vectorizer, tfidf_matrix = extract_key_themes(df)
    
    # Step 3: Topic modeling
    print("\nStep 3: Performing topic modeling...")
    topics, doc_topics, lda_model = perform_topic_modeling(df, n_topics=5)
    
    # Step 4: Model-specific keyword analysis
    print("\nStep 4: Analyzing keywords by model...")
    model_keywords = analyze_keywords_by_model(df)
    
    # Step 5: Sentiment analysis
    print("\nStep 5: Performing sentiment analysis...")
    df = analyze_sentiment(df)
    
    # Step 6: Detect concerns/factors
    print("\nStep 6: Detecting concerns and factors...")
    df = detect_concerns(df)
    
    # Step 7: Create visualizations
    print("\nStep 7: Creating visualizations...")
    create_topic_visualizations(df, topics, doc_topics, model_keywords)
    create_sentiment_visualizations(df)
    create_concern_visualizations(df)
    
    # Step 8: Save enriched dataset
    df.to_csv("analysis_output/topics/enriched_data_with_topics.csv", index=False)
    print(f"\nSaved enriched dataset: analysis_output/topics/enriched_data_with_topics.csv")
    
    # Step 9: Print summaries
    print_topic_summary(topics, model_keywords)
    print_sentiment_summary(df)
    print_concern_summary(df)
    
    print("\n✅ Topic and sentiment analysis complete!")
    print("   Check 'analysis_output/topics/' for all visualizations.")