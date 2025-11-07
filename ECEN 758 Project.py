#ECEN 758 Project Group 7


#***********************************data download******************************

import os, tarfile, pandas as pd, gdown
from pathlib import Path

#find the users downloads folder
downloads_folder = Path.home() / "Downloads"

#creates our project folder for storing data that we have to download
ROOT = downloads_folder / "Amazon Data"
os.makedirs(ROOT, exist_ok=True)
print("Dataset will be stored at:", ROOT)

#Download the official AmazonReviewFull tar.gz from Google Drive
url_id = "0Bz8a_Dbh9QhbZVhsUnRWRDhETzA"   #same ID torchtext uses
tar_path = os.path.join(ROOT, "amazon_review_full_csv.tar.gz") #checks if zip file is there
if not os.path.exists(tar_path): # if file not there then it will begin to download file to the root directory
    print("Downloading dataset tar…")
    gdown.download(id=url_id, output=tar_path, quiet=False, resume=True)

#Extract zip file if needed, skipped if already done
extract_dir = os.path.join(ROOT, "amazon_review_full_csv")
train_csv = os.path.join(extract_dir, "train.csv")
test_csv  = os.path.join(extract_dir, "test.csv")
if not os.path.exists(train_csv):
    print("Extracting tar…")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=ROOT)

#confirms both files are in folder to user
train_path = os.path.join(ROOT, "amazon_review_full_csv", "train.csv")
test_path = os.path.join(ROOT, "amazon_review_full_csv", "test.csv")
print("Train file exists:", os.path.exists(train_path))
print("Test file exists:", os.path.exists(test_path))

#Count lines in each file 
with open(train_path, "r", encoding="utf-8") as f:
    n_lines = sum(1 for _ in f)
print("Train.csv lines:", n_lines)

with open(test_path, "r", encoding="utf-8") as f:
    n_lines = sum(1 for _ in f)
print("Test.csv lines:", n_lines)




#******************************Data cleansing****************************************

import re

#Ask the user how many rows to load for train/val
try:
    num_rows = int(input("Enter how many rows you want to load from Train.csv(max 3000000): "))
except ValueError:
    print("Invalid input. Defaulting to 50000 rows")
    num_rows = 50000

#Ask the user how many rows to load for test
try:
    test_rows = int(input("Enter how many rows to load from test.csv (max 650000): "))
except ValueError:
    print("Invalid input. Defaulting to 10000 rows")
    test_rows = 10000


#Load that many rows
cols = ["label", "title", "review"]
df = pd.read_csv(train_csv, header=None, names=cols, nrows=num_rows)
test_df = pd.read_csv(test_csv, header=None, names=cols, nrows=test_rows)
print(f"Loaded {len(df)} rows from train.csv")
print(f"Loaded {len(test_df)} rows from test.csv")

#Combine title + review into one text column
df["text"] = df["title"].fillna("") + " " + df["review"].fillna("")
test_df["text"] = test_df["title"].fillna("") + " " + test_df["review"].fillna("")

#Convert everything to lowercase
df["text"] = df["text"].str.lower()
test_df["text"] = test_df["text"].str.lower()

#Remove punctuation but keep letters, numbers, and space
df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-z0-9\s]", "", x))
test_df["text"] = test_df["text"].apply(lambda x: re.sub(r"[^a-z0-9\s]", "", x))
label_test = test_df["label"].to_numpy()







#***************************Splitting train data to train/Validation**********************************

import numpy as np
from sklearn.model_selection import train_test_split
print("\nTrain/Validation Split:")

#Keep labels as 1–5
df["label"] = df["label"].astype(int)
label_full = df["label"].to_numpy() 

# Make a validation split from the train dataframe
try:
    val_frac = float(input("Enter validation fraction(0-1): "))
    if not (0 < val_frac < 1):
        raise ValueError
except ValueError:
    print("Invalid input, defaulting to 0.2")
    val_frac = 0.2

data_train_text, data_val_text, label_train, label_val = train_test_split(
    df["text"].values, label_full, test_size=val_frac, random_state=42, stratify=label_full)
print(f"Train texts: {len(data_train_text)} | Val texts: {len(data_val_text)}")
print(f"Label range: {min(label_full)}–{max(label_full)}")




#*************************************Tokenizing data*****************************************

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
print(f"\nTokenizing and padding:")

#Ask user for vocab size, how many unique words to keep 
try:
    vocab_size = int(input("Enter vocab size (e.g. 10000): "))
except ValueError:
    print("Invalid input. Defaulting to 10000.")
    vocab_size = 10000

#Ask user for maxlen, how long each review should be after padding
try:
    maxlen = int(input("Enter max sequence length (e.g. 200): "))
except ValueError:
    print("Invalid input. Defaulting to 200.")
    maxlen = 200


#Fit tokenizer on TRAIN DATA ONLY since its now allowed to have val or test data in here
tokenizer = Tokenizer(num_words=vocab_size, oov_token="(OOV)")
tokenizer.fit_on_texts(data_train_text)

#Convert text to numeric sequences
seq_train = tokenizer.texts_to_sequences(data_train_text)
seq_val   = tokenizer.texts_to_sequences(data_val_text)
seq_test  = tokenizer.texts_to_sequences(test_df["text"].values)

#Pad/cut sequences to fixed length
data_train = pad_sequences(seq_train, maxlen=maxlen, padding="post", truncating="post")
data_val   = pad_sequences(seq_val,   maxlen=maxlen, padding="post", truncating="post")
data_test  = pad_sequences(seq_test,  maxlen=maxlen, padding="post", truncating="post")

print(f"\nData Sizes(rows, sequence length):  data_train: {data_train.shape}, data_val: {data_val.shape}, data_test: {data_test.shape}")
print(f"\nLabel Sizes(rows):  label_train: {label_train.shape}, label_val: {label_val.shape}, label_test: {label_test.shape}")
print("Label ranges -> train:", label_train.min(), "-", label_train.max(), "| val:", label_val.min(), "-", label_val.max(),
    "| test:", label_test.min(), "-", label_test.max())





#***************************************Exploratory Data Analysis***************************
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("\nEXPLORATORY DATA ANALYSIS")
print("="*80)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

#***********************************Rating Distribution**************************************
print("\n1. RATING DISTRIBUTION ANALYSIS")
print("-"*80) 
#Analyze original dataframe
print("\nOverall Dataset (Before Split):")
rating_counts = df["label"].value_counts().sort_index()
for rating, count in rating_counts.items():
    percentage = (count / len(df)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  Rating {rating}: {count:>6,} reviews ({percentage:>5.2f}%) {bar}")

print("\nTraining Set:")
train_rating_counts = pd.Series(label_train).value_counts().sort_index()
for rating, count in train_rating_counts.items():
    percentage = (count / len(label_train)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  Rating {rating}: {count:>6,} reviews ({percentage:>5.2f}%) {bar}")

print("\nValidation Set:")
val_rating_counts = pd.Series(label_val).value_counts().sort_index()
for rating, count in val_rating_counts.items():
    percentage = (count / len(label_val)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  Rating {rating}: {count:>6,} reviews ({percentage:>5.2f}%) {bar}")

print("\nTest Set:")
test_rating_counts = pd.Series(label_test).value_counts().sort_index()
for rating, count in test_rating_counts.items():
    percentage = (count / len(label_test)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  Rating {rating}: {count:>6,} reviews ({percentage:>5.2f}%) {bar}")

#**********************************Text Length Statistics***************************************
print("\n2. TEXT LENGTH STATISTICS")
print("-"*80)
# Calculate text lengths for original text
df['text_length'] = df['text'].apply(lambda x: len(x.split()))
test_df['text_length'] = test_df['text'].apply(lambda x: len(x.split()))

print("\nTraining Data (Original Text):")
train_texts_df = pd.DataFrame({'text': data_train_text})
train_texts_df['length'] = train_texts_df['text'].apply(lambda x: len(x.split()))
print(f"  Mean length: {train_texts_df['length'].mean():.2f} words")
print(f"  Median length: {train_texts_df['length'].median():.2f} words")
print(f"  Min length: {train_texts_df['length'].min()} words")
print(f"  Max length: {train_texts_df['length'].max()} words")
print(f"  Std deviation: {train_texts_df['length'].std():.2f} words")

print("\nTest Data (Original Text):")
print(f"  Mean length: {test_df['text_length'].mean():.2f} words")
print(f"  Median length: {test_df['text_length'].median():.2f} words")
print(f"  Min length: {test_df['text_length'].min()} words")
print(f"  Max length: {test_df['text_length'].max()} words")
print(f"  Std deviation: {test_df['text_length'].std():.2f} words")

print(f"\nTokenized Sequence Length (after padding/truncating): {maxlen} words")

#***************************************Most Common Words Analysis***************************
print("\n3. MOST COMMON WORDS ANALYSIS")
print("-"*80)
# Get word frequency from training text
all_train_words = []
for text in data_train_text:
    all_train_words.extend(text.split())

word_freq = Counter(all_train_words)
most_common_words = word_freq.most_common(20)

print("\nTop 20 Most Common Words in Training Data:")
for idx, (word, count) in enumerate(most_common_words, 1):
    print(f"  {idx:>2}. '{word:<15}': {count:>7,} occurrences")

#***************************************N-Gram Phrase Analysis***************************
print("\n4. N-GRAM PHRASE ANALYSIS")
print("-"*80)

# Overall most common bigrams(2 word phrases) and trigrams(3 words phrases)
print("\nTop 15 Most Common 2-Word Phrases (Bigrams):")
try:
    vectorizer_bigram = CountVectorizer(
        ngram_range=(2, 2),
        max_features=15,
        token_pattern=r'\b\w+\b'
    )
    bigram_matrix = vectorizer_bigram.fit_transform(data_train_text)
    bigrams = vectorizer_bigram.get_feature_names_out()
    bigram_counts = bigram_matrix.sum(axis=0).A1
    
    bigram_df = pd.DataFrame({
        'phrase': bigrams,
        'count': bigram_counts
    }).sort_values('count', ascending=False)
    
    for idx, row in bigram_df.iterrows():
        print(f"  - '{row['phrase']}': {row['count']:,} occurrences")
except Exception as e:
    print(f"Error analyzing bigrams: {e}")

print("\nTop 15 Most Common 3-Word Phrases (Trigrams):")
try:
    vectorizer_trigram = CountVectorizer(
        ngram_range=(3, 3),
        max_features=15,
        token_pattern=r'\b\w+\b'
    )
    trigram_matrix = vectorizer_trigram.fit_transform(data_train_text)
    trigrams = vectorizer_trigram.get_feature_names_out()
    trigram_counts = trigram_matrix.sum(axis=0).A1
    
    trigram_df = pd.DataFrame({
        'phrase': trigrams,
        'count': trigram_counts
    }).sort_values('count', ascending=False)
    
    for idx, row in trigram_df.iterrows():
        print(f"  - '{row['phrase']}': {row['count']:,} occurrences")
except Exception as e:
    print(f"Error analyzing trigrams: {e}")

#***************************************Phrases by Rating***************************
print("\n5. TOP PHRASES BY RATING CATEGORY")
print("-"*80)

# Create dataframe with training text and labels
train_analysis_df = pd.DataFrame({
    'text': data_train_text,
    'label': label_train
})

for rating in sorted(train_analysis_df['label'].unique()):
    print(f"\n  Rating {rating} - Top 5 Bigrams:")
    rating_texts = train_analysis_df[train_analysis_df['label'] == rating]['text']
    
    try:
        vec = CountVectorizer(
            ngram_range=(2, 2),
            max_features=5,
            token_pattern=r'\b\w+\b',
            min_df=2
        )
        dt = vec.fit_transform(rating_texts)
        phrases = vec.get_feature_names_out()
        counts = dt.sum(axis=0).A1
        
        phrase_count_pairs = sorted(zip(phrases, counts), key=lambda x: x[1], reverse=True)
        for phrase, count in phrase_count_pairs:
            print(f"    - '{phrase}': {count:,}")
    except Exception as e:
        print(f"Error in phrases by rating: {e}")
#***************************************Data Visualizations***************************
print("\n8. VISUALIZATIONS")
print("-"*80)

# Create comprehensive visualization figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Rating Distribution for Train/Val/Test
ax1 = fig.add_subplot(gs[0, 0])
rating_data = pd.DataFrame({
    'Train': train_rating_counts,
    'Validation': val_rating_counts,
    'Test': test_rating_counts
})
rating_data.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax1.set_xlabel('Rating', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
ax1.set_title('Rating Distribution Across Splits', fontsize=14, fontweight='bold')
ax1.legend(title='Dataset', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

# Plot 2: Rating Distribution Percentages (Stacked)
ax2 = fig.add_subplot(gs[0, 1])
train_pct = (train_rating_counts / train_rating_counts.sum() * 100).values
val_pct = (val_rating_counts / val_rating_counts.sum() * 100).values
test_pct = (test_rating_counts / test_rating_counts.sum() * 100).values

x = np.arange(len(train_rating_counts))
width = 0.25

ax2.bar(x - width, train_pct, width, label='Train', color='#1f77b4')
ax2.bar(x, val_pct, width, label='Validation', color='#ff7f0e')
ax2.bar(x + width, test_pct, width, label='Test', color='#2ca02c')

ax2.set_xlabel('Rating', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Rating Distribution (Percentage)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(train_rating_counts.index)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Top 10 Most Common Words
ax3 = fig.add_subplot(gs[1, 0])
top_10_words = most_common_words[:10]
words, counts = zip(*top_10_words)
y_pos = np.arange(len(words))
ax3.barh(y_pos, counts, color='steelblue')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(words)
ax3.invert_yaxis()
ax3.set_xlabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 Most Common Words', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Text Length Distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(train_texts_df['length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(train_texts_df['length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train_texts_df["length"].mean():.1f}')
ax4.axvline(train_texts_df['length'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {train_texts_df["length"].median():.1f}')
ax4.axvline(maxlen, color='orange', linestyle='--', linewidth=2, label=f'Max Length: {maxlen}')
ax4.set_xlabel('Text Length (words)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('Distribution of Text Lengths', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Top 10 Bigrams
ax5 = fig.add_subplot(gs[2, 0])
if len(bigram_df) > 0:
    top_10_bigrams = bigram_df.head(10)
    y_pos = np.arange(len(top_10_bigrams))
    ax5.barh(y_pos, top_10_bigrams['count'].values, color='coral')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(top_10_bigrams['phrase'].values)
    ax5.invert_yaxis()
    ax5.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('Top 10 Most Common Bigrams', fontsize=14, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)

# Plot 6: Sequence Length Distribution (before padding)
seq_train_lengths=[len(seq) for seq in seq_train]
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(seq_train_lengths, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax6.axvline(np.mean(seq_train_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(seq_train_lengths):.1f}')
ax6.axvline(maxlen, color='orange', linestyle='--', linewidth=2, label=f'Padded Length: {maxlen}')
ax6.set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title('Distribution of Sequence Lengths (Before Padding)', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('Amazon Reviews Dataset - Exploratory Data Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.show()
# Save figure
eda_output_path = os.path.join(ROOT, "eda_visualization.png")
try:
    plt.savefig(eda_output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {eda_output_path}")
except Exception as e:
    print(f"Cannot save the plot: {e}")
