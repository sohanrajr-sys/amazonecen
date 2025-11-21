import pandas as pd
test=pd.read_csv('test.csv')

import re


def clean_text(text: str) -> str:
    """
    Clean review text for transformer fine-tuning.
    Keeps semantic content intact.
    """
    # 1. Lowercase (optional for uncased models)
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 4. Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # 5. Remove special characters and digits (keep punctuation)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\[\] ]", " ", text)

    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Handle long repeated characters (like “cooooool”)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    return text

test.columns = ['rating', 'short_text', 'long_text']
test['long_text'] = test['long_text'].astype(str).apply(clean_text)
test['short_text'] = test['short_text'].astype(str).apply(clean_text)
test['rating']=test['rating'].astype(int)-1

# ============================================================================
# USER INPUT: NUMBER OF SAMPLES PER CLASS
# ============================================================================
print("\n" + "="*70)
print("TEST DATASET CONFIGURATION")
print("="*70)

while True:
    try:
        num_samples_per_class = int(input("Enter number of samples per class to test (e.g., 100, 250, 500): "))
        if num_samples_per_class <= 0:
            print("Error: Number of samples must be positive. Please try again.")
            continue
        break
    except ValueError:
        print("Error: Please enter a valid integer.")

print(f"\nSelected: {num_samples_per_class} samples per class")
print(f"Total test samples: {num_samples_per_class * 5}")

# code to get balanced samples from test with equal no. of classes
df_list = []
for i in range(5):
    df_class = test[test['rating'] == i].sample(n=num_samples_per_class, random_state=42)
    df_list.append(df_class)
balanced_test = pd.concat(df_list).reset_index(drop=True)

# ============================================================================
# MODEL INFERENCE AND ANALYTICS
# ============================================================================

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# 1. LOAD MODEL AND TOKENIZER
# ============================================================================

print("Loading model and tokenizer from epoch2 folder (or Hugging Face identifier)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use Auto classes so this can load DeBERTa (or any compatible) model saved in ./epoch2
model = AutoModelForSequenceClassification.from_pretrained('./epoch2')
tokenizer = AutoTokenizer.from_pretrained('./epoch2', use_fast=True)
model.to(device)
model.eval()

print("Model and tokenizer loaded successfully!")
print(f"Model config: {model.config}")

# ============================================================================
# 2. PREPARE TEST DATASET FOR INFERENCE
# ============================================================================
print("\nPreparing test dataset for inference...")

texts = balanced_test['long_text'].tolist()
true_labels = balanced_test['rating'].tolist()

# Tokenize all texts
print("Tokenizing test dataset...")
# Respect tokenizer/model max length (DeBERTa typically uses 512). Use tokenizer.model_max_length when available.
model_max_length = getattr(tokenizer, "model_max_length", None)
if model_max_length is None or model_max_length <= 0:
    model_max_length = getattr(model.config, "max_position_embeddings", 512)
# Cap max length to a reasonable number (avoid huge values)
max_length = min(int(model_max_length), 4096)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors='pt'
)

input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

# ============================================================================
# 3. RUN INFERENCE
# ============================================================================
print(f"\nRunning inference on {len(texts)} test samples with parallel batch processing...")

all_logits = []
all_predictions = []
all_probabilities = []

batch_size = 8
num_batches = (len(texts) + batch_size - 1) // batch_size

def process_batch(batch_idx):
    """Process a single batch and return results"""
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(texts))
    
    batch_input_ids = input_ids[start_idx:end_idx]
    batch_attention_mask = attention_mask[start_idx:end_idx]
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    return {
        'batch_idx': batch_idx,
        'logits': logits.cpu().numpy(),
        'probabilities': probabilities.cpu().numpy(),
        'predictions': predictions.cpu().numpy()
    }

# Run inference in parallel
batch_results = {}
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(process_batch, i): i 
        for i in range(num_batches)
    }
    
    for future in tqdm(as_completed(futures), total=num_batches, desc="Inference"):
        result = future.result()
        batch_results[result['batch_idx']] = result

# Reconstruct results in order
for batch_idx in sorted(batch_results.keys()):
    result = batch_results[batch_idx]
    all_logits.append(result['logits'])
    all_probabilities.append(result['probabilities'])
    all_predictions.extend(result['predictions'].tolist())

# Concatenate all results
all_logits = np.concatenate(all_logits, axis=0)
all_probabilities = np.concatenate(all_probabilities, axis=0)

print("Inference completed!")


# ============================================================================
# 4. CALCULATE METRICS
# ============================================================================
print("\n" + "="*70)
print("CLASSIFICATION METRICS")
print("="*70)

# Basic metrics
accuracy = accuracy_score(true_labels, all_predictions)
print(f"\nAccuracy: {accuracy:.4f}")

# Per-class metrics
precision = precision_score(true_labels, all_predictions, average=None, zero_division=0)
recall = recall_score(true_labels, all_predictions, average=None, zero_division=0)
f1 = f1_score(true_labels, all_predictions, average=None, zero_division=0)

print("\nPer-class Metrics:")
print("-" * 70)
for i in range(5):
    print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

# Weighted/macro averages
precision_macro = precision_score(true_labels, all_predictions, average='macro', zero_division=0)
recall_macro = recall_score(true_labels, all_predictions, average='macro', zero_division=0)
f1_macro = f1_score(true_labels, all_predictions, average='macro', zero_division=0)

precision_weighted = precision_score(true_labels, all_predictions, average='weighted', zero_division=0)
recall_weighted = recall_score(true_labels, all_predictions, average='weighted', zero_division=0)
f1_weighted = f1_score(true_labels, all_predictions, average='weighted', zero_division=0)

print("\nMacro Averages:")
print(f"Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")

print("\nWeighted Averages:")
print(f"Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")

# ============================================================================
# 5. CONFUSION MATRIX
# ============================================================================
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(true_labels, all_predictions)
print(cm)

# ============================================================================
# 6. CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(
    true_labels,
    all_predictions,
    target_names=[f'Class {i}' for i in range(5)],
    zero_division=0
))

# ============================================================================
# 7. PREDICTION CONFIDENCE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("PREDICTION CONFIDENCE ANALYSIS")
print("="*70)

max_probabilities = np.max(all_probabilities, axis=1)
print(f"Mean confidence: {np.mean(max_probabilities):.4f}")
print(f"Std confidence: {np.std(max_probabilities):.4f}")
print(f"Min confidence: {np.min(max_probabilities):.4f}")
print(f"Max confidence: {np.max(max_probabilities):.4f}")

# Count correct predictions with high/low confidence
correct_predictions = np.array(all_predictions) == np.array(true_labels)
high_conf_correct = np.sum(correct_predictions & (max_probabilities > 0.9))
high_conf_incorrect = np.sum(~correct_predictions & (max_probabilities > 0.9))
low_conf_correct = np.sum(correct_predictions & (max_probabilities <= 0.5))
low_conf_incorrect = np.sum(~correct_predictions & (max_probabilities <= 0.5))

print(f"\nCorrect predictions with high confidence (>0.9): {high_conf_correct}")
print(f"Incorrect predictions with high confidence (>0.9): {high_conf_incorrect}")
print(f"Correct predictions with low confidence (<=0.5): {low_conf_correct}")
print(f"Incorrect predictions with low confidence (<=0.5): {low_conf_incorrect}")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\nGenerating visualizations...")

# Plot 1: Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=[f'Class {i}' for i in range(5)],
            yticklabels=[f'Class {i}' for i in range(5)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved confusion_matrix.png")
plt.close()

# Plot 2: Per-class F1 scores
plt.figure(figsize=(10, 6))
bars = plt.bar([f'Class {i}' for i in range(5)], f1, color='steelblue', edgecolor='black')
plt.axhline(y=f1_macro, color='r', linestyle='--', label=f'Macro Avg: {f1_macro:.4f}')
plt.ylabel('F1 Score')
plt.title('Per-Class F1 Scores')
plt.ylim(0, 1)
plt.legend()
for i, (bar, score) in enumerate(zip(bars, f1)):
    plt.text(bar.get_x() + bar.get_width()/2, score + 0.02, f'{score:.3f}', 
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('f1_scores.png', dpi=150, bbox_inches='tight')
print("✓ Saved f1_scores.png")
plt.close()

# Plot 3: Confidence Distribution
plt.figure(figsize=(10, 6))
plt.hist(max_probabilities, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=np.mean(max_probabilities), color='r', linestyle='--', 
            label=f'Mean: {np.mean(max_probabilities):.4f}')
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Distribution of Maximum Prediction Probabilities')
plt.legend()
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved confidence_distribution.png")
plt.close()

# Plot 4: Correct vs Incorrect by Confidence
plt.figure(figsize=(10, 6))
bins = np.linspace(0, 1, 11)
correct_hist, _ = np.histogram(max_probabilities[correct_predictions], bins=bins)
incorrect_hist, _ = np.histogram(max_probabilities[~correct_predictions], bins=bins)

bin_centers = (bins[:-1] + bins[1:]) / 2
width = 0.035
plt.bar(bin_centers - width/2, correct_hist, width, label='Correct', alpha=0.8, color='green')
plt.bar(bin_centers + width/2, incorrect_hist, width, label='Incorrect', alpha=0.8, color='red')
plt.xlabel('Prediction Confidence')
plt.ylabel('Count')
plt.title('Correct vs Incorrect Predictions by Confidence')
plt.legend()
plt.tight_layout()
plt.savefig('correct_incorrect_by_confidence.png', dpi=150, bbox_inches='tight')
print("✓ Saved correct_incorrect_by_confidence.png")
plt.close()

# Plot 5: Precision, Recall, F1 Comparison
plt.figure(figsize=(12, 6))
x = np.arange(5)
width = 0.25

plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
plt.bar(x, recall, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1, width, label='F1 Score', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score by Class')
plt.xticks(x, [f'Class {i}' for i in range(5)])
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved metrics_comparison.png")
plt.close()

# ============================================================================
# 9. SAVE RESULTS SUMMARY
# ============================================================================
print("\nSaving results summary to file...")

results_summary = f"""
{'='*70}
DEBERTA MODEL EVALUATION - TEST DATASET RESULTS
{'='*70}

Test Dataset Size: {len(texts)}
Number of Classes: 5
Device: {device}

{'='*70}
OVERALL METRICS
{'='*70}

Accuracy: {accuracy:.4f}

Macro Averages:
  Precision: {precision_macro:.4f}
  Recall: {recall_macro:.4f}
  F1 Score: {f1_macro:.4f}

Weighted Averages:
  Precision: {precision_weighted:.4f}
  Recall: {recall_weighted:.4f}
  F1 Score: {f1_weighted:.4f}

{'='*70}
PER-CLASS METRICS
{'='*70}

"""

for i in range(5):
    results_summary += f"Class {i}:\n"
    results_summary += f"  Precision: {precision[i]:.4f}\n"
    results_summary += f"  Recall: {recall[i]:.4f}\n"
    results_summary += f"  F1 Score: {f1[i]:.4f}\n"

results_summary += f"\n{'='*70}\nCONFUSION MATRIX\n{'='*70}\n"
results_summary += str(cm) + "\n"

results_summary += f"\n{'='*70}\nPREDICTION CONFIDENCE ANALYSIS\n{'='*70}\n"
results_summary += f"Mean Confidence: {np.mean(max_probabilities):.4f}\n"
results_summary += f"Std Confidence: {np.std(max_probabilities):.4f}\n"
results_summary += f"Min Confidence: {np.min(max_probabilities):.4f}\n"
results_summary += f"Max Confidence: {np.max(max_probabilities):.4f}\n"
results_summary += f"\nCorrect predictions with high confidence (>0.9): {high_conf_correct}\n"
results_summary += f"Incorrect predictions with high confidence (>0.9): {high_conf_incorrect}\n"
results_summary += f"Correct predictions with low confidence (<=0.5): {low_conf_correct}\n"
results_summary += f"Incorrect predictions with low confidence (<=0.5): {low_conf_incorrect}\n"

results_summary += f"\n{'='*70}\nVISUALIZATIONS GENERATED\n{'='*70}\n"
results_summary += "✓ confusion_matrix.png\n"
results_summary += "✓ f1_scores.png\n"
results_summary += "✓ confidence_distribution.png\n"
results_summary += "✓ correct_incorrect_by_confidence.png\n"
results_summary += "✓ metrics_comparison.png\n"

with open('evaluation_results.txt', 'w') as f:
    f.write(results_summary)

print("✓ Saved evaluation_results.txt")

print("\n" + "="*70)
print("EVALUATION COMPLETED SUCCESSFULLY!")
print("="*70)
