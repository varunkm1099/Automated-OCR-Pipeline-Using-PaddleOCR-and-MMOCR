import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from difflib import SequenceMatcher
from Levenshtein import ratio as lev_ratio

# Set your file paths here
PRED_CSV = r"C:\Users\varun\Desktop\Projects\ANNA\cropped_texts\paddleocr_results.csv"
GT_CSV   = r"C:\Users\varun\Desktop\Projects\ANNA\GROUND_TRUTH_CSV.csv"
RESULT_CSV = r"C:\Users\varun\Desktop\Projects\ANNA\paddleocr_eval_results.csv"

# --- Step 1: Load CSVs and sanitize column names ---
def clean_cols(df):
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

pred = clean_cols(pd.read_csv(PRED_CSV))
gt   = clean_cols(pd.read_csv(GT_CSV))

# Standardize to: filename, extracted_text, ground_truth
# Prediction file
if 'image_path' in pred.columns:
    pred['filename'] = pred['image_path'].apply(os.path.basename)
if 'pred_text' in pred.columns:
    pred['extracted_text'] = pred['pred_text']
if 'result' in pred.columns:
    pred['extracted_text'] = pred['result']

# Ground truth file
if 'image_path' in gt.columns:
    gt['filename'] = gt['image_path'].apply(os.path.basename)
if 'gt_text' in gt.columns:
    gt['ground_truth'] = gt['gt_text']
if 'extracted_text' in gt.columns and 'ground_truth' not in gt.columns:
    gt['ground_truth'] = gt['extracted_text']

# Only keep needed columns
pred = pred[['filename', 'extracted_text']].dropna()
gt   = gt[['filename', 'ground_truth']].dropna()

# --- Step 2: Merge ---
df = pd.merge(pred, gt, on='filename', how='inner')

print("Merged columns:", df.columns.tolist())
print(f"Merged data shape: {df.shape}")

# --- Step 3: Metrics ---

# Levenshtein similarity
df['levenshtein'] = [lev_ratio(str(p), str(g)) for p, g in zip(df['extracted_text'], df['ground_truth'])]

# Word-level accuracy (simple set intersection over union)
def word_accuracy(a, b):
    wa = set(str(a).split())
    wb = set(str(b).split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

df['word_accuracy'] = [word_accuracy(a, b) for a, b in zip(df['extracted_text'], df['ground_truth'])]

# Exact match rate
df['exact'] = [str(a).strip() == str(b).strip() for a, b in zip(df['extracted_text'], df['ground_truth'])]

# CER/WER can be added if needed

# --- Step 4: Plots ---

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Levenshtein histogram
axs[0].hist(df['levenshtein'], bins=20, color='steelblue', alpha=0.8)
axs[0].set_title('Levenshtein Similarity Histogram')
axs[0].set_xlabel('Similarity')
axs[0].set_ylabel('Count')

# Word accuracy histogram
axs[1].hist(df['word_accuracy'], bins=20, color='orange', alpha=0.8)
axs[1].set_title('Word Accuracy Histogram')
axs[1].set_xlabel('Word Accuracy')
axs[1].set_ylabel('Count')

# Pie chart for exact matches
exact_counts = df['exact'].value_counts()
axs[2].pie(
    [exact_counts.get(True, 0), exact_counts.get(False, 0)],
    labels=['Exact Match', 'Not Exact'],
    autopct='%1.1f%%',
    colors=['#3477eb', '#fa7816']
)
axs[2].set_title('Exact Match Rate')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(PRED_CSV), 'paddleocr_eval_results.png'))
plt.show()

# --- Step 5: Save results ---
df.to_csv(RESULT_CSV, index=False)
print(f"\nResults with metrics saved to: {RESULT_CSV}")
print(f"Plot image saved to: {os.path.dirname(PRED_CSV)}\\paddleocr_eval_results.png")

# --- Extra: Print quick summary ---
print("\nQuick summary:")
print(f"Avg Levenshtein similarity: {df['levenshtein'].mean():.3f}")
print(f"Avg word accuracy: {df['word_accuracy'].mean():.3f}")
print(f"Exact match rate: {df['exact'].mean()*100:.2f}%")
