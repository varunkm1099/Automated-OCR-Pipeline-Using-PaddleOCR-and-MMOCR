from paddleocr import PaddleOCR
import os
import pandas as pd

IMAGES_DIR = r'C:\Users\varun\Desktop\Projects\ANNA\cropped_texts'
GROUND_TRUTH_CSV = r"C:\Users\varun\Desktop\Projects\ANNA\GROUND_TRUTH_CSV.csv"
RESULT_CSV = os.path.join(IMAGES_DIR, "paddleocr_results.csv")

# Enable GPU if available
ocr = PaddleOCR(use_gpu=True, lang='en', use_textline_orientation=True)

# Load ground truth if available
if os.path.exists(GROUND_TRUTH_CSV):
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    if 'filename' in gt_df.columns:
        gt_dict = dict(zip(gt_df['filename'], gt_df['gt_text']))
    elif 'image_path' in gt_df.columns and 'extracted_text' in gt_df.columns:
        gt_dict = dict(zip([os.path.basename(p) for p in gt_df['image_path']], gt_df['extracted_text']))
    else:
        gt_dict = {}
else:
    gt_dict = {}

results = []

for filename in os.listdir(IMAGES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(IMAGES_DIR, filename)
        result = ocr.ocr(img_path)
        pred_text = " ".join([line[1][0] for line in result[0]]) if result and result[0] else ""
        gt_text = gt_dict.get(filename, '')
        results.append({
            'filename': filename,
            'extracted_text': pred_text,
            'ground_truth': gt_text
        })

df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"Done! Results saved to {RESULT_CSV}")
