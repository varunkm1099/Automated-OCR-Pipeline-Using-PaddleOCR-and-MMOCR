from mmocr.apis import MMOCRInferencer
import os
import pandas as pd

IMAGES_DIR = r'C:\Users\varun\Desktop\Projects\ANNA\cropped_texts'
RESULT_CSV = os.path.join(IMAGES_DIR, "mmocr_results.csv")

ocr = MMOCRInferencer(det='dbnet', rec='crnn', device='cpu')

results = []
for filename in os.listdir(IMAGES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(IMAGES_DIR, filename)
        result = ocr(img_path)
        pred_text = " ".join(result['predictions'][0]['rec_texts'])
        results.append({
            'filename': filename,
            'extracted_text': pred_text,
        })

df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"Done! Results saved to {RESULT_CSV}")
