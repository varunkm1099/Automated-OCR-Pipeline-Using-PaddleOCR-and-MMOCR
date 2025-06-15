import os
import pandas as pd
from tqdm import tqdm
from google.cloud import vision

# --- CONFIGURATION ---
IMAGE_DIR = r"C:\Users\varun\Desktop\Projects\ANNA\cropped_texts"
OUTPUT_CSV = r"C:\Users\varun\Desktop\Projects\ANNA\results.csv"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\varun\Desktop\Projects\ANNA\speedy-defender-461311-q8-54b8a0b14622.json"  # Update path

def extract_text_google_vision(image_path):
    client = vision.ImageAnnotatorClient()
    try:
        with open(image_path, "rb") as img_file:
            content = img_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description.strip()
        else:
            return ""
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return f"ERROR: {e}"

if __name__ == "__main__":
    # Find all images in the directory
    image_files = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
    ]
    print(f"📦 Found {len(image_files)} images. Processing…")

    results = []
    for image_path in tqdm(image_files):
        text = extract_text_google_vision(image_path)
        results.append({"image_path": image_path, "extracted_text": text})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, quoting=1)
    print(f"✅ Done! Results saved to {OUTPUT_CSV}")
