import cv2
import json
import os
import glob
import numpy as np
import tensorflow as tf

# --- CONFIG ---
REAL_DATA_DIR = "dataset_sample" # Folder with your samples
MODEL_PATH = "error_bar_model_ml.h5"     
OUTPUT_DIR = "visual_audit"
IMG_H, IMG_W = 192, 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_discrepancy():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    img_dir = os.path.join(REAL_DATA_DIR, "images")
    lbl_dir = os.path.join(REAL_DATA_DIR, "labels")
    json_files = sorted(glob.glob(os.path.join(lbl_dir, "*.json")))
    
    print(f"Auditing {len(json_files)} images...")
    
    for j_file in json_files:
        filename = os.path.basename(j_file)
        img_path = os.path.join(img_dir, filename.replace(".json", ".png"))
        
        if not os.path.exists(img_path): continue
        
        # FIX 1: Force 3-channel BGR loading
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        if img is None: continue
        
        debug_img = img.copy()
        
        with open(j_file, 'r') as f: data = json.load(f)
        
        patches = []
        meta = []
        
        for series in data:
            if "Layout_Markers" in series['label']['lineName']: continue
            for pt in series['points']:
                px, py = int(pt['x']), int(pt['y'])
                
                # Robust Padding & Cropping
                pad_h, pad_w = IMG_H, IMG_W
                # Add white border so we can crop even if point is at the edge
                img_padded = cv2.copyMakeBorder(img, pad_h, pad_w, pad_h, pad_w, 
                                              cv2.BORDER_CONSTANT, value=[255,255,255])
                
                # Calculate slice coordinates
                y_start = py + pad_h - (IMG_H // 2)
                y_end   = py + pad_h + (IMG_H // 2)
                x_start = px + pad_w - (IMG_W // 2)
                x_end   = px + pad_w + (IMG_W // 2)
                
                crop = img_padded[y_start:y_end, x_start:x_end]
                
                # FIX 2: Check shape before appending
                if crop.shape != (IMG_H, IMG_W, 3):
                    print(f"⚠️ Warning: Skipping bad point at ({px},{py}) in {filename}. Shape: {crop.shape}")
                    continue
                
                patches.append(crop.astype('float32') / 255.0)
                meta.append(pt)
        
        if len(patches) == 0: continue
        
        # Predict Batch
        try:
            preds = model.predict(np.array(patches), verbose=0)
        except Exception as e:
            print(f"Prediction error on {filename}: {e}")
            continue
        
        # Draw Results
        for i, pt in enumerate(meta):
            x, y = int(pt['x']), int(pt['y'])
            
            # 1. GROUND TRUTH (GREEN)
            gt_top = pt['topBarPixelDistance']
            gt_bot = pt['bottomBarPixelDistance']
            
            # Draw GT only if it exists
            if gt_top > 0:
                cv2.line(debug_img, (x-2, int(y-gt_top)), (x+8, int(y-gt_top)), (0, 255, 0), 2)
                cv2.line(debug_img, (x+3, y), (x+3, int(y-gt_top)), (0, 255, 0), 1) # Thin connector
            if gt_bot > 0:
                cv2.line(debug_img, (x-2, int(y+gt_bot)), (x+8, int(y+gt_bot)), (0, 255, 0), 2)
                cv2.line(debug_img, (x+3, y), (x+3, int(y+gt_bot)), (0, 255, 0), 1)
            
            # 2. MODEL PREDICTION (BLUE)
            pred_top = preds[i][0] * (IMG_H/2)
            pred_bot = preds[i][1] * (IMG_H/2)
            
            # Draw Pred only if significant
            if pred_top > 2:
                cv2.line(debug_img, (x-8, int(y-pred_top)), (x+2, int(y-pred_top)), (255, 0, 0), 2)
                cv2.line(debug_img, (x-3, y), (x-3, int(y-pred_top)), (255, 0, 0), 1)
            if pred_bot > 2:
                cv2.line(debug_img, (x-8, int(y+pred_bot)), (x+2, int(y+pred_bot)), (255, 0, 0), 2)
                cv2.line(debug_img, (x-3, y), (x-3, int(y+pred_bot)), (255, 0, 0), 1)

        # Save result
        out_name = "audit_" + filename.replace(".json", ".png")
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), debug_img)
    
    print(f"✅ Audit Complete. Check the '{OUTPUT_DIR}' folder.")
    print("Legend: GREEN = Dataset Label | BLUE = Model Prediction")

if __name__ == "__main__":
    visualize_discrepancy()