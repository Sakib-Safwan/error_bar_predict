import os

# 1. Suppress TensorFlow Warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import json
import numpy as np
import glob
import tensorflow as tf

# --- CONFIGURATION ---
DATASET_DIR = "dataset_v7_production"
MODEL_PATH = "error_bar_model_ml.h5"
RESULTS_DIR = "detection_results_ml"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Must match training config
IMG_H, IMG_W = 192, 64 

def run_prediction_pipeline():
    print(f"Loading model from {MODEL_PATH}...")
    
    # --- FIXED LINE BELOW ---
    # compile=False tells Keras: "Don't try to load the broken loss function. 
    # Just load the weights so I can run predictions."
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model.\nDetail: {e}")
        return
    
    img_dir = os.path.join(DATASET_DIR, "images")
    lbl_dir = os.path.join(DATASET_DIR, "labels")
    json_files = sorted(glob.glob(os.path.join(lbl_dir, "*.json")))
    
    total_err_t = 0
    total_err_b = 0
    count = 0
    
    print(f"Running inference on {len(json_files)} images...")
    
    for j_file in json_files:
        filename = os.path.basename(j_file)
        img_path = os.path.join(img_dir, filename.replace(".json", ".png"))
        
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img, _ = img.shape
        
        with open(j_file, 'r') as f:
            data = json.load(f)
            
        patches = []
        metadata = [] 
        
        # --- PREPARE BATCH ---
        for s_idx, series in enumerate(data):
            line_name = series['label']['lineName']
            if "Layout_Markers" in line_name: continue
            
            for p_idx, pt in enumerate(series['points']):
                px, py = int(pt['x']), int(pt['y'])
                
                # Crop
                x1, x2 = px - IMG_W//2, px + IMG_W//2
                y1, y2 = py - IMG_H//2, py + IMG_H//2
                
                crop = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8) + 255
                src_x1, src_y1 = max(0, x1), max(0, y1)
                src_x2, src_y2 = min(w_img, x2), min(h_img, y2)
                dst_x1, dst_y1 = max(0, src_x1 - x1), max(0, src_y1 - y1)
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                
                if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                    crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
                
                crop = crop.astype('float32') / 255.0
                patches.append(crop)
                metadata.append((s_idx, p_idx))
                
        if not patches: continue
        
        # --- BATCH PREDICT ---
        preds = model.predict(np.array(patches), verbose=0)
        
        # --- RECONSTRUCT RESULTS ---
        result_data = json.loads(json.dumps(data)) 
        
        for i, (s_idx, p_idx) in enumerate(metadata):
            pred_norm = preds[i]
            
            # De-normalize: Pred (0-1) -> Pixels (0 - 96)
            p_top = max(0, pred_norm[0] * (IMG_H / 2))
            p_bot = max(0, pred_norm[1] * (IMG_H / 2))
            
            if p_top < 3: p_top = 0
            if p_bot < 3: p_bot = 0
            
            pt = result_data[s_idx]['points'][p_idx]
            pt['topBarPixelDistance_Pred'] = float(p_top)
            pt['bottomBarPixelDistance_Pred'] = float(p_bot)
            
            total_err_t += abs(p_top - pt['topBarPixelDistance'])
            total_err_b += abs(p_bot - pt['bottomBarPixelDistance'])
            count += 1
            
        with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
            json.dump(result_data, f, indent=2)
            
    if count > 0:
        print("="*40)
        print(f"FINAL ML REPORT")
        print(f"Processed {count} points")
        print(f"MAE Top: {total_err_t/count:.2f} px")
        print(f"MAE Bottom: {total_err_b/count:.2f} px")
        print("="*40)

if __name__ == "__main__":
    run_prediction_pipeline()