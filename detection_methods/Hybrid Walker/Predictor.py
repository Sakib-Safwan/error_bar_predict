import cv2
import numpy as np
import json
import os
import glob

# --- CONFIGURATION ---
DATASET_DIR = "dataset_v7_production"
RESULTS_DIR = "detection_results_v6_hybrid"
os.makedirs(RESULTS_DIR, exist_ok=True)

class HybridErrorDetector:
    def __init__(self):
        pass

    def get_smart_mask(self, img, points=None):
        """
        ROBUST GRAYSCALE MASKING.
        Ignores specific colors and looks for 'Dark Ink' on 'Light Background'.
        This fixes the issue where Blue markers hid Black error bars.
        
        Args:
            img: The BGR image.
            points: (Optional) Kept for compatibility with audit scripts, but ignored.
        """
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Adaptive Threshold (Local Binarization)
        # Block Size 25, C=10 are tuned for scientific plots to ignore faint grid lines
        mask = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            25, 
            10
        )
        
        # 3. Morphological Cleanup (Remove tiny noise specks)
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def auto_align_x(self, mask, start_x, start_y):
        """
        Scans left/right to find the vertical stem of the bar.
        Solves the "Offset" problem in grouped bar charts.
        """
        h, w = mask.shape
        start_x, start_y = int(start_x), int(start_y)
        
        # Scan window: +/- 10 pixels wide
        x1, x2 = max(0, start_x - 10), min(w, start_x + 11)
        y1, y2 = max(0, start_y - 20), min(h, start_y + 20)
        
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0: return start_x
        
        # Sum vertically to find the column with the most ink
        col_sums = np.sum(roi, axis=0) 
        best_col_local = np.argmax(col_sums)
        
        # Only shift if there is significant ink found
        if col_sums[best_col_local] > 0:
            return x1 + best_col_local
        
        return start_x

    def detect_cap(self, mask, cx, cy):
        """
        Checks for a horizontal T-shape (Cap).
        """
        h, w = mask.shape
        if cy < 0 or cy >= h: return False
        
        # Look 6px left and 6px right (Total 13px width)
        x1, x2 = max(0, cx - 6), min(w, cx + 7)
        row_slice = mask[cy, x1:x2]
        
        # If we see > 5 pixels of ink horizontally, it's a cap.
        # (Vertical stems are usually only 1-3px wide, so 5+ means horizontal line)
        return np.count_nonzero(row_slice) > 5

    def walk_and_scan(self, mask, start_x, start_y, direction):
        """
        Robust Walker:
        1. Starts at marker center.
        2. Walks Up/Down.
        3. Jumps gaps (e.g., gap between hollow marker and line).
        4. Stops at Cap or End of Line.
        """
        h, w = mask.shape
        curr_y = int(start_y)
        cx = int(start_x)
        
        step = -1 if direction == "up" else 1
        
        last_ink_y = start_y
        gap_count = 0
        max_gap_jump = 12  # Large jump to handle hollow markers/dashed lines
        
        # Safety: Don't detect caps immediately inside the marker
        min_dist_for_cap = 8 
        
        while 0 <= curr_y < h:
            # Check for ink at current (x, y) + neighbors (for anti-aliasing)
            # We look at a 3px wide strip (cx-1, cx, cx+1)
            x_min, x_max = max(0, cx-1), min(w, cx+2)
            is_ink = np.any(mask[curr_y, x_min:x_max] > 0)
            
            if is_ink:
                gap_count = 0 # Reset gap counter
                last_ink_y = curr_y # Update "furthest ink seen"
                
                # Check for Cap (Stop Condition)
                dist_moved = abs(curr_y - start_y)
                if dist_moved > min_dist_for_cap:
                    if self.detect_cap(mask, cx, curr_y):
                        # Found a cap! Return this distance.
                        return float(dist_moved)
            else:
                gap_count += 1
                if gap_count > max_gap_jump:
                    # Gap is too huge (we fell off the line)
                    break 
            
            curr_y += step
            
        return float(abs(last_ink_y - start_y))

    def process_image(self, img_path, json_data):
        """
        Main pipeline processing for a single image.
        """
        img = cv2.imread(img_path)
        if img is None: return []
        
        # 1. Get the Robust Grayscale Mask (One time per image)
        mask = self.get_smart_mask(img)
        
        output_series = []
        
        for series in json_data:
            line_name = series['label']['lineName']
            if "Layout_Markers" in line_name: continue
            
            detected_points = []
            for pt in series['points']:
                px, py = pt['x'], pt['y']
                
                # 2. Align X to the ink column
                aligned_x = self.auto_align_x(mask, px, py)
                
                # 3. Walk
                pred_top = self.walk_and_scan(mask, aligned_x, py, "up")
                pred_bot = self.walk_and_scan(mask, aligned_x, py, "down")
                
                detected_points.append({
                    "x": px, "y": py,
                    "topBarPixelDistance_Pred": pred_top,
                    "bottomBarPixelDistance_Pred": pred_bot,
                    "topBarPixelDistance_GT": pt.get('topBarPixelDistance', 0),
                    "bottomBarPixelDistance_GT": pt.get('bottomBarPixelDistance', 0)
                })
                
            output_series.append({
                "label": {"lineName": line_name},
                "points": detected_points
            })
            
        return output_series

# --- EXECUTION BLOCK (Run this file to generate JSON results) ---
def run_pipeline():
    detector = HybridErrorDetector()
    
    # Update these paths if your folders are different
    img_dir = os.path.join(DATASET_DIR, "images")
    lbl_dir = os.path.join(DATASET_DIR, "labels")
    
    if not os.path.exists(lbl_dir):
        print(f"❌ Error: Could not find label directory: {lbl_dir}")
        return

    json_files = sorted(glob.glob(os.path.join(lbl_dir, "*.json")))
    
    total_err_t, total_err_b, count = 0, 0, 0
    
    print(f"Running Hybrid Walker (v6 Robust) on {len(json_files)} images...")
    
    for j_file in json_files:
        filename = os.path.basename(j_file)
        img_path = os.path.join(img_dir, filename.replace(".json", ".png"))
        
        if not os.path.exists(img_path): continue
        
        with open(j_file, 'r') as f: data = json.load(f)
        
        # Run Detection
        results = detector.process_image(img_path, data)
        
        # Save Result
        with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
            json.dump(results, f, indent=2)
            
        # Calculate Error for Report
        for s in results:
            for p in s['points']:
                total_err_t += abs(p['topBarPixelDistance_Pred'] - p['topBarPixelDistance_GT'])
                total_err_b += abs(p['bottomBarPixelDistance_Pred'] - p['bottomBarPixelDistance_GT'])
                count += 1
                
    if count > 0:
        print("="*40)
        print(f"FINAL REPORT (Hybrid Walker v6)")
        print(f"Processed {count} points")
        print(f"MAE Top:    {total_err_t/count:.2f} px")
        print(f"MAE Bottom: {total_err_b/count:.2f} px")
        print("="*40)
        print(f"Detailed JSON results saved in: {RESULTS_DIR}")
    else:
        print("No points processed. Check dataset paths.")

if __name__ == "__main__":
    run_pipeline()