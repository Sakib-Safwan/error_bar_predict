import matplotlib.pyplot as plt
import numpy as np
import json
import os
import random
import cv2
import string
import matplotlib.lines as mlines

# --- CONFIGURATION ---
NUM_IMAGES = 3000  # Change to 3000 for full run
OUTPUT_DIR = "dataset_v8_backgrounds"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_DIR, "labels")
IMG_SIZE = (12, 10) 

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def get_pixel_coordinates(transform, x, y):
    return transform.transform((x, y))

def apply_background_style(fig, axes):
    """
    Applies background styles: 
    - 5% Handwritten/Notebook (Ruled Paper)
    - 10% Colored Backgrounds (Non-White)
    - 85% Standard White
    """
    roll = random.random()
    
    # === 1. HANDWRITTEN / NOTEBOOK STYLE (5%) ===
    if roll < 0.05:
        # Notebook Yellow-ish tint
        paper_color = "#fffdf0" # Very pale yellow
        line_color = "#a4c2f4"  # Pale blue lines
        
        # Set overall background
        fig.patch.set_facecolor(paper_color)
        for ax in axes:
            ax.set_facecolor(paper_color)
            
            # Draw "Notebook Lines" (Horizontal only)
            # We use the axis coordinates (0 to 1)
            # Draw lines every 5-10% of height
            y_pos = np.arange(0, 1, 0.05)
            for y in y_pos:
                line = mlines.Line2D([0, 1], [y, y], color=line_color, 
                                     linewidth=1, alpha=0.5, transform=ax.transAxes, zorder=0)
                ax.add_line(line)
                
            # Remove standard grid if we have notebook lines (too messy otherwise)
            ax.grid(False)
            
            # Make spines look like pen? (Optional, kept simple for robustness)
            for spine in ax.spines.values():
                spine.set_color('#333333') # Dark grey ink look

    # === 2. NON-WHITE COLORED BACKGROUNDS (10%) ===
    elif roll < 0.15:
        # Professional tints (Beige, Grey, Blue, Pink)
        tints = ["#f5f5f5", "#fdf5e6", "#f0f8ff", "#f5fffa", "#fff0f5"] 
        color = random.choice(tints)
        
        fig.patch.set_facecolor(color)
        for ax in axes:
            ax.set_facecolor(color)
            # Often colored plots have white grids
            if random.random() > 0.5:
                ax.grid(True, color='white', linestyle='-', linewidth=1.5, alpha=0.7)

    # === 3. STANDARD WHITE (85%) ===
    else:
        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')

def apply_image_degradation(img_path):
    """Applies Blur, Noise, or Low-Res (25% chance)."""
    if random.random() > 0.75: 
        img = cv2.imread(img_path)
        if img is None: return

        mode = random.choices(["blur", "noise", "low_res"], weights=[50, 25, 25])[0]
        
        if mode == "blur":
            ksize = random.choice([3, 5, 7, 9])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif mode == "low_res":
            h, w = img.shape[:2]
            scale = random.uniform(0.3, 0.5)
            small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif mode == "noise":
            noise = np.zeros(img.shape, np.uint8)
            cv2.randn(noise, (0,0,0), (20,20,20))
            img = cv2.add(img, noise)

        cv2.imwrite(img_path, img)

def generate_trend_data(num_points, trend_type):
    x = np.linspace(0, 100, num_points)
    
    if trend_type == "sawtooth":
        peaks = random.randint(2, 4)
        y = np.zeros_like(x)
        period_len = len(x) // peaks
        for i in range(len(x)):
            cycle_pos = i % period_len
            if cycle_pos < (period_len * 0.2): 
                y[i] = (cycle_pos / (period_len * 0.2)) * 100
            else: 
                decay_steps = cycle_pos - (period_len * 0.2)
                y[i] = 100 * np.exp(-0.05 * decay_steps)
            y[i] += (i // period_len) * 20 
        y += np.random.normal(0, 5, num_points)

    elif trend_type == "surge":
        y = 500 * (x/20) * np.exp(1 - (x/20))
        y += np.random.normal(0, 10, num_points)

    else: 
        start = random.uniform(50, 1000)
        y = start * np.exp(-0.03 * x)
        y += np.random.normal(0, start*0.05, num_points)
    
    return x, np.maximum(y, 0.1)

# --- AXIS POPULATOR ---
def populate_axis(ax, plot_type, sample_id, sub_index=0):
    
    # 1. BASIC SETUP (Log & Grid)
    is_log = False
    if plot_type == "standard" and random.random() < 0.3:
        is_log = True
        ax.set_yscale('log')

    # Grid logic is now partly handled by 'apply_background_style', 
    # but we set a default here which might be overwritten.
    if random.random() > 0.4:
        ax.grid(True, linestyle='--', alpha=0.5)

    titles = ["Serum Conc.", "PK Profile", "Response Rate", "HbA1c", "C-peptide", "Clearance"]
    title_prefix = f"{string.ascii_uppercase[sub_index]}  " if "subplot" in sample_id else ""
    ax.set_title(f"{title_prefix}{random.choice(titles)}")
    ax.set_xlabel("Time (Days)" if "bar" not in plot_type else "Group")
    ax.set_ylabel("Concentration (ng/mL)")

    # 2. PLOT CONTENT
    
    # === SPAGHETTI (2%) ===
    if plot_type == "spaghetti":
        x_spag = np.linspace(0, 50, 10)
        mean_y = 100 * np.exp(-0.05 * x_spag)
        for _ in range(15):
            noise = np.random.normal(0, 20, len(x_spag))
            ax.plot(x_spag, np.maximum(mean_y + noise, 0), color='gray', lw=0.5, alpha=0.4)
        ax.plot(x_spag, mean_y, color='black', lw=2.5, label="Mean")

    # === BOX PLOT (5%) ===
    elif plot_type == "box":
        num_groups = random.randint(3, 5)
        data = [np.random.normal(50, 10, 20) for _ in range(num_groups)]
        labels = [f"G{i}" for i in range(num_groups)]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.get_cmap('Pastel1')
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=colors(i % 8))

    # === SIMPLE BAR (10%) ===
    elif plot_type == "bar_simple":
        num_groups = random.randint(4, 8)
        x_bar = np.arange(num_groups)
        y_bar = np.random.uniform(20, 100, num_groups)
        y_err = y_bar * 0.1
        hatch = random.choice(['', '//', '..', 'xx'])
        ax.bar(x_bar, y_bar, yerr=y_err, capsize=4, alpha=0.7, 
               hatch=hatch, color='skyblue', edgecolor='black', label="Group A")
        ax.set_xticks(x_bar)
        ax.set_xticklabels([f"T{i}" for i in range(num_groups)])

    # === GROUPED BAR (5%) ===
    elif plot_type == "bar_grouped":
        num_groups = random.randint(3, 5)
        num_series = random.randint(2, 3)
        x_indices = np.arange(num_groups)
        width = 0.8 / num_series
        
        for i in range(num_series):
            y_bar = np.random.uniform(20, 100, num_groups)
            y_err = y_bar * 0.1
            offset = (i - num_series/2) * width + (width/2)
            color = plt.cm.get_cmap('tab10')(i)
            ax.bar(x_indices + offset, y_bar, width, yerr=y_err, capsize=3,
                   color=color, label=f"Trt {i+1}")
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"M{i*3}" for i in range(num_groups)])
        ax.legend(loc='best')

    # === STANDARD / DUAL (~78%) ===
    else: 
        is_dual = random.random() < 0.15 
        trend = "sawtooth" if random.random() < 0.20 else "decay"
        
        num_lines = random.randint(1, 4)
        markers = ['o', 's', '^', 'v', 'D']
        ax2 = ax.twinx() if is_dual else None
        if is_dual: ax2.set_ylabel("Secondary Metric")

        for i in range(num_lines):
            target_ax = ax2 if (is_dual and i >= num_lines//2) else ax
            x, y = generate_trend_data(10, trend)
            y_err = y * 0.15
            color = plt.cm.get_cmap('tab10')(i)
            marker = random.choice(markers)
            mfc = 'white' if random.random() > 0.5 else color # Hollow markers
            
            target_ax.errorbar(x, y, yerr=y_err, fmt=marker, linestyle='-', 
                             color=color, markerfacecolor=mfc, capsize=3, label=f"Trt {i+1}")

        if random.random() < 0.05:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        else:
            ax.legend(loc='best')

    # 3. ANNOTATIONS
    if random.random() < 0.03: 
        lim_y = 5 if not is_log else 1
        ax.axhline(lim_y, color='black', linestyle='--', lw=1)
        ax.text(0, lim_y*1.1, "LLOQ", fontsize=8)

    if random.random() < 0.08 and "bar" in plot_type: 
        x_loc = 1
        y_h = 100
        ax.plot([x_loc-0.2, x_loc-0.2, x_loc+0.2, x_loc+0.2], [y_h, y_h+5, y_h+5, y_h], lw=1, c='k')
        ax.text(x_loc, y_h+5, "**", ha='center', va='bottom')

# --- MAIN GENERATOR ---
def generate_single_sample(sample_id):
    if random.random() > 0.5:
        plt.rcParams["font.family"] = "serif"
    else:
        plt.rcParams["font.family"] = "sans-serif"

    # --- LAYOUT SELECTION ---
    layout_roll = random.random()
    layout_type = "single"
    axes_list = []
    fig = None
    
    if layout_roll < 0.01: # 4 Panels
        layout_type = "subplot_4"
        fig, axes = plt.subplots(2, 2, figsize=IMG_SIZE, dpi=100)
        axes_list = axes.flatten().tolist()
    elif layout_roll < 0.03: # 2 Panels
        layout_type = "subplot_2"
        if random.random() > 0.5:
            fig, axes = plt.subplots(1, 2, figsize=IMG_SIZE, dpi=100)
        else:
            fig, axes = plt.subplots(2, 1, figsize=IMG_SIZE, dpi=100)
        axes_list = axes.flatten().tolist()
    elif layout_roll < 0.05: # Inset
        layout_type = "inset"
        fig, ax_main = plt.subplots(figsize=IMG_SIZE, dpi=100)
        ax_ins = ax_main.inset_axes([0.60, 0.60, 0.35, 0.35])
        axes_list = [ax_main, ax_ins]
    else: # Single
        fig, ax = plt.subplots(figsize=IMG_SIZE, dpi=100)
        axes_list = [ax]
    
    # --- APPLY BACKGROUNDS (New Feature) ---
    apply_background_style(fig, axes_list)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)
    all_series_json = []

    # --- POPULATE AXES ---
    for idx, ax in enumerate(axes_list):
        ptype = random.choices(
            ["spaghetti", "box", "bar_simple", "bar_grouped", "standard"], 
            weights=[2, 5, 10, 5, 78]
        )[0]
        
        if layout_type == "inset" and idx == 1: ptype = "standard"
        if layout_type == "subplot_4": ptype = random.choice(["standard", "bar_simple"])

        populate_axis(ax, ptype, f"{layout_type}_{sample_id}", idx)

    # --- DATA EXTRACTION ---
    fig.canvas.draw()
    
    for idx, current_ax in enumerate(fig.axes):
        transform = current_ax.transData
        lines = [l for l in current_ax.lines if "_line" not in l.get_label()]
        
        for line in lines:
            if line.get_color() == 'gray' or "Line2D" in str(type(line)) and line.get_alpha() == 0.5: 
                continue # Skip background/notebook lines
            
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            label = line.get_label()
            
            if len(x_data) < 2: continue
            
            points_data = []
            for j in range(len(x_data)):
                cx, cy = x_data[j], y_data[j]
                center_pix = get_pixel_coordinates(transform, cx, cy)
                
                # Idealized Error for JSON
                err_val = cy * 0.15
                top_math = cy + err_val
                bot_math = cy - err_val
                if current_ax.get_yscale() == 'log' and bot_math <= 0: bot_math = cy*0.01
                
                top_pix = get_pixel_coordinates(transform, cx, top_math)
                bot_pix = get_pixel_coordinates(transform, cx, bot_math)
                
                points_data.append({
                    "x": float(center_pix[0]),
                    "y": float(fig.get_window_extent().height - center_pix[1]),
                    "topBarPixelDistance": float(abs(top_pix[1] - center_pix[1])),
                    "bottomBarPixelDistance": float(abs(bot_pix[1] - center_pix[1])),
                    "label": label if label != "" else "Data"
                })

            all_series_json.append({
                "label": {"lineName": f"Ax{idx}_{label}"},
                "points": points_data
            })
        
        # Layout Markers
        xlim = current_ax.get_xlim()
        ylim = current_ax.get_ylim()
        corners = [("xmin", xlim[0], ylim[0]), ("xmax", xlim[1], ylim[0]), 
                   ("ymin", xlim[0], ylim[0]), ("ymax", xlim[0], ylim[1])]
        
        layout_points = []
        for lbl, mx, my in corners:
            pix = get_pixel_coordinates(transform, mx, my)
            layout_points.append({
                "x": float(pix[0]),
                "y": float(fig.get_window_extent().height - pix[1]), 
                "label": f"Ax{idx}_{lbl}", 
                "topBarPixelDistance": 0, "bottomBarPixelDistance": 0
            })
        all_series_json.append({"label": {"lineName": "Layout_Markers"}, "points": layout_points})

    # Save
    img_name = f"{sample_id}.png"
    json_name = f"{sample_id}.json"
    full_img_path = os.path.join(IMG_DIR, img_name)
    
    plt.savefig(full_img_path)
    with open(os.path.join(LBL_DIR, json_name), 'w') as f:
        json.dump(all_series_json, f, indent=2)

    plt.close(fig)
    apply_image_degradation(full_img_path)

# --- EXECUTION ---
print(f"Generating {NUM_IMAGES} samples with background variations...")
for i in range(NUM_IMAGES):
    if i % 10 == 0: print(f".. {i}")
    generate_single_sample(f"{i:04d}")
print("Done.")