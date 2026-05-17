# Stage 2A: Task-Aware Object Detection Pipeline

###  File Overview

* **`task_guided_selector.py`**: Handles the core logic for selecting the most contextually appropriate object. 
* **`[Insert File 2 Name Here]`**: Glues backend.
* **`[Insert File 3 Name Here]`**: Sets up a basic UI.

###  Running the pipeline:

1) Run the .ipynb file
2) 
** Note on Execution Time:** To comply with Stage 2A guidelines, this notebook runs entirely on **CPU inference**. If this is your first time running the cell in a new Colab session, it will automatically download the YOLOv8-nano and MobileCLIP-S1 weights. Depending on your internet connection, **this initial setup may take a few minutes**. Subsequent runs in the same session will execute much faster.

---

### 📊 For Judges: How to View Detailed Object Scores

By default, the notebook visually outputs the single best bounding box for the given task. However, if you would like to inspect the internal scoring engine and verify exactly how the model ranked every object detected by YOLO, you can easily print the mathematical breakdown.

**Step-by-step guide:**
1. Run the main inference cells normally so the image is processed and the variables are stored in the runtime memory.
2. Create a **new code cell** immediately below the main pipeline cell.
3. Copy and paste the following debugging snippet into the new cell:

```python
# Print the scoring breakdown for all detected objects
print(f"Task Evaluated: '{current_task}'")
print("-" * 65)
print(f"{'Detected Object':<15} | {'Raw CLIP':<10} | {'CAM Score':<10} | {'YOLO Conf':<10} | {'FINAL SCORE'}")
print("-" * 65)

for i, obj in enumerate(object_crops):
    marker = "  <-- WINNER" if i == best_idx else ""
    print(f"{obj['label']:<15} | {raw_clip_scores[i]:.4f}     | {context_scores[i]:.4f}     | {obj.get('confidence', 0.0):.4f}     | {final_scores[i]:.4f}{marker}")# dv-con_proj
