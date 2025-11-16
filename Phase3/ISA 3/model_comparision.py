import cv2
import numpy as np
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from collections import defaultdict
import pandas as pd
from ultralytics import YOLO
import torch

class HumanDetectionComparison:
    def __init__(self, dataset_path, yolov5_model_path):
        self.dataset_path = Path(dataset_path)
        self.yolov5_model_path = yolov5_model_path
        # track which API loaded the yolov5 model ('hub' or 'ultralytics')
        self.yolov5_model_type = None
        self.results = defaultdict(list)
        self.detection_results = {}
        
        # Create output directories
        self.output_dir = Path("detection_results")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        
        # Load models
        print("Loading models...")
        self.load_models()
        
    def load_models(self):
        """Load all detection models"""
        # 1. YOLOv5 (custom model)
        try:
            # Try normal load first
            self.yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                               path=self.yolov5_model_path, force_reload=False)
            self.yolov5_model.conf = 0.25
            self.yolov5_model_type = 'hub'
            print("✓ YOLOv5 model loaded (cache)")
        except Exception as e:
            # Some hub cache states cause errors like: "'Detect' object has no attribute 'grid'"
            # Retry with force_reload to clear the hub cache.
            print(f"⚠ YOLOv5 initial load failed: {e}")
            try:
                print("Retrying YOLOv5 load with force_reload=True...")
                self.yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                   path=self.yolov5_model_path, force_reload=True)
                self.yolov5_model.conf = 0.25
                self.yolov5_model_type = 'hub'
                print("✓ YOLOv5 model loaded (force_reload)")
            except Exception as e2:
                print(f"✗ YOLOv5 loading failed after retry: {e2}")
                print("Attempting to load the local weights with the ultralytics.YOLO loader as a fallback...")
                try:
                    # Try to load the local .pt with the ultralytics YOLO class
                    self.yolov5_model = YOLO(self.yolov5_model_path)
                    self.yolov5_model_type = 'ultralytics'
                    print("✓ YOLOv5 weights loaded via ultralytics.YOLO fallback")
                except Exception as e3:
                    print(f"✗ YOLOv5 loading failed after ultralytics fallback: {e3}")
                    print("YOLOv5 will be skipped. If you keep seeing this, try clearing the torch hub cache or ensure the weights file is a valid yolov5/ultralytics .pt file.")
                    self.yolov5_model = None
        
        # 2. YOLOv8 (pretrained)
        try:
            self.yolov8_model = YOLO('yolov8n.pt')
            print("✓ YOLOv8 model loaded")
        except Exception as e:
            print(f"✗ YOLOv8 loading failed: {e}")
            self.yolov8_model = None
        
        # 3. Haar Cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            if self.haar_cascade.empty():
                raise Exception("Failed to load Haar Cascade")
            print("✓ Haar Cascade loaded")
        except Exception as e:
            print(f"✗ Haar Cascade loading failed: {e}")
            self.haar_cascade = None
        
        # 4. HOG Detector
        try:
            self.hog_detector = cv2.HOGDescriptor()
            self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("✓ HOG Detector loaded")
        except Exception as e:
            print(f"✗ HOG Detector loading failed: {e}")
            self.hog_detector = None
        
        # MobileNet SSD removed — not used in this comparison
    
    def detect_yolov5(self, image):
        """YOLOv5 detection"""
        if self.yolov5_model is None:
            return [], 0
        
        start_time = time.time()
        try:
            # If model was loaded with ultralytics.YOLO the call signature is similar to YOLOv8
            if self.yolov5_model_type == 'ultralytics':
                results = self.yolov5_model(image, verbose=False)
            else:
                results = self.yolov5_model(image)
        except Exception as e:
            # Catch runtime inference errors and skip
            print(f"YOLOv5 inference error: {e}")
            return [], 0
        inference_time = time.time() - start_time
        
        detections = []
        # Filter for person class (class 0 in COCO)
    # Different versions of the yolov5 hub return results in slightly different
    # structures. Try a few common access patterns. If the model was loaded via
    # ultralytics.YOLO, its results behave like yolov8 outputs (iterable of result objects).
        parsed = False
        # Pattern 1: results.xyxy[0]
        try:
            arr = results.xyxy[0].cpu().numpy()
            for *box, conf, cls in arr:
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({'bbox': [x1, y1, x2, y2], 'confidence': float(conf)})
            parsed = True
        except Exception:
            pass

        # Pattern 2: results.pandas().xyxy[0]
        if not parsed:
            try:
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    if int(row['class']) == 0:
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        detections.append({'bbox': [x1, y1, x2, y2], 'confidence': float(row['confidence'])})
                parsed = True
            except Exception:
                pass

        # Pattern 3: results may be a list of detections per image
        if not parsed:
            try:
                # Try iterate over results if it's list-like
                for r in results:
                    # each r may have .boxes or .xyxy
                    if hasattr(r, 'boxes'):
                        for box in r.boxes:
                            cls_val = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else None
                            if cls_val == 0:
                                xy = box.xyxy[0] if hasattr(box, 'xyxy') else None
                                if xy is not None:
                                    x1, y1, x2, y2 = map(int, xy)
                                    conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 1.0
                                    detections.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})
                    elif hasattr(r, 'xyxy'):
                        arr = r.xyxy.cpu().numpy()
                        for *box, conf, cls in arr:
                            if int(cls) == 0:
                                x1, y1, x2, y2 = map(int, box)
                                detections.append({'bbox': [x1, y1, x2, y2], 'confidence': float(conf)})
                parsed = True
            except Exception:
                pass
        # If model was loaded by ultralytics and parsed is still False, try the yolov8-style path
        if not parsed and self.yolov5_model_type == 'ultralytics':
            try:
                for result in results:
                    boxes = getattr(result, 'boxes', None)
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_val = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else None
                        if cls_val == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 1.0
                            detections.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})
                parsed = True
            except Exception:
                pass
        
        return detections, inference_time

    # Ad-Hoc detector removed

    
    def detect_yolov8(self, image):
        """YOLOv8 detection"""
        if self.yolov8_model is None:
            return [], 0
        
        start_time = time.time()
        results = self.yolov8_model(image, verbose=False)
        inference_time = time.time() - start_time
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(box.conf[0])
                    })
        
        return detections, inference_time
    
    def detect_haar_cascade(self, image):
        """Haar Cascade detection"""
        if self.haar_cascade is None:
            return [], 0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        start_time = time.time()
        bodies = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        inference_time = time.time() - start_time
        
        detections = []
        for (x, y, w, h) in bodies:
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 1.0  # Haar Cascade doesn't provide confidence
            })
        
        return detections, inference_time
    
    def detect_hog(self, image):
        """HOG detection"""
        if self.hog_detector is None:
            return [], 0
        
        start_time = time.time()
        boxes, weights = self.hog_detector.detectMultiScale(
            image, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        inference_time = time.time() - start_time
        
        detections = []
        for i, (x, y, w, h) in enumerate(boxes):
            # Handle both scalar and array weights
            weight = weights[i]
            if isinstance(weight, np.ndarray):
                confidence = float(weight[0]) if len(weight) > 0 else 1.0
            else:
                confidence = float(weight)
            
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': confidence
            })
        
        return detections, inference_time

    
    def process_dataset(self):
        """Process all images in the dataset"""
        image_files = list(self.dataset_path.glob("*.jpg")) + \
                     list(self.dataset_path.glob("*.png")) + \
                     list(self.dataset_path.glob("*.jpeg"))
        
        if not image_files:
            print(f"No images found in {self.dataset_path}")
            return
        
        # Limit to random 50 images
        if len(image_files) > 50:
            import random
            random.seed(42)  # For reproducibility
            image_files = random.sample(image_files, 50)
            print(f"\nRandomly selected 50 images out of {len(list(self.dataset_path.glob('*.jpg')) + list(self.dataset_path.glob('*.png')) + list(self.dataset_path.glob('*.jpeg')))} total images")
        
        print(f"\nProcessing {len(image_files)} images...")
        
        models = {
            'YOLOv5': self.detect_yolov5,
            'YOLOv8': self.detect_yolov8,
            'Haar Cascade': self.detect_haar_cascade,
            'HOG': self.detect_hog
        }
        
        for idx, image_path in enumerate(image_files):
            print(f"Processing image {idx + 1}/{len(image_files)}: {image_path.name}")
            image = cv2.imread(str(image_path))
            
            if image is None:
                continue
            
            self.detection_results[image_path.name] = {}
            
            for model_name, detect_func in models.items():
                detections, inference_time = detect_func(image)
                
                self.results[model_name].append({
                    'image': image_path.name,
                    'num_detections': len(detections),
                    'inference_time': inference_time,
                    'detections': detections
                })
                
                self.detection_results[image_path.name][model_name] = {
                    'detections': detections,
                    'inference_time': inference_time
                }
            
            # Save visualization for first 10 images
            if idx < 10:
                self.visualize_detections(image, image_path.name)
    
    def visualize_detections(self, image, image_name):
        """Create visualization comparing all models"""
        models = ['YOLOv5', 'YOLOv8', 'Haar Cascade', 'HOG', 'Ad-Hoc']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        for idx, model_name in enumerate(models, 1):
            img_copy = image.copy()

            if image_name in self.detection_results and model_name in self.detection_results[image_name]:
                detections = self.detection_results[image_name][model_name]['detections']
                inference_time = self.detection_results[image_name][model_name]['inference_time']

                # Draw bounding boxes
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_copy, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                axes[idx].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(f'{model_name}\n{len(detections)} detections | {inference_time*1000:.1f}ms')
            else:
                axes[idx].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(f'{model_name}\nNot Available')

            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / f"comparison_{image_name}")
        plt.close()
    
    def generate_performance_report(self):
        """Generate comprehensive performance analysis"""
        print("\nGenerating performance report...")
        
        # Calculate statistics
        stats_data = []
        for model_name, results in self.results.items():
            if not results:
                continue
                
            inference_times = [r['inference_time'] for r in results]
            num_detections = [r['num_detections'] for r in results]
            
            stats_data.append({
                'Model': model_name,
                'Avg Inference Time (ms)': np.mean(inference_times) * 1000,
                'Std Inference Time (ms)': np.std(inference_times) * 1000,
                'Min Inference Time (ms)': np.min(inference_times) * 1000,
                'Max Inference Time (ms)': np.max(inference_times) * 1000,
                'Avg Detections': np.mean(num_detections),
                'Total Detections': np.sum(num_detections),
                'Images Processed': len(results)
            })
        
        df_stats = pd.DataFrame(stats_data)
        
        # Save to CSV
        df_stats.to_csv(self.output_dir / "performance_statistics.csv", index=False)
        
        # Print summary
        print("\n" + "="*164)
        print("PERFORMANCE SUMMARY")
        print("="*164)
        print(df_stats.to_string(index=False))
        print("="*164)
        
        # Generate graphs
        self.plot_inference_time_comparison(df_stats)
        self.plot_detection_count_comparison(df_stats)
        self.plot_inference_time_distribution()
        self.plot_detection_heatmap()
        
        print(f"\n✓ Results saved to: {self.output_dir}")
        print(f"  - Statistics: performance_statistics.csv")
        print(f"  - Visualizations: visualizations/")
        print(f"  - Graphs: graphs/")
    
    def plot_inference_time_comparison(self, df_stats):
        """Plot inference time comparison"""
        plt.figure(figsize=(12, 6))

        # Left: Average inference time on a logarithmic (exponential) y-scale
        ax1 = plt.subplot(1, 2, 1)
        times = df_stats['Avg Inference Time (ms)'].astype(float).copy()
        # Replace non-positive values with a small epsilon to allow log scaling
        epsilon = 1e-3
        times = times.where(times > 0, epsilon)

        ax1.bar(df_stats['Model'], times, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Inference Time (ms) (log scale)')
        ax1.set_title('Average Inference Time by Model (log scale)')
        ax1.set_xticklabels(df_stats['Model'], rotation=45, ha='right')
        ax1.set_yscale('log')
        # Use base-10 log ticks for readability
        ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax1.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
        ax1.grid(axis='y', alpha=0.3, which='both')

        # Right: Average detections (linear)
        ax2 = plt.subplot(1, 2, 2)
        ax2.bar(df_stats['Model'], df_stats['Avg Detections'], color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Average Number of Detections')
        ax2.set_title('Average Detections per Image by Model')
        ax2.set_xticklabels(df_stats['Model'], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / "inference_time_comparison.png", dpi=300)
        plt.close()
    
    def plot_detection_count_comparison(self, df_stats):
        """Plot detection count comparison"""
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(df_stats))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, df_stats['Avg Detections'], width, 
                       label='Avg Detections', color='lightgreen')
        bars2 = ax.bar(x + width/2, df_stats['Total Detections']/10, width, 
                       label='Total Detections (÷10)', color='orange')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Count')
        ax.set_title('Detection Count Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df_stats['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / "detection_count_comparison.png", dpi=300)
        plt.close()
    
    def plot_inference_time_distribution(self):
        """Plot inference time distribution for all models"""
        plt.figure(figsize=(14, 8))
        
        for model_name, results in self.results.items():
            if not results:
                continue
            inference_times = [r['inference_time'] * 1000 for r in results]
            plt.hist(inference_times, alpha=0.5, label=model_name, bins=20)
        
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution Across Models')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / "inference_time_distribution.png", dpi=300)
        plt.close()
    
    def plot_detection_heatmap(self):
        """Plot heatmap of detections per image"""
        models = list(self.results.keys())
        images = list(self.detection_results.keys())[:20]  # First 20 images
        
        if not models or not images:
            return
        
        heatmap_data = []
        for img in images:
            row = []
            for model in models:
                if model in self.detection_results[img]:
                    row.append(self.detection_results[img][model]['detections'].__len__())
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, xticklabels=models, 
                   yticklabels=[img[:20] for img in images],
                   annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Detections'})
        plt.xlabel('Model')
        plt.ylabel('Image')
        plt.title('Detection Heatmap (Detections per Image per Model)')
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / "detection_heatmap.png", dpi=300)
        plt.close()

def main():
    # Configuration
    DATASET_PATH = "human detection dataset/images"
    YOLOV5_MODEL_PATH = "yolov5nu.pt"
    
    # Verify paths
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' not found!")
        return
    
    if not os.path.exists(YOLOV5_MODEL_PATH):
        print(f"Error: YOLOv5 model '{YOLOV5_MODEL_PATH}' not found!")
        return
    
    # Run comparison
    comparator = HumanDetectionComparison(DATASET_PATH, YOLOV5_MODEL_PATH)
    comparator.process_dataset()
    comparator.generate_performance_report()
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
