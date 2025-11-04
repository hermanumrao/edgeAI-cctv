import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from PIL import Image
import torch
from collections import defaultdict
from ultralytics import YOLO

# Create directories if they don't exist
os.makedirs('results', exist_ok=True)



class HaarCascadeDetector:
    def __init__(self):
        # Load the pre-trained Haar Cascade classifier for human detection
        self.human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.upper_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        self.lower_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
    
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect full body
        bodies = self.human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Detect upper body if full body not found
        if len(bodies) == 0:
            upper_bodies = self.upper_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            bodies = upper_bodies
        
        # Detect lower body if others not found
        if len(bodies) == 0:
            lower_bodies = self.lower_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            bodies = lower_bodies
            
        return bodies
    
    def draw_detections(self, image, detections):
        img_copy = image.copy()
        for (x, y, w, h) in detections:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img_copy



class HOGDetector:
    def __init__(self):
        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect(self, image):
        # Detect people in the image
        boxes, weights = self.hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.05)
        return boxes, weights
    
    def draw_detections(self, image, detections):
        img_copy = image.copy()
        boxes, weights = detections
        for i, (x, y, w, h) in enumerate(boxes):
            if weights[i] > 0.5:  # Filter by confidence
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img_copy



class YOLODetector:
    def __init__(self):
        try:
            self.model = YOLO('yolov5nu.pt')  # use your local model file
            self.model.to('cpu')
        except Exception as e:
            print(f"YOLO model loading failed: {e}")
            self.model = None

    def detect(self, image):
        if self.model is None:
            return []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb_image)
        detections = []
        for r in results:
            for box in r.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                if int(cls) == 0:  # person class
                    detections.append([x1, y1, x2, y2, conf, cls])
        return np.array(detections)
    
    def draw_detections(self, image, detections):
        img_copy = image.copy()
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf > 0.5:  # Confidence threshold
                    cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img_copy, f'Person: {conf:.2f}', (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return img_copy



class BackgroundSubtractorDetector:
    def __init__(self):
        # Initialize background subtractor
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def detect(self, image):
        # Apply background subtraction
        fgMask = self.backSub.apply(image)
        
        # Apply morphological operations to clean up the mask
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, self.kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (assuming humans are large objects)
        human_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                human_contours.append((x, y, w, h))
        
        return human_contours, fgMask
    
    def draw_detections(self, image, detections):
        img_copy = image.copy()
        contours, mask = detections
        for (x, y, w, h) in contours:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 0), 2)
        return img_copy



def measure_performance(detector, image, detector_name):
    """Measure detection performance including time and basic metrics"""
    start_time = time.time()
    
    try:
        detections = detector.detect(image)
        inference_time = time.time() - start_time
        
        # Count detections
        if isinstance(detections, tuple):
            if len(detections) > 0 and len(detections[0]) > 0:
                detection_count = len(detections[0])
            else:
                detection_count = 0
        elif isinstance(detections, np.ndarray):
            detection_count = len(detections)
        else:
            detection_count = len(detections) if detections is not None else 0
            
        return {
            'detections': detections,
            'inference_time': inference_time,
            'detection_count': detection_count,
            'success': True
        }
    except Exception as e:
        print(f"Error in {detector_name}: {e}")
        return {
            'detections': [],
            'inference_time': 0,
            'detection_count': 0,
            'success': False
        }

def compare_detectors(image_path, detectors_dict):
    """Compare all detectors on a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    results = {}
    
    # Test each detector
    for name, detector in detectors_dict.items():
        print(f"Testing {name}...")
        result = measure_performance(detector, image, name)
        results[name] = result
        
        # Save detection visualization
        if result['success']:
            try:
                visualized = detector.draw_detections(image, result['detections'])
                cv2.imwrite(f'results/{name}_detection_{os.path.basename(image_path)}', visualized)
            except Exception as e:
                print(f"Could not save visualization for {name}: {e}")
    
    return results, image



# Initialize all detectors
print("Initializing detectors...")

# Haar Cascade
haar_detector = HaarCascadeDetector()
print("Haar Cascade initialized")

# HOG + SVM
hog_detector = HOGDetector()
print("HOG detector initialized")

# YOLO (if available)
try:
    yolo_detector = YOLODetector()
    print("YOLO detector initialized")
except:
    yolo_detector = None
    print("YOLO detector not available")

# Background Subtractor
bg_detector = BackgroundSubtractorDetector()
print("Background subtractor initialized")

# Create detectors dictionary
detectors = {
    'Haar_Cascade': haar_detector,
    'HOG_SVM': hog_detector,
    'Background_Subtraction': bg_detector
}

if yolo_detector is not None and yolo_detector.model is not None:
    detectors['YOLOv5'] = yolo_detector

print(f"Initialized {len(detectors)} detectors")



# Set your dataset path
DATASET_PATH = "human detection dataset"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"Dataset folder '{DATASET_PATH}' not found!")
    print("Please make sure your dataset is in the correct location.")
else:
    print(f"Dataset found at: {DATASET_PATH}")

# Find all PNG images in the images folder
test_image_paths = []
if os.path.exists(IMAGES_PATH):
    for file in os.listdir(IMAGES_PATH):
        if file.lower().endswith('.png'):
            test_image_paths.append(os.path.join(IMAGES_PATH, file))
    
    print(f"Found {len(test_image_paths)} PNG images in the dataset")
    if len(test_image_paths) > 0:
        print("First few images:")
        for i, path in enumerate(test_image_paths[:5]):
            print(f"  {i+1}. {os.path.basename(path)}")
else:
    print(f"Images folder '{IMAGES_PATH}' not found!")



# Process a subset of images (to avoid long processing times)
MAX_IMAGES = 20  # Adjust this number based on your needs
selected_images = test_image_paths[:MAX_IMAGES] if len(test_image_paths) > 0 else []

print(f"Processing {len(selected_images)} images from your dataset...")

# Run detection on selected images
all_results = {}

for i, img_path in enumerate(selected_images):
    print(f"\nProcessing image {i+1}/{len(selected_images)}: {os.path.basename(img_path)}")
    results, original_image = compare_detectors(img_path, detectors)
    
    if results is not None:
        all_results[img_path] = results
        
        # Display progress results
        print(f"  Detection Results:")
        for detector_name, result in results.items():
            print(f"    {detector_name:20} | Time: {result['inference_time']:.4f}s | Detections: {result['detection_count']}")
        


def generate_comparison_report(all_results):
    """Generate a comprehensive comparison report"""
    
    # Aggregate results
    performance_stats = defaultdict(list)
    
    for img_path, results in all_results.items():
        for detector_name, result in results.items():
            performance_stats[detector_name].append({
                'time': result['inference_time'],
                'detections': result['detection_count']
            })
    
    # Calculate averages
    avg_stats = {}
    for detector_name, stats in performance_stats.items():
        avg_time = np.mean([s['time'] for s in stats])
        avg_detections = np.mean([s['detections'] for s in stats])
        avg_stats[detector_name] = {
            'avg_time': avg_time,
            'avg_detections': avg_detections,
            'total_tests': len(stats)
        }
    
    return avg_stats

# Generate report
if all_results:
    report = generate_comparison_report(all_results)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON REPORT")
    print("="*60)
    print(f"{'Detector':20} | {'Avg Time (s)':12} | {'Avg Detections':15} | {'Tests':6}")
    print("-"*60)
    
    for detector_name, stats in report.items():
        print(f"{detector_name:20} | {stats['avg_time']:12.4f} | {stats['avg_detections']:15.2f} | {stats['total_tests']:6}")
    
    # Save report to file
    with open('results/comparison_report.txt', 'w') as f:
        f.write("Human Detection Performance Comparison Report\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Images processed: {len(all_results)}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Detector':20} | {'Avg Time (s)':12} | {'Avg Detections':15} | {'Tests':6}\n")
        f.write("-"*60 + "\n")
        for detector_name, stats in report.items():
            f.write(f"{detector_name:20} | {stats['avg_time']:12.4f} | {stats['avg_detections']:15.2f} | {stats['total_tests']:6}\n")
    
    print(f"\nDetailed report saved to results/comparison_report.txt")
else:
    print("No results to generate report from.")



def plot_performance_comparison(report):
    """Create visualizations of the performance comparison"""
    
    if not report:
        print("No data to plot")
        return
    
    detector_names = list(report.keys())
    avg_times = [report[name]['avg_time'] for name in detector_names]
    avg_detections = [report[name]['avg_detections'] for name in detector_names]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average Inference Time
    bars1 = ax1.bar(detector_names, avg_times, color=['red', 'blue', 'green', 'orange'][:len(detector_names)])
    ax1.set_title('Average Inference Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Detectors')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, avg_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    # Plot 2: Average Number of Detections
    bars2 = ax2.bar(detector_names, avg_detections, color=['red', 'blue', 'green', 'orange'][:len(detector_names)])
    ax2.set_title('Average Number of Detections')
    ax2.set_ylabel('Number of Detections')
    ax2.set_xlabel('Detectors')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, det_val in zip(bars2, avg_detections):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{det_val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create performance plots
if 'report' in locals() and report:
    plot_performance_comparison(report)
    print("Performance comparison plots saved to results/performance_comparison.png")
else:
    print("No performance data to plot")



def print_recommendations(report):
    """Print recommendations based on performance"""
    
    if not report:
        return
    
    # Find best performers
    fastest_detector = min(report.items(), key=lambda x: x[1]['avg_time'])
    most_accurate_detector = max(report.items(), key=lambda x: x[1]['avg_detections'])
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"Fastest Detector: {fastest_detector[0]} ({fastest_detector[1]['avg_time']:.4f}s avg)")
    print(f"Most Detections: {most_accurate_detector[0]} ({most_accurate_detector[1]['avg_detections']:.2f} avg)")
    
    print("\nDetailed Analysis:")
    
    # Performance characteristics
    for name, stats in report.items():
        time_category = "Fast" if stats['avg_time'] < 0.1 else "Medium" if stats['avg_time'] < 0.5 else "Slow"
        detection_category = "High" if stats['avg_detections'] > 2 else "Medium" if stats['avg_detections'] > 0 else "Low"
        
        print(f"\n{name}:")
        print(f"  - Speed: {time_category} ({stats['avg_time']:.4f}s)")
        print(f"  - Detection Rate: {detection_category} ({stats['avg_detections']:.2f} detections)")
        
        # Specific recommendations
        if name == "Haar_Cascade":
            print("  - Best for: Real-time applications with moderate accuracy requirements")
        elif name == "HOG_SVM":
            print("  - Best for: Applications requiring good balance of speed and accuracy")
        elif name == "YOLOv5":
            print("  - Best for: High accuracy requirements (if available)")
        elif name == "Background_Subtraction":
            print("  - Best for: Video surveillance with static background")

# Print recommendations
if 'report' in locals() and report:
    print_recommendations(report)



def analyze_detailed_results(all_results):
    """Provide detailed analysis of detection results"""
    
    if not all_results:
        print("No results to analyze")
        return
    
    print("\n" + "="*60)
    print("DETAILED RESULTS ANALYSIS")
    print("="*60)
    
    # Count total detections per detector
    total_detections = defaultdict(int)
    total_images_with_detections = defaultdict(int)
    
    for img_path, results in all_results.items():
        for detector_name, result in results.items():
            total_detections[detector_name] += result['detection_count']
            if result['detection_count'] > 0:
                total_images_with_detections[detector_name] += 1
    
    print(f"Total Images Processed: {len(all_results)}")
    print("\nDetection Summary:")
    print("-" * 50)
    
    for detector_name in detectors.keys():
        if detector_name in total_detections:
            avg_detections = total_detections[detector_name] / len(all_results)
            detection_rate = (total_images_with_detections[detector_name] / len(all_results)) * 100
            print(f"{detector_name:20} | Total: {total_detections[detector_name]:4} | "
                  f"Avg: {avg_detections:5.2f} | Detection Rate: {detection_rate:5.1f}%")

# Run detailed analysis
if all_results:
    analyze_detailed_results(all_results)
