"""
AI Service for clothing detection and analysis
Multi-model ensemble for maximum accuracy
"""

import asyncio
import io
import base64
import hashlib
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import logging

logger = logging.getLogger(__name__)

class AIService:
    """
    Ensemble AI service combining multiple models
    for accurate clothing detection and analysis
    """
    
    def __init__(self, cache_service=None):
        self.cache_service = cache_service
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.yolo_model = None
        self.segformer_model = None
        self.segformer_processor = None
        
        # Model weights
        self.ensemble_weights = {
            "yolo": 0.4,
            "segformer": 0.6
        }
        
        # Clothing categories
        self.categories = {
            0: "top",
            1: "bottom", 
            2: "dress",
            3: "outerwear",
            4: "shoes",
            5: "bag",
            6: "accessory"
        }
        
        # Color detection
        self.colors = {
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "gray": [128, 128, 128],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "green": [0, 255, 0],
            "yellow": [255, 255, 0],
            "pink": [255, 192, 203],
            "purple": [128, 0, 128],
            "orange": [255, 165, 0],
            "brown": [165, 42, 42],
            "beige": [245, 245, 220],
            "navy": [0, 0, 128]
        }
        
    async def warmup(self):
        """Initialize and warm up models"""
        try:
            logger.info("Loading AI models...")
            
            # Load YOLOv8n (fastest variant)
            self.yolo_model = YOLO('yolov8n.pt')
            
            # Load Segformer-b0 (lightweight variant)
            self.segformer_processor = SegformerImageProcessor.from_pretrained(
                "mattmdjaga/segformer_b0_clothes"
            )
            self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                "mattmdjaga/segformer_b0_clothes"
            ).to(self.device)
            self.segformer_model.eval()
            
            # Warm up with dummy image
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            await self._process_yolo(dummy_img)
            await self._process_segformer(dummy_img)
            
            logger.info("✅ AI models loaded and warmed up")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    async def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze clothing image with ensemble approach
        """
        start_time = asyncio.get_event_loop().time()
        
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Run models in parallel
        yolo_task = asyncio.create_task(self._process_yolo(image_np))
        segformer_task = asyncio.create_task(self._process_segformer(image_np))
        
        yolo_results, segformer_results = await asyncio.gather(
            yolo_task, segformer_task
        )
        
        # Ensemble results
        final_results = self._ensemble_results(yolo_results, segformer_results)
        
        # Extract attributes
        attributes = await self._extract_attributes(image_np, final_results)
        
        # Add metadata
        final_results.update({
            "attributes": attributes,
            "processing_time": asyncio.get_event_loop().time() - start_time,
            "confidence": final_results.get("confidence", 0.0)
        })
        
        return final_results
    
    async def _process_yolo(self, image: np.ndarray) -> Dict:
        """Process image with YOLOv8"""
        try:
            results = self.yolo_model(image, conf=0.5)
            
            detections = []
            for r in results:
                for box in r.boxes:
                    detection = {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class": int(box.cls[0])
                    }
                    detections.append(detection)
            
            return {"detections": detections, "model": "yolo"}
            
        except Exception as e:
            logger.error(f"YOLO processing failed: {e}")
            return {"detections": [], "model": "yolo"}
    
    async def _process_segformer(self, image: np.ndarray) -> Dict:
        """Process image with Segformer"""
        try:
            # Preprocess
            inputs = self.segformer_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.segformer_model(**inputs)
                
            # Post-process
            predicted = torch.nn.functional.interpolate(
                outputs.logits,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False
            )
            
            seg_mask = predicted.argmax(dim=1)[0].cpu().numpy()
            
            # Extract segments
            segments = []
            for class_id in np.unique(seg_mask):
                if class_id == 0:  # Skip background
                    continue
                    
                mask = (seg_mask == class_id).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    
                    if area > 1000:  # Filter small regions
                        segments.append({
                            "bbox": [x, y, x+w, y+h],
                            "class": int(class_id),
                            "area": float(area),
                            "confidence": 0.9  # Segformer doesn't provide confidence
                        })
            
            return {"segments": segments, "model": "segformer"}
            
        except Exception as e:
            logger.error(f"Segformer processing failed: {e}")
            return {"segments": [], "model": "segformer"}
    
    def _ensemble_results(self, yolo_results: Dict, segformer_results: Dict) -> Dict:
        """Combine results from multiple models"""
        combined = []
        
        # Process YOLO detections
        for det in yolo_results.get("detections", []):
            combined.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"] * self.ensemble_weights["yolo"],
                "class": det["class"],
                "source": "yolo"
            })
        
        # Process Segformer segments
        for seg in segformer_results.get("segments", []):
            combined.append({
                "bbox": seg["bbox"],
                "confidence": seg["confidence"] * self.ensemble_weights["segformer"],
                "class": seg["class"],
                "source": "segformer"
            })
        
        # Merge overlapping detections
        final_items = self._merge_overlapping(combined)
        
        return {
            "items": final_items,
            "count": len(final_items),
            "confidence": np.mean([item["confidence"] for item in final_items]) if final_items else 0
        }
    
    def _merge_overlapping(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Merge overlapping detections using NMS"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            # Find overlapping detections
            overlapping = [det1]
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                iou = self._calculate_iou(det1["bbox"], det2["bbox"])
                if iou > iou_threshold:
                    overlapping.append(det2)
                    used.add(j)
            
            # Merge overlapping detections
            if len(overlapping) > 1:
                merged_det = self._merge_detections(overlapping)
            else:
                merged_det = det1
                
            merged.append(merged_det)
            used.add(i)
        
        return merged
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _merge_detections(self, detections: List[Dict]) -> Dict:
        """Merge multiple detections into one"""
        # Average bounding box
        bbox = np.mean([d["bbox"] for d in detections], axis=0).tolist()
        
        # Weighted confidence
        confidence = sum([d["confidence"] for d in detections]) / len(detections)
        
        # Most common class
        classes = [d["class"] for d in detections]
        class_id = max(set(classes), key=classes.count)
        
        # Combined sources
        sources = list(set([d["source"] for d in detections]))
        
        return {
            "bbox": bbox,
            "confidence": min(confidence * 1.2, 1.0),  # Boost confidence for agreement
            "class": class_id,
            "category": self.categories.get(class_id, "unknown"),
            "sources": sources
        }
    
    async def _extract_attributes(self, image: np.ndarray, results: Dict) -> Dict:
        """Extract detailed attributes from detected items"""
        attributes = []
        
        for item in results.get("items", []):
            bbox = item["bbox"]
            x1, y1, x2, y2 = [int(b) for b in bbox]
            
            # Crop item
            item_img = image[y1:y2, x1:x2]
            
            if item_img.size == 0:
                continue
            
            # Extract color
            dominant_color = self._get_dominant_color(item_img)
            
            # Estimate pattern
            pattern = self._detect_pattern(item_img)
            
            attributes.append({
                "category": item["category"],
                "color": dominant_color,
                "pattern": pattern,
                "confidence": item["confidence"]
            })
        
        return attributes
    
    def _get_dominant_color(self, image: np.ndarray) -> str:
        """Get dominant color from image region"""
        # Resize for faster processing
        small = cv2.resize(image, (50, 50))
        
        # K-means clustering for dominant color
        pixels = small.reshape(-1, 3)
        pixels = pixels[np.random.choice(pixels.shape[0], min(100, pixels.shape[0]), replace=False)]
        
        if len(pixels) == 0:
            return "unknown"
        
        # Find closest named color
        mean_color = np.mean(pixels, axis=0)
        
        min_dist = float('inf')
        closest_color = "unknown"
        
        for name, rgb in self.colors.items():
            dist = np.linalg.norm(mean_color - rgb)
            if dist < min_dist:
                min_dist = dist
                closest_color = name
        
        return closest_color
    
    def _detect_pattern(self, image: np.ndarray) -> str:
        """Detect pattern in clothing item"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate variance for pattern detection
        variance = np.var(gray)
        
        # Simple pattern detection based on variance
        if variance < 100:
            return "solid"
        elif variance < 500:
            return "subtle_pattern"
        else:
            # Check for stripes
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
            
            if lines is not None and len(lines) > 10:
                return "striped"
            else:
                return "patterned"
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.yolo_model:
            del self.yolo_model
        if self.segformer_model:
            del self.segformer_model
        torch.cuda.empty_cache()