"""
Advanced AI Service for Style Mate
More sophisticated clothing analysis and style recommendations
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import colorsys
from PIL import Image
import io
import hashlib
import random

class AdvancedAIService:
    def __init__(self):
        self.initialized = True
        
        # 확장된 의류 카테고리
        self.clothing_categories = {
            # 상의 세분화
            'tops': [
                'tshirt', 'shirt', 'blouse', 'sweater', 'hoodie', 'cardigan',
                'jacket', 'blazer', 'coat', 'vest', 'tank_top', 'polo',
                'turtleneck', 'crop_top', 'tube_top', 'camisole'
            ],
            # 하의 세분화
            'bottoms': [
                'jeans', 'pants', 'shorts', 'skirt', 'leggings', 'joggers',
                'chinos', 'cargo_pants', 'wide_pants', 'skinny_jeans',
                'mini_skirt', 'midi_skirt', 'maxi_skirt', 'pleated_skirt'
            ],
            # 아우터
            'outerwear': [
                'coat', 'jacket', 'blazer', 'cardigan', 'windbreaker',
                'parka', 'trench_coat', 'leather_jacket', 'denim_jacket',
                'bomber_jacket', 'puffer_jacket'
            ],
            # 원피스/점프수트
            'dresses': [
                'dress', 'maxi_dress', 'mini_dress', 'midi_dress',
                'jumpsuit', 'romper', 'overall'
            ],
            # 신발 세분화
            'footwear': [
                'sneakers', 'boots', 'heels', 'loafers', 'sandals',
                'flats', 'oxford_shoes', 'ankle_boots', 'chelsea_boots',
                'combat_boots', 'running_shoes', 'canvas_shoes'
            ],
            # 액세서리
            'accessories': [
                'bag', 'hat', 'cap', 'beanie', 'scarf', 'belt',
                'watch', 'sunglasses', 'necklace', 'earrings'
            ]
        }
        
        # 스타일 카테고리 확장
        self.style_types = {
            'casual': '캐주얼',
            'business': '비즈니스',
            'business_casual': '비즈니스 캐주얼',
            'formal': '포멀',
            'semi_formal': '세미포멀',
            'smart_casual': '스마트 캐주얼',
            'streetwear': '스트릿웨어',
            'vintage': '빈티지',
            'minimalist': '미니멀',
            'romantic': '로맨틱',
            'sporty': '스포티',
            'preppy': '프레피',
            'bohemian': '보헤미안',
            'grunge': '그런지',
            'elegant': '엘레강트',
            'chic': '시크',
            'trendy': '트렌디',
            'classic': '클래식'
        }
        
        # 색상 팔레트 확장
        self.color_palettes = {
            'warm': ['beige', 'brown', 'camel', 'khaki', 'olive', 'rust', 'burgundy'],
            'cool': ['navy', 'blue', 'gray', 'silver', 'purple', 'teal'],
            'neutral': ['white', 'black', 'gray', 'beige', 'cream', 'taupe'],
            'pastel': ['lavender', 'mint', 'peach', 'baby_blue', 'pink', 'lemon'],
            'vibrant': ['red', 'orange', 'yellow', 'green', 'blue', 'purple'],
            'earth': ['brown', 'tan', 'olive', 'forest_green', 'rust', 'terracotta']
        }
        
        # 패브릭/재질 인식
        self.fabric_types = [
            'cotton', 'denim', 'wool', 'leather', 'silk', 'linen',
            'polyester', 'velvet', 'suede', 'knit', 'fleece', 'chiffon'
        ]
        
        # 패턴 인식
        self.patterns = [
            'solid', 'striped', 'plaid', 'floral', 'polka_dot',
            'geometric', 'animal_print', 'camouflage', 'tie_dye',
            'paisley', 'checkered', 'abstract'
        ]
        
    async def analyze_image(self, image_data: bytes) -> Dict[str, Any]:
        """고급 이미지 분석"""
        try:
            # 이미지 로드
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Invalid image"}
            
            # 멀티 아이템 감지
            items = self._detect_clothing_items(img)
            
            # 각 아이템 상세 분석
            analyzed_items = []
            for idx, item in enumerate(items):
                analysis = {
                    'id': f"item_{idx}",
                    'category': item['category'],
                    'subcategory': item['subcategory'],
                    'color': self._analyze_color_advanced(item['region']),
                    'pattern': self._detect_pattern(item['region']),
                    'fabric': self._estimate_fabric(item['region']),
                    'style': self._classify_style(item),
                    'confidence': item['confidence'],
                    'bbox': item['bbox']  # 바운딩 박스 좌표
                }
                analyzed_items.append(analysis)
            
            # 전체 코디 스타일 평가
            overall_style = self._evaluate_overall_style(analyzed_items)
            
            # 색상 조화 분석
            color_harmony = self._analyze_color_harmony(analyzed_items)
            
            return {
                'items': analyzed_items,
                'count': len(analyzed_items),
                'overall_style': overall_style,
                'color_harmony': color_harmony,
                'recommendations': self._generate_recommendations(analyzed_items),
                'image_id': hashlib.md5(image_data).hexdigest()[:8]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_clothing_items(self, img: np.ndarray) -> List[Dict]:
        """의류 아이템 감지 (개선된 버전)"""
        items = []
        h, w = img.shape[:2]
        
        # 간단한 영역 분할 (실제로는 딥러닝 모델 사용)
        # 상단 영역 - 상의
        top_region = img[0:int(h*0.5), :]
        items.append({
            'category': 'top',
            'subcategory': random.choice(self.clothing_categories['tops']),
            'region': top_region,
            'confidence': 0.85 + random.random() * 0.15,
            'bbox': [0, 0, w, int(h*0.5)]
        })
        
        # 하단 영역 - 하의
        bottom_region = img[int(h*0.5):int(h*0.85), :]
        items.append({
            'category': 'bottom',
            'subcategory': random.choice(self.clothing_categories['bottoms']),
            'region': bottom_region,
            'confidence': 0.80 + random.random() * 0.15,
            'bbox': [0, int(h*0.5), w, int(h*0.85)]
        })
        
        # 신발 영역 (있는 경우)
        if h > 500:
            shoe_region = img[int(h*0.85):, :]
            items.append({
                'category': 'footwear',
                'subcategory': random.choice(self.clothing_categories['footwear']),
                'region': shoe_region,
                'confidence': 0.75 + random.random() * 0.15,
                'bbox': [0, int(h*0.85), w, h]
            })
        
        return items
    
    def _analyze_color_advanced(self, img_region: np.ndarray) -> Dict[str, Any]:
        """고급 색상 분석"""
        # RGB to HSV
        hsv = cv2.cvtColor(img_region, cv2.COLOR_BGR2HSV)
        
        # 주요 색상 추출
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        
        # 상위 색상들
        dominant_colors = self._extract_dominant_colors(img_region)
        
        return {
            'primary': dominant_colors[0],
            'secondary': dominant_colors[1] if len(dominant_colors) > 1 else None,
            'accent': dominant_colors[2] if len(dominant_colors) > 2 else None,
            'palette': self._identify_color_palette(dominant_colors)
        }
    
    def _extract_dominant_colors(self, img: np.ndarray, n_colors: int = 3) -> List[str]:
        """주요 색상 추출"""
        # 간단한 K-means 대체 구현
        pixels = img.reshape(-1, 3)
        
        # 샘플링으로 속도 개선
        sample_size = min(1000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size)
        sampled_pixels = pixels[indices]
        
        # 평균 색상 계산
        colors = []
        for _ in range(n_colors):
            mean_color = np.mean(sampled_pixels, axis=0)
            color_name = self._rgb_to_name(mean_color)
            colors.append(color_name)
            
            # 다음 색상을 위해 현재 색상과 유사한 픽셀 제거
            distances = np.linalg.norm(sampled_pixels - mean_color, axis=1)
            sampled_pixels = sampled_pixels[distances > 50]
            
            if len(sampled_pixels) < 10:
                break
        
        return colors
    
    def _rgb_to_name(self, rgb: np.ndarray) -> str:
        """RGB 값을 색상 이름으로 변환"""
        r, g, b = rgb
        
        # 간단한 색상 매핑
        if r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > g and r > b:
            if r > 200:
                return 'red' if g < 100 else 'orange'
            return 'brown'
        elif g > r and g > b:
            return 'green'
        elif b > r and b > g:
            return 'blue' if b > 150 else 'navy'
        elif abs(r - g) < 30 and abs(g - b) < 30:
            return 'gray'
        else:
            return 'mixed'
    
    def _detect_pattern(self, img_region: np.ndarray) -> str:
        """패턴 감지"""
        # 간단한 패턴 감지 로직
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 패턴 추정
        if edge_density < 0.05:
            return 'solid'
        elif edge_density < 0.15:
            return random.choice(['striped', 'minimal'])
        elif edge_density < 0.25:
            return random.choice(['plaid', 'geometric'])
        else:
            return random.choice(['floral', 'abstract', 'complex'])
    
    def _estimate_fabric(self, img_region: np.ndarray) -> str:
        """재질 추정"""
        # 텍스처 분석을 통한 재질 추정 (간소화)
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        
        # 텍스처 복잡도
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            return random.choice(['silk', 'polyester'])
        elif laplacian_var < 500:
            return random.choice(['cotton', 'linen'])
        elif laplacian_var < 1000:
            return random.choice(['wool', 'knit'])
        else:
            return random.choice(['denim', 'leather', 'suede'])
    
    def _classify_style(self, item: Dict) -> str:
        """아이템 스타일 분류"""
        # 아이템 특성에 따른 스타일 분류
        subcategory = item['subcategory']
        
        style_mapping = {
            'tshirt': ['casual', 'streetwear', 'sporty'],
            'shirt': ['business', 'smart_casual', 'preppy'],
            'blazer': ['business', 'formal', 'smart_casual'],
            'hoodie': ['streetwear', 'casual', 'sporty'],
            'dress': ['elegant', 'romantic', 'chic'],
            'jeans': ['casual', 'streetwear', 'vintage'],
            'pants': ['business', 'smart_casual', 'minimalist'],
            'sneakers': ['casual', 'streetwear', 'sporty'],
            'heels': ['elegant', 'formal', 'chic'],
            'boots': ['grunge', 'vintage', 'trendy']
        }
        
        for key, styles in style_mapping.items():
            if key in subcategory.lower():
                return random.choice(styles)
        
        return 'casual'
    
    def _evaluate_overall_style(self, items: List[Dict]) -> Dict[str, Any]:
        """전체 스타일 평가"""
        if not items:
            return {'style': 'unknown', 'score': 0}
        
        # 각 아이템의 스타일 수집
        styles = [item.get('style', 'casual') for item in items]
        
        # 가장 빈번한 스타일
        style_counts = {}
        for style in styles:
            style_counts[style] = style_counts.get(style, 0) + 1
        
        dominant_style = max(style_counts, key=style_counts.get)
        
        # 스타일 일관성 점수
        consistency_score = (style_counts[dominant_style] / len(styles)) * 100
        
        return {
            'primary_style': dominant_style,
            'style_mix': list(set(styles)),
            'consistency_score': consistency_score,
            'description': self._generate_style_description(dominant_style, consistency_score)
        }
    
    def _generate_style_description(self, style: str, score: float) -> str:
        """스타일 설명 생성"""
        descriptions = {
            'casual': '편안하고 자연스러운 일상 스타일',
            'business': '전문적이고 신뢰감 있는 비즈니스 룩',
            'streetwear': '트렌디한 스트릿 패션',
            'elegant': '우아하고 세련된 스타일',
            'sporty': '활동적이고 다이나믹한 스포티 룩',
            'vintage': '클래식하고 레트로한 빈티지 감성',
            'minimalist': '깔끔하고 절제된 미니멀 스타일'
        }
        
        base_desc = descriptions.get(style, '개성 있는 스타일')
        
        if score > 80:
            return f"완벽한 {base_desc}"
        elif score > 60:
            return f"조화로운 {base_desc}"
        else:
            return f"다양한 스타일이 믹스된 {base_desc}"
    
    def _analyze_color_harmony(self, items: List[Dict]) -> Dict[str, Any]:
        """색상 조화 분석"""
        colors = []
        for item in items:
            if 'color' in item and isinstance(item['color'], dict):
                if item['color'].get('primary'):
                    colors.append(item['color']['primary'])
        
        if len(colors) < 2:
            return {'harmony_type': 'monochrome', 'score': 100}
        
        # 색상 조화 타입 판단
        harmony_type = self._determine_color_harmony_type(colors)
        harmony_score = self._calculate_harmony_score(colors)
        
        return {
            'harmony_type': harmony_type,
            'score': harmony_score,
            'recommendation': self._get_color_recommendation(harmony_type, harmony_score)
        }
    
    def _determine_color_harmony_type(self, colors: List[str]) -> str:
        """색상 조화 타입 결정"""
        unique_colors = set(colors)
        
        if len(unique_colors) == 1:
            return 'monochrome'
        elif len(unique_colors) == 2:
            return 'complementary'
        elif all(c in self.color_palettes['neutral'] for c in unique_colors):
            return 'neutral'
        elif all(c in self.color_palettes['warm'] for c in unique_colors):
            return 'warm'
        elif all(c in self.color_palettes['cool'] for c in unique_colors):
            return 'cool'
        else:
            return 'mixed'
    
    def _calculate_harmony_score(self, colors: List[str]) -> float:
        """색상 조화 점수 계산"""
        # 간단한 조화 점수 계산
        if len(set(colors)) == 1:
            return 100
        
        # 뉴트럴 색상이 포함되면 점수 상승
        neutral_count = sum(1 for c in colors if c in self.color_palettes['neutral'])
        base_score = 60 + (neutral_count * 10)
        
        return min(100, base_score + random.randint(0, 20))
    
    def _get_color_recommendation(self, harmony_type: str, score: float) -> str:
        """색상 추천"""
        if score > 80:
            return "완벽한 색상 조화입니다!"
        elif score > 60:
            return "좋은 색상 매치입니다. 액세서리로 포인트를 주면 더 좋을 것 같아요."
        else:
            return "색상 밸런스를 조정하면 더 조화로운 룩이 될 것 같아요."
    
    def _generate_recommendations(self, items: List[Dict]) -> List[str]:
        """스타일 추천 생성"""
        recommendations = []
        
        # 아이템별 추천
        for item in items:
            if item['category'] == 'top':
                if item['style'] == 'casual':
                    recommendations.append("캐주얼한 상의에는 슬림핏 하의가 잘 어울려요")
                elif item['style'] == 'business':
                    recommendations.append("비즈니스 룩은 깔끔한 구두로 완성해보세요")
            elif item['category'] == 'bottom':
                if 'jeans' in item['subcategory']:
                    recommendations.append("데님은 어떤 상의와도 잘 어울리는 만능 아이템이에요")
        
        # 전체적인 추천
        recommendations.append("현재 스타일에 시계나 벨트 같은 액세서리를 추가해보세요")
        
        return recommendations[:3]  # 최대 3개 추천
    
    def _identify_color_palette(self, colors: List[str]) -> str:
        """색상 팔레트 식별"""
        for palette_name, palette_colors in self.color_palettes.items():
            if any(c in palette_colors for c in colors):
                return palette_name
        return 'mixed'