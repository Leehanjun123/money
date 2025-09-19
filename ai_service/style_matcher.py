import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class StyleMatcher:
    """AI 기반 스타일 매칭 엔진"""
    
    # 스타일 카테고리 정의
    STYLE_CATEGORIES = {
        'minimal': {
            'name': '미니멀',
            'keywords': ['심플', '깔끔', '모노톤', '베이직', '무채색'],
            'colors': ['화이트', '블랙', '그레이', '베이지', '네이비'],
            'brands': ['COS', 'MUJI', 'UNIQLO', 'ZARA'],
            'characteristics': {
                'pattern_complexity': 0.1,
                'color_variety': 0.2,
                'silhouette': 'straight',
                'fit': 'relaxed'
            }
        },
        'casual': {
            'name': '캐주얼',
            'keywords': ['편안', '데일리', '활동적', '자연스러운'],
            'colors': ['블루', '화이트', '카키', '베이지', '그레이'],
            'brands': ['GAP', 'POLO', 'TOMMY', 'LACOSTE'],
            'characteristics': {
                'pattern_complexity': 0.3,
                'color_variety': 0.4,
                'silhouette': 'regular',
                'fit': 'regular'
            }
        },
        'street': {
            'name': '스트릿',
            'keywords': ['힙합', '스케이터', '오버사이즈', '그래픽', '로고'],
            'colors': ['블랙', '레드', '옐로우', '네온', '화이트'],
            'brands': ['SUPREME', 'STUSSY', 'THISISNEVERTHAT', 'COVERNAT'],
            'characteristics': {
                'pattern_complexity': 0.7,
                'color_variety': 0.6,
                'silhouette': 'oversized',
                'fit': 'loose'
            }
        },
        'business': {
            'name': '비즈니스',
            'keywords': ['포멀', '정장', '오피스', '클래식', '단정'],
            'colors': ['네이비', '차콜', '화이트', '블랙', '그레이'],
            'brands': ['ZEGNA', 'BOSS', 'BROOKS BROTHERS', 'RALPH LAUREN'],
            'characteristics': {
                'pattern_complexity': 0.2,
                'color_variety': 0.1,
                'silhouette': 'tailored',
                'fit': 'slim'
            }
        },
        'vintage': {
            'name': '빈티지',
            'keywords': ['레트로', '구제', '클래식', '유니크', '올드스쿨'],
            'colors': ['브라운', '머스타드', '버건디', '올리브', '크림'],
            'brands': ['LEVI\'S VINTAGE', 'CHAMPION REVERSE WEAVE', 'DICKIES'],
            'characteristics': {
                'pattern_complexity': 0.5,
                'color_variety': 0.5,
                'silhouette': 'relaxed',
                'fit': 'regular'
            }
        },
        'sporty': {
            'name': '스포티',
            'keywords': ['애슬레저', '액티브', '테크웨어', '기능성', '퍼포먼스'],
            'colors': ['블랙', '화이트', '네온', '그레이', '블루'],
            'brands': ['NIKE', 'ADIDAS', 'UNDER ARMOUR', 'LULULEMON'],
            'characteristics': {
                'pattern_complexity': 0.4,
                'color_variety': 0.3,
                'silhouette': 'athletic',
                'fit': 'performance'
            }
        }
    }
    
    def __init__(self):
        self.style_vectors = self._build_style_vectors()
    
    def _build_style_vectors(self) -> Dict[str, np.ndarray]:
        """스타일별 특징 벡터 생성"""
        vectors = {}
        for style_id, style_data in self.STYLE_CATEGORIES.items():
            chars = style_data['characteristics']
            vector = np.array([
                chars['pattern_complexity'],
                chars['color_variety'],
                self._encode_silhouette(chars['silhouette']),
                self._encode_fit(chars['fit']),
                len(style_data['colors']) / 10,  # 색상 다양성
                len(style_data['keywords']) / 10,  # 키워드 풍부도
            ])
            vectors[style_id] = vector
        return vectors
    
    def _encode_silhouette(self, silhouette: str) -> float:
        """실루엣 인코딩"""
        encoding = {
            'straight': 0.2,
            'regular': 0.4,
            'relaxed': 0.5,
            'oversized': 0.7,
            'tailored': 0.3,
            'athletic': 0.6
        }
        return encoding.get(silhouette, 0.5)
    
    def _encode_fit(self, fit: str) -> float:
        """핏 인코딩"""
        encoding = {
            'slim': 0.2,
            'regular': 0.5,
            'relaxed': 0.6,
            'loose': 0.8,
            'performance': 0.4
        }
        return encoding.get(fit, 0.5)
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """이미지 분석하여 스타일 특징 추출"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                image_pil = Image.open(image_path)
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # 색상 분석
            color_features = self._analyze_colors(image)
            
            # 패턴 복잡도 분석
            pattern_complexity = self._analyze_pattern_complexity(image)
            
            # 실루엣 추정
            silhouette = self._estimate_silhouette(image)
            
            return {
                'color_features': color_features,
                'pattern_complexity': pattern_complexity,
                'silhouette': silhouette,
                'style_vector': self._create_item_vector(
                    color_features, pattern_complexity, silhouette
                )
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._default_features()
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """색상 분석"""
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 색상 히스토그램
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # 주요 색상 추출
        dominant_hue = np.argmax(hist_h)
        avg_saturation = np.mean(hist_s)
        avg_value = np.mean(hist_v)
        
        # 색상 다양성
        color_variety = np.std(hist_h) / np.mean(hist_h) if np.mean(hist_h) > 0 else 0
        
        return {
            'dominant_hue': float(dominant_hue),
            'saturation': float(avg_saturation),
            'brightness': float(avg_value),
            'variety': float(color_variety)
        }
    
    def _analyze_pattern_complexity(self, image: np.ndarray) -> float:
        """패턴 복잡도 분석"""
        # 엣지 검출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 엣지 밀도로 복잡도 측정
        edge_density = np.sum(edges > 0) / edges.size
        
        # 텍스처 분석 (간단한 버전)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = np.var(laplacian) / 10000
        
        return min(float(edge_density + texture_complexity), 1.0)
    
    def _estimate_silhouette(self, image: np.ndarray) -> str:
        """실루엣 추정"""
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 0.8:
            return 'oversized'
        elif aspect_ratio > 0.6:
            return 'relaxed'
        elif aspect_ratio > 0.4:
            return 'regular'
        else:
            return 'slim'
    
    def _create_item_vector(self, color_features: Dict, 
                           pattern_complexity: float, 
                           silhouette: str) -> np.ndarray:
        """아이템 특징 벡터 생성"""
        return np.array([
            pattern_complexity,
            color_features['variety'],
            self._encode_silhouette(silhouette),
            0.5,  # 기본 핏 값
            color_features['saturation'] / 255,
            color_features['brightness'] / 255,
        ])
    
    def _default_features(self) -> Dict[str, Any]:
        """기본 특징값"""
        return {
            'color_features': {
                'dominant_hue': 0,
                'saturation': 128,
                'brightness': 128,
                'variety': 0.5
            },
            'pattern_complexity': 0.3,
            'silhouette': 'regular',
            'style_vector': np.array([0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
        }
    
    def match_style(self, item_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """스타일 매칭"""
        item_vector = item_features.get('style_vector', self._default_features()['style_vector'])
        
        matches = []
        for style_id, style_vector in self.style_vectors.items():
            similarity = cosine_similarity(
                item_vector.reshape(1, -1),
                style_vector.reshape(1, -1)
            )[0][0]
            
            matches.append({
                'style_id': style_id,
                'style_name': self.STYLE_CATEGORIES[style_id]['name'],
                'similarity': float(similarity),
                'confidence': float(min(similarity * 100, 100))
            })
        
        # 유사도 순으로 정렬
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:3]  # 상위 3개 스타일 반환
    
    def recommend_coordination(self, 
                             user_items: List[Dict], 
                             target_style: str = None) -> List[Dict[str, Any]]:
        """코디 추천"""
        recommendations = []
        
        # 타겟 스타일이 없으면 사용자 아이템에서 추론
        if not target_style:
            style_scores = {}
            for item in user_items:
                if 'style_matches' in item:
                    for match in item['style_matches']:
                        style_id = match['style_id']
                        style_scores[style_id] = style_scores.get(style_id, 0) + match['similarity']
            
            if style_scores:
                target_style = max(style_scores, key=style_scores.get)
            else:
                target_style = 'casual'  # 기본값
        
        style_data = self.STYLE_CATEGORIES.get(target_style, self.STYLE_CATEGORIES['casual'])
        
        # 카테고리별 추천 아이템
        outfit_template = {
            'TOP': {'required': True, 'items': []},
            'BOTTOM': {'required': True, 'items': []},
            'OUTER': {'required': False, 'items': []},
            'SHOES': {'required': True, 'items': []},
            'ACC': {'required': False, 'items': []}
        }
        
        # 사용자 아이템 분류
        for item in user_items:
            category = item.get('category', 'TOP')
            if category in outfit_template:
                outfit_template[category]['items'].append(item)
        
        # 부족한 카테고리에 대한 추천
        for category, data in outfit_template.items():
            if data['required'] and not data['items']:
                recommendations.append({
                    'category': category,
                    'style': target_style,
                    'suggestion': f"{style_data['name']} 스타일 {category} 추천",
                    'keywords': style_data['keywords'],
                    'recommended_colors': style_data['colors'][:3],
                    'recommended_brands': style_data['brands'][:3]
                })
        
        return recommendations
    
    def calculate_outfit_score(self, outfit_items: List[Dict]) -> Dict[str, Any]:
        """전체 코디 점수 계산"""
        if not outfit_items:
            return {'score': 0, 'feedback': '아이템이 없습니다'}
        
        # 색상 조화 점수
        color_score = self._calculate_color_harmony(outfit_items)
        
        # 스타일 일관성 점수
        style_score = self._calculate_style_consistency(outfit_items)
        
        # 계절 적합성 점수
        season_score = self._calculate_season_compatibility(outfit_items)
        
        # 전체 점수
        total_score = (color_score * 0.4 + style_score * 0.4 + season_score * 0.2)
        
        # 피드백 생성
        feedback = []
        if color_score < 60:
            feedback.append("색상 조합을 다시 고려해보세요")
        if style_score < 60:
            feedback.append("스타일 통일성이 부족합니다")
        if season_score < 60:
            feedback.append("계절에 맞지 않는 아이템이 있습니다")
        
        if not feedback:
            feedback.append("훌륭한 코디입니다!")
        
        return {
            'total_score': float(total_score),
            'color_score': float(color_score),
            'style_score': float(style_score),
            'season_score': float(season_score),
            'feedback': feedback
        }
    
    def _calculate_color_harmony(self, items: List[Dict]) -> float:
        """색상 조화도 계산"""
        # 간단한 구현 - 실제로는 더 정교한 색상 이론 적용
        colors = [item.get('color', '') for item in items]
        unique_colors = len(set(colors))
        
        # 2-4개 색상이 이상적
        if 2 <= unique_colors <= 4:
            return 90.0
        elif unique_colors == 1:
            return 70.0  # 모노톤
        elif unique_colors > 4:
            return max(50.0, 100 - (unique_colors - 4) * 10)
        else:
            return 60.0
    
    def _calculate_style_consistency(self, items: List[Dict]) -> float:
        """스타일 일관성 계산"""
        if not items:
            return 0.0
        
        style_counts = {}
        for item in items:
            if 'style_matches' in item:
                for match in item['style_matches']:
                    style_id = match['style_id']
                    style_counts[style_id] = style_counts.get(style_id, 0) + 1
        
        if not style_counts:
            return 50.0
        
        # 가장 많은 스타일의 비율
        max_count = max(style_counts.values())
        consistency_ratio = max_count / len(items)
        
        return min(consistency_ratio * 100, 100)
    
    def _calculate_season_compatibility(self, items: List[Dict]) -> float:
        """계절 적합성 계산"""
        # 현재 계절 판단 (간단한 구현)
        import datetime
        month = datetime.datetime.now().month
        
        if 3 <= month <= 5:
            season = 'spring'
        elif 6 <= month <= 8:
            season = 'summer'
        elif 9 <= month <= 11:
            season = 'fall'
        else:
            season = 'winter'
        
        # 계절별 적합 아이템 체크 (간단한 구현)
        season_score = 80.0  # 기본 점수
        
        for item in items:
            material = item.get('material', '').lower()
            category = item.get('category', '')
            
            if season == 'summer':
                if '울' in material or '니트' in material:
                    season_score -= 10
                if category == 'OUTER' and '패딩' in item.get('name', ''):
                    season_score -= 20
            elif season == 'winter':
                if '린넨' in material or '메쉬' in material:
                    season_score -= 10
                    
        return max(season_score, 0)