import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

class VirtualFittingEngine:
    """AR 가상 피팅 엔진"""
    
    # 표준 사이즈 차트 (cm)
    SIZE_CHART = {
        'TOP': {
            'XS': {'chest': 88, 'shoulder': 40, 'length': 62},
            'S': {'chest': 92, 'shoulder': 42, 'length': 64},
            'M': {'chest': 96, 'shoulder': 44, 'length': 66},
            'L': {'chest': 100, 'shoulder': 46, 'length': 68},
            'XL': {'chest': 104, 'shoulder': 48, 'length': 70},
        },
        'BOTTOM': {
            'XS': {'waist': 68, 'hip': 90, 'length': 95},
            'S': {'waist': 72, 'hip': 94, 'length': 97},
            'M': {'waist': 76, 'hip': 98, 'length': 99},
            'L': {'waist': 80, 'hip': 102, 'length': 101},
            'XL': {'waist': 84, 'hip': 106, 'length': 103},
        }
    }
    
    # 체형별 아바타 조정 값
    BODY_TYPE_ADJUSTMENTS = {
        'straight': {'chest_adj': 0, 'waist_adj': 0, 'hip_adj': 0},
        'pear': {'chest_adj': -2, 'waist_adj': -2, 'hip_adj': 3},
        'apple': {'chest_adj': 2, 'waist_adj': 3, 'hip_adj': -1},
        'hourglass': {'chest_adj': 1, 'waist_adj': -3, 'hip_adj': 1},
        'inverted_triangle': {'chest_adj': 3, 'waist_adj': 0, 'hip_adj': -2},
    }
    
    def __init__(self):
        self.pose_detector = self._initialize_pose_detector()
    
    def _initialize_pose_detector(self):
        """포즈 검출기 초기화"""
        try:
            # MediaPipe를 사용한 포즈 검출 (실제 구현에서는 import 필요)
            # import mediapipe as mp
            # mp_pose = mp.solutions.pose
            # return mp_pose.Pose()
            
            # 현재는 Mock 객체
            return MockPoseDetector()
        except Exception as e:
            logger.warning(f"Pose detector initialization failed: {e}")
            return None
    
    def analyze_body_measurements(self, user_photo: np.ndarray, 
                                height: int = None) -> Dict[str, float]:
        """사용자 신체 치수 분석"""
        
        if self.pose_detector is None:
            return self._default_measurements(height)
        
        try:
            # 포즈 검출
            pose_landmarks = self.pose_detector.detect(user_photo)
            
            if not pose_landmarks:
                return self._default_measurements(height)
            
            # 키포인트에서 신체 치수 계산
            measurements = self._calculate_measurements_from_pose(
                pose_landmarks, user_photo.shape, height
            )
            
            return measurements
            
        except Exception as e:
            logger.error(f"Body measurement analysis failed: {e}")
            return self._default_measurements(height)
    
    def _calculate_measurements_from_pose(self, landmarks: List, 
                                        image_shape: Tuple, 
                                        height: int) -> Dict[str, float]:
        """포즈 랜드마크에서 신체 치수 계산"""
        
        # 이미지 크기
        img_height, img_width = image_shape[:2]
        
        # 주요 포인트 추출 (정규화된 좌표를 픽셀로 변환)
        def get_point(landmark_id):
            landmark = landmarks[landmark_id]
            return (landmark['x'] * img_width, landmark['y'] * img_height)
        
        # 어깨 포인트
        left_shoulder = get_point(11)  # MediaPipe 기준
        right_shoulder = get_point(12)
        
        # 허리 포인트 (힙과 어깨 중간)
        left_hip = get_point(23)
        right_hip = get_point(24)
        
        # 거리 계산
        shoulder_width_px = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        hip_width_px = np.linalg.norm(np.array(left_hip) - np.array(right_hip))
        
        # 전체 신장 픽셀
        body_height_px = abs(left_shoulder[1] - left_hip[1]) * 2  # 대략적 계산
        
        # 실제 높이를 기준으로 픽셀-cm 비율 계산
        if height:
            px_to_cm_ratio = height / body_height_px
        else:
            px_to_cm_ratio = 170 / body_height_px  # 기본값 170cm
        
        # 실제 치수 계산
        measurements = {
            'height': height or 170,
            'shoulder_width': shoulder_width_px * px_to_cm_ratio,
            'chest': shoulder_width_px * px_to_cm_ratio * 1.1,  # 어깨보다 약간 큼
            'waist': hip_width_px * px_to_cm_ratio * 0.8,  # 힙보다 작음
            'hip': hip_width_px * px_to_cm_ratio,
        }
        
        return measurements
    
    def _default_measurements(self, height: int = None) -> Dict[str, float]:
        """기본 신체 치수"""
        height = height or 170
        
        # 평균적인 비율 적용
        return {
            'height': height,
            'shoulder_width': height * 0.25,
            'chest': height * 0.55,
            'waist': height * 0.45,
            'hip': height * 0.58,
        }
    
    def recommend_size(self, measurements: Dict[str, float], 
                      product_category: str, 
                      product_sizes: List[str],
                      fit_preference: str = 'regular') -> Dict[str, any]:
        """사이즈 추천"""
        
        if product_category not in self.SIZE_CHART:
            return {'recommended_size': 'M', 'confidence': 50}
        
        size_chart = self.SIZE_CHART[product_category]
        
        # 핏 선호도에 따른 조정
        fit_adjustments = {
            'slim': -2,
            'regular': 0,
            'relaxed': 2,
            'oversized': 4
        }
        
        adjustment = fit_adjustments.get(fit_preference, 0)
        
        # 각 사이즈별 적합도 계산
        size_scores = {}
        
        for size, size_specs in size_chart.items():
            if size not in product_sizes:
                continue
                
            score = 0
            criteria_count = 0
            
            # 카테고리별 주요 치수 비교
            if product_category == 'TOP':
                if 'chest' in measurements and 'chest' in size_specs:
                    chest_diff = abs(measurements['chest'] + adjustment - size_specs['chest'])
                    score += max(0, 100 - chest_diff * 5)
                    criteria_count += 1
                
                if 'shoulder_width' in measurements and 'shoulder' in size_specs:
                    shoulder_diff = abs(measurements['shoulder_width'] - size_specs['shoulder'])
                    score += max(0, 100 - shoulder_diff * 8)
                    criteria_count += 1
            
            elif product_category == 'BOTTOM':
                if 'waist' in measurements and 'waist' in size_specs:
                    waist_diff = abs(measurements['waist'] + adjustment - size_specs['waist'])
                    score += max(0, 100 - waist_diff * 6)
                    criteria_count += 1
                
                if 'hip' in measurements and 'hip' in size_specs:
                    hip_diff = abs(measurements['hip'] - size_specs['hip'])
                    score += max(0, 100 - hip_diff * 4)
                    criteria_count += 1
            
            if criteria_count > 0:
                size_scores[size] = score / criteria_count
        
        # 최적 사이즈 선택
        if size_scores:
            best_size = max(size_scores, key=size_scores.get)
            confidence = size_scores[best_size]
            
            # 대안 사이즈 제안
            sorted_sizes = sorted(size_scores.items(), key=lambda x: x[1], reverse=True)
            alternatives = [size for size, score in sorted_sizes[1:3]]
            
            return {
                'recommended_size': best_size,
                'confidence': min(confidence, 100),
                'alternatives': alternatives,
                'fit_notes': self._generate_fit_notes(best_size, measurements, size_chart, product_category)
            }
        
        # 기본값
        return {
            'recommended_size': 'M',
            'confidence': 50,
            'alternatives': ['S', 'L'],
            'fit_notes': ['표준 사이즈를 추천합니다']
        }
    
    def _generate_fit_notes(self, size: str, measurements: Dict, 
                          size_chart: Dict, category: str) -> List[str]:
        """핏 노트 생성"""
        
        notes = []
        size_specs = size_chart.get(size, {})
        
        if category == 'TOP':
            if 'chest' in measurements and 'chest' in size_specs:
                diff = measurements['chest'] - size_specs['chest']
                if diff > 3:
                    notes.append("가슴둘레가 조금 클 수 있습니다")
                elif diff < -3:
                    notes.append("가슴둘레가 여유 있게 맞을 것입니다")
        
        elif category == 'BOTTOM':
            if 'waist' in measurements and 'waist' in size_specs:
                diff = measurements['waist'] - size_specs['waist']
                if diff > 2:
                    notes.append("허리가 조금 타이트할 수 있습니다")
                elif diff < -2:
                    notes.append("허리가 여유 있게 맞을 것입니다")
        
        if not notes:
            notes.append("적절한 핏으로 예상됩니다")
        
        return notes
    
    def create_avatar(self, measurements: Dict[str, float], 
                     body_type: str = 'straight') -> Dict[str, Any]:
        """3D 아바타 생성"""
        
        # 체형에 따른 조정
        adjustments = self.BODY_TYPE_ADJUSTMENTS.get(body_type, 
                                                   self.BODY_TYPE_ADJUSTMENTS['straight'])
        
        # 조정된 치수
        adjusted_measurements = {
            'height': measurements['height'],
            'chest': measurements['chest'] + adjustments['chest_adj'],
            'waist': measurements['waist'] + adjustments['waist_adj'],
            'hip': measurements['hip'] + adjustments['hip_adj'],
            'shoulder_width': measurements.get('shoulder_width', measurements['chest'] * 0.9)
        }
        
        # 3D 모델 파라미터 생성
        avatar_params = {
            'body_type': body_type,
            'measurements': adjusted_measurements,
            'proportions': self._calculate_proportions(adjusted_measurements),
            'model_config': {
                'chest_scale': adjusted_measurements['chest'] / 90,
                'waist_scale': adjusted_measurements['waist'] / 70,
                'hip_scale': adjusted_measurements['hip'] / 95,
                'height_scale': adjusted_measurements['height'] / 170,
            }
        }
        
        return avatar_params
    
    def _calculate_proportions(self, measurements: Dict) -> Dict[str, float]:
        """신체 비율 계산"""
        
        height = measurements['height']
        
        return {
            'head_ratio': 7.5,  # 8등신 기준
            'torso_ratio': measurements['chest'] / height,
            'waist_ratio': measurements['waist'] / measurements['chest'],
            'hip_ratio': measurements['hip'] / measurements['waist'],
            'shoulder_ratio': measurements.get('shoulder_width', 0) / height
        }
    
    def virtual_try_on(self, avatar_params: Dict, 
                      clothing_item: Dict) -> Dict[str, Any]:
        """가상 착용"""
        
        item_category = clothing_item.get('category', 'TOP')
        item_size = clothing_item.get('size', 'M')
        
        # 사이즈 매칭 확인
        size_fit = self.recommend_size(
            avatar_params['measurements'],
            item_category,
            [item_size],
            clothing_item.get('fit', 'regular')
        )
        
        # 착용 시뮬레이션
        try_on_result = {
            'avatar_id': f"avatar_{hash(str(avatar_params)) % 10000}",
            'clothing_item': clothing_item,
            'fit_analysis': size_fit,
            'visual_params': {
                'clothing_scale': self._calculate_clothing_scale(
                    avatar_params, item_category, item_size
                ),
                'positioning': self._calculate_positioning(
                    avatar_params, item_category
                ),
                'material_properties': {
                    'fabric_type': clothing_item.get('material', 'cotton'),
                    'elasticity': 0.1,
                    'drape': 0.5
                }
            },
            'style_match': self._analyze_style_compatibility(avatar_params, clothing_item)
        }
        
        return try_on_result
    
    def _calculate_clothing_scale(self, avatar_params: Dict, 
                                category: str, size: str) -> Dict[str, float]:
        """의류 스케일 계산"""
        
        measurements = avatar_params['measurements']
        size_chart = self.SIZE_CHART.get(category, {})
        size_specs = size_chart.get(size, {})
        
        if category == 'TOP' and size_specs:
            return {
                'width_scale': size_specs.get('chest', 90) / measurements['chest'],
                'length_scale': size_specs.get('length', 65) / (measurements['height'] * 0.38),
                'shoulder_scale': size_specs.get('shoulder', 42) / measurements.get('shoulder_width', 42)
            }
        
        elif category == 'BOTTOM' and size_specs:
            return {
                'waist_scale': size_specs.get('waist', 70) / measurements['waist'],
                'hip_scale': size_specs.get('hip', 95) / measurements['hip'],
                'length_scale': size_specs.get('length', 100) / (measurements['height'] * 0.6)
            }
        
        # 기본값
        return {'width_scale': 1.0, 'length_scale': 1.0, 'shoulder_scale': 1.0}
    
    def _calculate_positioning(self, avatar_params: Dict, category: str) -> Dict[str, float]:
        """의류 위치 계산"""
        
        height = avatar_params['measurements']['height']
        
        if category == 'TOP':
            return {
                'y_offset': height * 0.15,  # 목 위치
                'x_center': 0,
                'rotation': 0
            }
        elif category == 'BOTTOM':
            return {
                'y_offset': height * 0.45,  # 허리 위치
                'x_center': 0,
                'rotation': 0
            }
        
        return {'y_offset': 0, 'x_center': 0, 'rotation': 0}
    
    def _analyze_style_compatibility(self, avatar_params: Dict, 
                                   clothing_item: Dict) -> Dict[str, Any]:
        """스타일 호환성 분석"""
        
        body_type = avatar_params['body_type']
        item_style = clothing_item.get('style', 'casual')
        
        # 체형별 추천 스타일
        body_type_styles = {
            'straight': ['minimal', 'casual', 'business'],
            'pear': ['casual', 'business', 'elegant'],
            'apple': ['casual', 'sporty', 'relaxed'],
            'hourglass': ['elegant', 'vintage', 'fitted'],
            'inverted_triangle': ['casual', 'sporty', 'relaxed']
        }
        
        recommended_styles = body_type_styles.get(body_type, ['casual'])
        is_compatible = item_style in recommended_styles
        
        return {
            'compatibility_score': 85 if is_compatible else 60,
            'is_recommended': is_compatible,
            'style_notes': [
                f"{body_type} 체형에 {'적합한' if is_compatible else '보통인'} 스타일입니다"
            ]
        }


class MockPoseDetector:
    """포즈 검출기 Mock 클래스"""
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Mock 포즈 검출"""
        # 실제 구현에서는 MediaPipe 등을 사용
        height, width = image.shape[:2]
        
        # 가상의 랜드마크 (정규화된 좌표)
        return [
            {'x': 0.3, 'y': 0.2},   # 11: 왼쪽 어깨
            {'x': 0.7, 'y': 0.2},   # 12: 오른쪽 어깨
            {'x': 0.35, 'y': 0.6},  # 23: 왼쪽 힙
            {'x': 0.65, 'y': 0.6},  # 24: 오른쪽 힙
        ]