from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser
from django.core.cache import cache
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from ai_service.virtual_fitting import VirtualFittingEngine
from wardrobe.models import ClothingItem
import logging

logger = logging.getLogger(__name__)

class ARBodyAnalysisView(APIView):
    """AR 신체 분석 API"""
    
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitting_engine = VirtualFittingEngine()
    
    def post(self, request):
        """사용자 신체 사진을 분석하여 치수 측정"""
        
        if 'photo' not in request.FILES:
            return Response(
                {'error': '신체 사진을 업로드해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        photo_file = request.FILES['photo']
        height = request.data.get('height')  # 사용자 입력 키 (cm)
        
        try:
            # 이미지를 numpy 배열로 변환
            image = Image.open(photo_file)
            image_array = np.array(image)
            
            # BGR로 변환 (OpenCV 형식)
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # 신체 치수 분석
            measurements = self.fitting_engine.analyze_body_measurements(
                image_array, int(height) if height else None
            )
            
            # 체형 추정 (간단한 비율 기반)
            body_type = self._estimate_body_type(measurements)
            
            # 아바타 생성
            avatar_params = self.fitting_engine.create_avatar(measurements, body_type)
            
            # 결과 캐싱 (30분)
            cache_key = f"body_analysis:{request.user.id}"
            result = {
                'measurements': measurements,
                'body_type': body_type,
                'avatar_params': avatar_params,
                'recommendations': self._get_body_type_recommendations(body_type)
            }
            
            cache.set(cache_key, result, 1800)
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Body analysis failed: {e}")
            return Response(
                {'error': '신체 분석 중 오류가 발생했습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _estimate_body_type(self, measurements: Dict[str, float]) -> str:
        """측정값으로 체형 추정"""
        
        chest = measurements.get('chest', 90)
        waist = measurements.get('waist', 70)
        hip = measurements.get('hip', 95)
        
        # 비율 계산
        waist_to_chest = waist / chest
        hip_to_waist = hip / waist
        chest_to_hip = chest / hip
        
        # 체형 판단
        if abs(chest - hip) <= 5 and waist_to_chest < 0.8:
            return 'hourglass'  # 모래시계형
        elif hip > chest * 1.05:
            return 'pear'  # 배형
        elif chest > hip * 1.05:
            return 'inverted_triangle'  # 역삼각형
        elif waist_to_chest > 0.9:
            return 'apple'  # 사과형
        else:
            return 'straight'  # 직선형
    
    def _get_body_type_recommendations(self, body_type: str) -> Dict[str, List[str]]:
        """체형별 스타일 추천"""
        
        recommendations = {
            'straight': {
                'flattering_styles': ['A라인 스커트', '허리 벨트', '레이어드 룩'],
                'avoid_styles': ['박시 핏', '스트레이트 라인'],
                'best_colors': ['밝은 색상', '패턴'],
                'tips': ['허리 라인을 강조하세요', '볼륨감 있는 아이템으로 곡선을 만드세요']
            },
            'pear': {
                'flattering_styles': ['A라인 상의', '밝은 색 상의', '다크 톤 하의'],
                'avoid_styles': ['타이트한 하의', '히프 강조 아이템'],
                'best_colors': ['상의는 밝게', '하의는 어둡게'],
                'tips': ['상체에 포인트를 주세요', '어깨 라인을 강조하세요']
            },
            'apple': {
                'flattering_styles': ['V넥', '엠파이어 라인', '긴 카디건'],
                'avoid_styles': ['크롭 탑', '허리 벨트'],
                'best_colors': ['세로 스트라이프', '다크 톤'],
                'tips': ['세로 라인을 강조하세요', '허리보다 가슴 아래를 강조하세요']
            },
            'hourglass': {
                'flattering_styles': ['허리 강조', '핏앤플레어', 'wrap 드레스'],
                'avoid_styles': ['박시 핏', '직선 라인'],
                'best_colors': ['모든 색상', '대비 색상'],
                'tips': ['자연스러운 곡선을 살리세요', '허리를 강조하세요']
            },
            'inverted_triangle': {
                'flattering_styles': ['A라인 하의', '볼륨 스커트', '와이드 팬츠'],
                'avoid_styles': ['어깨 패드', '수평 스트라이프 상의'],
                'best_colors': ['하의에 밝은 색', '상의에 어두운 색'],
                'tips': ['하체에 볼륨을 주세요', '어깨 라인을 부드럽게 하세요']
            }
        }
        
        return recommendations.get(body_type, recommendations['straight'])

class ARSizeRecommendationView(APIView):
    """AR 사이즈 추천 API"""
    
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitting_engine = VirtualFittingEngine()
    
    def post(self, request):
        """특정 상품의 사이즈 추천"""
        
        product_id = request.data.get('product_id')
        item_id = request.data.get('item_id')  # 옷장 아이템 ID
        fit_preference = request.data.get('fit_preference', 'regular')
        
        if not product_id and not item_id:
            return Response(
                {'error': '상품 ID 또는 아이템 ID를 제공해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 사용자 신체 측정 데이터 가져오기
        cache_key = f"body_analysis:{request.user.id}"
        body_data = cache.get(cache_key)
        
        if not body_data:
            return Response(
                {'error': '먼저 신체 측정을 완료해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        measurements = body_data['measurements']
        
        try:
            # 상품 정보 가져오기
            if item_id:
                item = ClothingItem.objects.get(id=item_id, user=request.user)
                category = item.category
                available_sizes = ['XS', 'S', 'M', 'L', 'XL']  # 기본값
            else:
                from crawler.models import Product
                product = Product.objects.get(id=product_id)
                category = product.category
                available_sizes = product.sizes or ['S', 'M', 'L']
            
            # 사이즈 추천
            size_recommendation = self.fitting_engine.recommend_size(
                measurements, category, available_sizes, fit_preference
            )
            
            # 추가 정보
            result = {
                'size_recommendation': size_recommendation,
                'measurements_used': measurements,
                'fit_preference': fit_preference,
                'size_chart': VirtualFittingEngine.SIZE_CHART.get(category, {}),
                'fit_tips': self._get_fit_tips(category, size_recommendation['recommended_size'])
            }
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Size recommendation failed: {e}")
            return Response(
                {'error': '사이즈 추천 중 오류가 발생했습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_fit_tips(self, category: str, size: str) -> List[str]:
        """핏 팁 제공"""
        
        tips = {
            'TOP': [
                f"{size} 사이즈는 표준 핏입니다",
                "어깨선이 맞는지 확인하세요",
                "암홀이 너무 크지 않은지 체크하세요"
            ],
            'BOTTOM': [
                f"{size} 사이즈는 표준 핏입니다", 
                "허리 둘레가 편안한지 확인하세요",
                "앉았을 때 불편하지 않은지 체크하세요"
            ],
            'OUTER': [
                f"{size} 사이즈는 표준 핏입니다",
                "내부 레이어를 고려하여 선택하세요",
                "소매 길이가 적절한지 확인하세요"
            ]
        }
        
        return tips.get(category, ["표준 핏으로 선택하세요"])

class ARVirtualTryOnView(APIView):
    """AR 가상 착용 API"""
    
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitting_engine = VirtualFittingEngine()
    
    def post(self, request):
        """가상 착용 시뮬레이션"""
        
        item_ids = request.data.get('item_ids', [])
        
        if not item_ids:
            return Response(
                {'error': '착용할 아이템을 선택해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 사용자 신체 데이터
        cache_key = f"body_analysis:{request.user.id}"
        body_data = cache.get(cache_key)
        
        if not body_data:
            return Response(
                {'error': '먼저 신체 측정을 완료해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        avatar_params = body_data['avatar_params']
        
        try:
            # 선택된 아이템들
            items = ClothingItem.objects.filter(
                id__in=item_ids,
                user=request.user
            )
            
            if not items:
                return Response(
                    {'error': '유효한 아이템이 없습니다'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # 각 아이템에 대한 가상 착용 시뮬레이션
            try_on_results = []
            
            for item in items:
                clothing_data = {
                    'id': item.id,
                    'name': item.name,
                    'category': item.category,
                    'color': item.color,
                    'material': item.material,
                    'size': item.size,
                    'image_url': item.image.url if item.image else None
                }
                
                try_on_result = self.fitting_engine.virtual_try_on(
                    avatar_params, clothing_data
                )
                
                try_on_results.append(try_on_result)
            
            # 전체 코디 분석
            from ai_service.style_matcher import StyleMatcher
            style_matcher = StyleMatcher()
            
            outfit_score = style_matcher.calculate_outfit_score([
                {
                    'name': item.name,
                    'category': item.category,
                    'color': item.color,
                    'material': item.material
                } for item in items
            ])
            
            result = {
                'avatar_id': avatar_params.get('avatar_id', 'default'),
                'try_on_results': try_on_results,
                'outfit_analysis': outfit_score,
                'styling_suggestions': self._get_styling_suggestions(items, outfit_score)
            }
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Virtual try-on failed: {e}")
            return Response(
                {'error': '가상 착용 중 오류가 발생했습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_styling_suggestions(self, items, outfit_score):
        """스타일링 제안"""
        
        suggestions = []
        
        # 카테고리 확인
        categories = [item.category for item in items]
        
        if 'TOP' not in categories:
            suggestions.append({
                'type': 'missing_item',
                'message': '상의를 추가해보세요',
                'recommendation': '현재 선택한 아이템과 잘 어울리는 상의를 추천합니다'
            })
        
        if 'BOTTOM' not in categories and 'DRESS' not in categories:
            suggestions.append({
                'type': 'missing_item', 
                'message': '하의를 추가해보세요',
                'recommendation': '바지나 스커트를 추가하면 완성도가 높아집니다'
            })
        
        if 'SHOES' not in categories:
            suggestions.append({
                'type': 'accessory',
                'message': '신발을 추가해보세요',
                'recommendation': '전체 룩을 완성하는 신발을 선택해보세요'
            })
        
        # 점수 기반 제안
        if outfit_score.get('color_score', 0) < 70:
            suggestions.append({
                'type': 'color_harmony',
                'message': '색상 조합을 개선해보세요',
                'recommendation': '비슷한 톤의 색상이나 무채색을 활용해보세요'
            })
        
        if outfit_score.get('style_score', 0) < 70:
            suggestions.append({
                'type': 'style_consistency',
                'message': '스타일 통일성을 높여보세요', 
                'recommendation': '하나의 스타일로 통일하거나 믹스매치를 시도해보세요'
            })
        
        return suggestions

class AROutfitComparisonView(APIView):
    """AR 코디 비교 API"""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """여러 코디 옵션 비교"""
        
        outfit_options = request.data.get('outfit_options', [])
        
        if len(outfit_options) < 2:
            return Response(
                {'error': '비교할 코디 옵션을 2개 이상 제공해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            comparisons = []
            
            # 각 코디 옵션에 대한 분석
            for idx, option in enumerate(outfit_options):
                item_ids = option.get('item_ids', [])
                occasion = option.get('occasion', 'daily')
                
                items = ClothingItem.objects.filter(
                    id__in=item_ids,
                    user=request.user
                )
                
                # 코디 점수 계산
                from ai_service.style_matcher import StyleMatcher
                style_matcher = StyleMatcher()
                
                outfit_score = style_matcher.calculate_outfit_score([
                    {
                        'name': item.name,
                        'category': item.category,
                        'color': item.color,
                        'material': item.material
                    } for item in items
                ])
                
                # 상황별 적합도 점수
                occasion_score = self._calculate_occasion_fitness(items, occasion)
                
                comparisons.append({
                    'option_id': idx + 1,
                    'item_count': len(items),
                    'outfit_score': outfit_score,
                    'occasion_fitness': occasion_score,
                    'total_score': (outfit_score.get('total_score', 0) + occasion_score) / 2,
                    'pros': self._get_outfit_pros(outfit_score),
                    'cons': self._get_outfit_cons(outfit_score)
                })
            
            # 최고 점수 코디 선택
            best_option = max(comparisons, key=lambda x: x['total_score'])
            
            return Response({
                'comparisons': comparisons,
                'recommendation': best_option,
                'summary': f"옵션 {best_option['option_id']}번이 가장 적합합니다"
            })
            
        except Exception as e:
            logger.error(f"Outfit comparison failed: {e}")
            return Response(
                {'error': '코디 비교 중 오류가 발생했습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _calculate_occasion_fitness(self, items, occasion: str) -> float:
        """상황별 적합도 계산"""
        
        occasion_requirements = {
            'business': {'formal_score': 0.8, 'color_limit': ['navy', 'black', 'gray', 'white']},
            'casual': {'formal_score': 0.3, 'color_limit': []},
            'party': {'formal_score': 0.6, 'color_limit': []},
            'date': {'formal_score': 0.5, 'color_limit': []},
            'workout': {'formal_score': 0.1, 'material_req': ['polyester', 'spandex']},
        }
        
        requirements = occasion_requirements.get(occasion, occasion_requirements['casual'])
        
        # 기본 점수
        score = 70.0
        
        # 아이템별 적합도 체크
        for item in items:
            # 소재 체크
            if 'material_req' in requirements:
                if any(mat in item.material.lower() for mat in requirements['material_req']):
                    score += 10
                else:
                    score -= 15
            
            # 색상 체크
            if requirements.get('color_limit'):
                if item.color.lower() in requirements['color_limit']:
                    score += 5
                else:
                    score -= 10
        
        return max(0, min(100, score))
    
    def _get_outfit_pros(self, outfit_score: Dict) -> List[str]:
        """코디 장점"""
        pros = []
        
        if outfit_score.get('color_score', 0) >= 80:
            pros.append('색상 조화가 훌륭합니다')
        
        if outfit_score.get('style_score', 0) >= 80:
            pros.append('스타일이 일관성 있습니다')
        
        if outfit_score.get('season_score', 0) >= 80:
            pros.append('계절에 적합합니다')
        
        if not pros:
            pros.append('기본적인 조합입니다')
        
        return pros
    
    def _get_outfit_cons(self, outfit_score: Dict) -> List[str]:
        """코디 단점"""
        cons = []
        
        if outfit_score.get('color_score', 0) < 60:
            cons.append('색상 조합을 개선할 수 있습니다')
        
        if outfit_score.get('style_score', 0) < 60:
            cons.append('스타일 통일성이 부족합니다')
        
        if outfit_score.get('season_score', 0) < 60:
            cons.append('계절에 맞지 않는 아이템이 있습니다')
        
        return cons