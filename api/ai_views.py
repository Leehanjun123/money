from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser
from django.core.cache import cache
from django.db import models
from django.utils import timezone
import tempfile
import os
from ai_service.style_matcher import StyleMatcher
from wardrobe.models import ClothingItem, Outfit
from crawler.models import Product
import logging

logger = logging.getLogger(__name__)

class AIStyleAnalysisView(APIView):
    """AI 스타일 분석 API"""
    
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.style_matcher = StyleMatcher()
    
    def post(self, request):
        """이미지를 분석하여 스타일 매칭"""
        
        if 'image' not in request.FILES:
            return Response(
                {'error': '이미지를 업로드해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES['image']
        
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                for chunk in image_file.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # 이미지 분석
            features = self.style_matcher.analyze_image(tmp_path)
            
            # 스타일 매칭
            style_matches = self.style_matcher.match_style(features)
            
            # 임시 파일 삭제
            os.unlink(tmp_path)
            
            # 결과 캐싱 (5분)
            cache_key = f"style_analysis:{request.user.id}:{image_file.name}"
            result = {
                'features': {
                    'colors': features['color_features'],
                    'pattern_complexity': features['pattern_complexity'],
                    'silhouette': features['silhouette']
                },
                'style_matches': style_matches,
                'recommendations': self._get_style_recommendations(style_matches[0]['style_id'])
            }
            
            cache.set(cache_key, result, 300)
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Style analysis failed: {e}")
            return Response(
                {'error': '스타일 분석 중 오류가 발생했습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_style_recommendations(self, style_id: str):
        """스타일에 맞는 상품 추천"""
        
        # 스타일에 맞는 키워드
        style_data = StyleMatcher.STYLE_CATEGORIES.get(style_id, {})
        keywords = style_data.get('keywords', [])
        brands = style_data.get('brands', [])
        
        # 관련 상품 검색
        products = Product.objects.filter(
            is_available=True
        ).filter(
            # 브랜드 또는 태그에서 키워드 매칭
            models.Q(brand__name__in=brands) |
            models.Q(style_tags__overlap=keywords)
        )[:10]
        
        return [{
            'product_id': p.id,
            'name': p.name,
            'brand': p.brand.name,
            'price': float(p.final_price),
            'image': p.main_image_url,
            'match_score': 85  # 실제로는 더 정교한 점수 계산
        } for p in products]

class AIOutfitRecommendationView(APIView):
    """AI 코디 추천 API"""
    
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.style_matcher = StyleMatcher()
    
    def get(self, request):
        """사용자의 옷장 기반 코디 추천"""
        
        # 사용자 옷장 아이템
        user_items = ClothingItem.objects.filter(
            user=request.user
        ).values('id', 'name', 'category', 'color', 'brand', 'tags')
        
        if not user_items:
            return Response({
                'message': '옷장에 아이템을 먼저 등록해주세요',
                'recommendations': []
            })
        
        # 타겟 스타일 (쿼리 파라미터 또는 자동 추론)
        target_style = request.query_params.get('style')
        
        # 코디 추천
        recommendations = self.style_matcher.recommend_coordination(
            list(user_items),
            target_style
        )
        
        # 추천 코디 생성
        outfits = self._generate_outfit_combinations(user_items, recommendations)
        
        return Response({
            'target_style': target_style or 'auto',
            'recommendations': recommendations,
            'suggested_outfits': outfits
        })
    
    def _generate_outfit_combinations(self, user_items, recommendations):
        """코디 조합 생성"""
        
        outfits = []
        
        # 카테고리별 아이템 그룹화
        items_by_category = {}
        for item in user_items:
            category = item['category']
            if category not in items_by_category:
                items_by_category[category] = []
            items_by_category[category].append(item)
        
        # 기본 코디 조합 (상의 + 하의 + 신발)
        tops = items_by_category.get('TOP', [])
        bottoms = items_by_category.get('BOTTOM', [])
        shoes = items_by_category.get('SHOES', [])
        
        # 최대 3개 조합 생성
        for i in range(min(3, len(tops))):
            if i < len(bottoms) and i < len(shoes):
                outfit = {
                    'items': [
                        tops[i],
                        bottoms[i % len(bottoms)],
                        shoes[i % len(shoes)]
                    ],
                    'style': 'casual',  # 실제로는 AI가 판단
                    'score': 85 - (i * 5)  # 실제로는 AI가 계산
                }
                outfits.append(outfit)
        
        return outfits

class AIOutfitScoringView(APIView):
    """코디 점수 평가 API"""
    
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.style_matcher = StyleMatcher()
    
    def post(self, request):
        """선택한 아이템들의 코디 점수 계산"""
        
        item_ids = request.data.get('item_ids', [])
        
        if not item_ids:
            return Response(
                {'error': '평가할 아이템을 선택해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 아이템 조회
        items = ClothingItem.objects.filter(
            id__in=item_ids,
            user=request.user
        ).values('id', 'name', 'category', 'color', 'material', 'tags')
        
        if not items:
            return Response(
                {'error': '유효한 아이템이 없습니다'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # 코디 점수 계산
        score_result = self.style_matcher.calculate_outfit_score(list(items))
        
        # 개선 제안
        suggestions = self._generate_improvement_suggestions(score_result, items)
        
        return Response({
            'outfit_score': score_result,
            'improvement_suggestions': suggestions,
            'item_count': len(items)
        })
    
    def _generate_improvement_suggestions(self, score_result, items):
        """코디 개선 제안"""
        
        suggestions = []
        
        if score_result['color_score'] < 70:
            suggestions.append({
                'type': 'color',
                'message': '색상 조합을 개선해보세요',
                'recommendation': '무채색 아이템을 추가하거나 색상 수를 줄여보세요'
            })
        
        if score_result['style_score'] < 70:
            suggestions.append({
                'type': 'style',
                'message': '스타일 일관성을 높여보세요',
                'recommendation': '같은 스타일 카테고리의 아이템으로 통일해보세요'
            })
        
        # 부족한 카테고리 체크
        categories = [item['category'] for item in items]
        if 'TOP' not in categories:
            suggestions.append({
                'type': 'missing',
                'message': '상의가 필요합니다',
                'recommendation': '상의를 추가해주세요'
            })
        
        if 'BOTTOM' not in categories and 'DRESS' not in categories:
            suggestions.append({
                'type': 'missing',
                'message': '하의가 필요합니다',
                'recommendation': '바지나 스커트를 추가해주세요'
            })
        
        return suggestions

class AITrendAnalysisView(APIView):
    """AI 트렌드 분석 API"""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        """현재 패션 트렌드 분석"""
        
        # 캐시 확인
        cache_key = 'fashion_trends'
        cached = cache.get(cache_key)
        
        if cached:
            return Response(cached)
        
        # 인기 상품 기반 트렌드 분석
        trending_products = Product.objects.filter(
            is_available=True
        ).order_by('-like_count', '-view_count')[:100]
        
        # 트렌드 추출
        trending_colors = self._extract_trending_colors(trending_products)
        trending_styles = self._extract_trending_styles(trending_products)
        trending_brands = self._extract_trending_brands(trending_products)
        
        result = {
            'season': self._get_current_season(),
            'trending_colors': trending_colors,
            'trending_styles': trending_styles,
            'trending_brands': trending_brands,
            'trend_keywords': [
                '오버사이즈', '미니멀', '스트릿', '빈티지', 'Y2K'
            ],
            'updated_at': timezone.now().isoformat()
        }
        
        # 캐시 저장 (1시간)
        cache.set(cache_key, result, 3600)
        
        return Response(result)
    
    def _extract_trending_colors(self, products):
        """트렌딩 색상 추출"""
        color_counts = {}
        
        for product in products:
            colors = product.colors or []
            for color in colors:
                color_counts[color] = color_counts.get(color, 0) + 1
        
        # 상위 5개 색상
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{
            'color': color,
            'count': count,
            'percentage': count / len(products) * 100
        } for color, count in sorted_colors[:5]]
    
    def _extract_trending_styles(self, products):
        """트렌딩 스타일 추출"""
        style_counts = {}
        
        for product in products:
            styles = product.style_tags or []
            for style in styles:
                style_counts[style] = style_counts.get(style, 0) + 1
        
        sorted_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{
            'style': style,
            'count': count,
            'percentage': count / len(products) * 100
        } for style, count in sorted_styles[:5]]
    
    def _extract_trending_brands(self, products):
        """트렌딩 브랜드 추출"""
        brand_counts = {}
        
        for product in products:
            brand = product.brand.name
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{
            'brand': brand,
            'count': count,
            'product_count': count
        } for brand, count in sorted_brands[:10]]
    
    def _get_current_season(self):
        """현재 시즌"""
        import datetime
        month = datetime.datetime.now().month
        
        if 3 <= month <= 5:
            return '2024 S/S'
        elif 6 <= month <= 8:
            return '2024 SUMMER'
        elif 9 <= month <= 11:
            return '2024 F/W'
        else:
            return '2024 WINTER'