from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.core.cache import cache
from django.db import models
from crawler.elasticsearch_service import elasticsearch_service
from crawler.models import Product, Brand
import logging

logger = logging.getLogger(__name__)

class ProductSearchView(APIView):
    """ElasticSearch 기반 상품 검색 API"""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        """상품 검색"""
        
        # 검색 파라미터 추출
        query_params = {
            'keyword': request.query_params.get('q', ''),
            'brand': request.query_params.get('brand'),
            'category': request.query_params.get('category'),
            'colors': request.query_params.getlist('color'),
            'min_price': self._safe_int(request.query_params.get('min_price')),
            'max_price': self._safe_int(request.query_params.get('max_price')),
            'sort_by': request.query_params.get('sort', 'relevance'),
            'offset': self._safe_int(request.query_params.get('offset', 0)),
            'limit': min(self._safe_int(request.query_params.get('limit', 20)), 100),
            'is_available': True
        }
        
        # 캐시 키 생성
        cache_key = f"search:{hash(str(sorted(query_params.items())))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return Response(cached_result)
        
        # ElasticSearch가 사용 가능한지 확인
        if not elasticsearch_service.check_connection():
            return self._fallback_search(query_params)
        
        try:
            # ElasticSearch 검색 실행
            search_result = elasticsearch_service.search_products(query_params)
            
            # 결과 포맷팅
            result = {
                'products': search_result['products'],
                'pagination': {
                    'total': search_result['total'],
                    'offset': query_params['offset'],
                    'limit': query_params['limit'],
                    'has_next': query_params['offset'] + query_params['limit'] < search_result['total']
                },
                'filters': self._format_aggregations(search_result.get('aggregations', {})),
                'search_info': {
                    'query': query_params['keyword'],
                    'took': search_result.get('took', 0),
                    'engine': 'elasticsearch'
                }
            }
            
            # 캐시 저장 (5분)
            cache.set(cache_key, result, 300)
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"ElasticSearch error: {e}")
            return self._fallback_search(query_params)
    
    def _safe_int(self, value, default=None):
        """안전한 정수 변환"""
        try:
            return int(value) if value else default
        except (ValueError, TypeError):
            return default
    
    def _fallback_search(self, query_params):
        """ElasticSearch 실패 시 Django ORM 대체 검색"""
        try:
            queryset = Product.objects.select_related('brand').filter(is_available=True)
            
            # 키워드 검색
            if query_params['keyword']:
                queryset = queryset.filter(
                    models.Q(name__icontains=query_params['keyword']) |
                    models.Q(brand__name__icontains=query_params['keyword']) |
                    models.Q(description__icontains=query_params['keyword'])
                )
            
            # 필터 적용
            if query_params['brand']:
                queryset = queryset.filter(brand__name=query_params['brand'])
            
            if query_params['category']:
                queryset = queryset.filter(medium_category=query_params['category'])
            
            if query_params['colors']:
                queryset = queryset.filter(colors__overlap=query_params['colors'])
            
            if query_params['min_price']:
                queryset = queryset.filter(final_price__gte=query_params['min_price'])
            
            if query_params['max_price']:
                queryset = queryset.filter(final_price__lte=query_params['max_price'])
            
            # 정렬
            sort_mapping = {
                'price_low': 'final_price',
                'price_high': '-final_price',
                'popularity': '-like_count',
                'rating': '-rating',
                'latest': '-crawled_at'
            }
            
            if query_params['sort_by'] in sort_mapping:
                queryset = queryset.order_by(sort_mapping[query_params['sort_by']])
            
            # 페이지네이션
            total = queryset.count()
            offset = query_params['offset']
            limit = query_params['limit']
            
            products = list(queryset[offset:offset + limit].values(
                'id', 'name', 'brand__name', 'final_price', 'original_price',
                'main_image_url', 'rating', 'review_count', 'like_count',
                'large_category', 'medium_category', 'small_category',
                'colors', 'sizes', 'url'
            ))
            
            # 결과 포맷팅
            for product in products:
                product['brand'] = {
                    'name': product.pop('brand__name', '')
                }
                product['price'] = {
                    'final': product.pop('final_price', 0),
                    'original': product.pop('original_price', 0)
                }
                product['category'] = {
                    'large': product.pop('large_category', ''),
                    'medium': product.pop('medium_category', ''),
                    'small': product.pop('small_category', '')
                }
                product['images'] = {
                    'main': product.pop('main_image_url', '')
                }
                product['rating'] = {
                    'average': product.pop('rating', 0),
                    'count': product.pop('review_count', 0)
                }
                product['popularity'] = {
                    'like_count': product.pop('like_count', 0)
                }
            
            result = {
                'products': products,
                'pagination': {
                    'total': total,
                    'offset': offset,
                    'limit': limit,
                    'has_next': offset + limit < total
                },
                'filters': {},
                'search_info': {
                    'query': query_params['keyword'],
                    'took': 0,
                    'engine': 'django_orm'
                }
            }
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return Response(
                {'error': '검색 중 오류가 발생했습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _format_aggregations(self, aggregations):
        """집계 결과 포맷팅"""
        filters = {}
        
        if 'brands' in aggregations:
            filters['brands'] = [
                {'name': bucket['key'], 'count': bucket['doc_count']}
                for bucket in aggregations['brands']['buckets']
            ]
        
        if 'categories' in aggregations:
            filters['categories'] = [
                {'name': bucket['key'], 'count': bucket['doc_count']}
                for bucket in aggregations['categories']['buckets']
            ]
        
        if 'colors' in aggregations:
            filters['colors'] = [
                {'name': bucket['key'], 'count': bucket['doc_count']}
                for bucket in aggregations['colors']['buckets']
            ]
        
        if 'price_ranges' in aggregations:
            filters['price_ranges'] = [
                {
                    'range': f"{bucket.get('from', 0)}-{bucket.get('to', '∞')}",
                    'count': bucket['doc_count']
                }
                for bucket in aggregations['price_ranges']['buckets']
            ]
        
        return filters

class ProductRecommendationView(APIView):
    """상품 추천 API"""
    
    permission_classes = [AllowAny]
    
    def get(self, request, product_id):
        """유사 상품 추천"""
        
        cache_key = f"recommendations:{product_id}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return Response(cached_result)
        
        try:
            # ElasticSearch 추천
            if elasticsearch_service.check_connection():
                recommendations = elasticsearch_service.get_product_recommendations(
                    product_id, count=12
                )
            else:
                recommendations = self._fallback_recommendations(product_id)
            
            result = {
                'product_id': product_id,
                'recommendations': recommendations,
                'total': len(recommendations)
            }
            
            # 캐시 저장 (30분)
            cache.set(cache_key, result, 1800)
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return Response(
                {'error': '추천 상품을 불러올 수 없습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _fallback_recommendations(self, product_id):
        """Django ORM 기반 추천"""
        try:
            base_product = Product.objects.select_related('brand').get(
                id=product_id, is_available=True
            )
            
            # 같은 카테고리, 비슷한 가격대 상품 추천
            recommendations = Product.objects.select_related('brand').filter(
                medium_category=base_product.medium_category,
                final_price__gte=base_product.final_price * 0.7,
                final_price__lte=base_product.final_price * 1.5,
                is_available=True
            ).exclude(
                id=product_id
            ).order_by('-like_count', '-rating')[:12]
            
            return [
                {
                    'id': p.id,
                    'name': p.name,
                    'brand': {'name': p.brand.name if p.brand else ''},
                    'price': {
                        'final': int(p.final_price) if p.final_price else 0,
                        'original': int(p.original_price) if p.original_price else 0
                    },
                    'images': {'main': p.main_image_url},
                    'rating': {
                        'average': float(p.rating) if p.rating else 0,
                        'count': p.review_count or 0
                    },
                    'similarity_score': 0.8  # 기본값
                }
                for p in recommendations
            ]
        except Product.DoesNotExist:
            return []

class TrendingProductsView(APIView):
    """인기 상품 API"""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        """인기 상품 조회"""
        
        category = request.query_params.get('category')
        limit = min(int(request.query_params.get('limit', 20)), 50)
        
        cache_key = f"trending:{category}:{limit}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return Response(cached_result)
        
        try:
            # ElasticSearch 조회
            if elasticsearch_service.check_connection():
                trending = elasticsearch_service.get_trending_products(
                    category=category, limit=limit
                )
            else:
                trending = self._fallback_trending(category, limit)
            
            result = {
                'trending_products': trending,
                'category': category,
                'total': len(trending)
            }
            
            # 캐시 저장 (15분)
            cache.set(cache_key, result, 900)
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Trending products error: {e}")
            return Response(
                {'error': '인기 상품을 불러올 수 없습니다'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _fallback_trending(self, category, limit):
        """Django ORM 기반 인기 상품"""
        queryset = Product.objects.select_related('brand').filter(is_available=True)
        
        if category:
            queryset = queryset.filter(medium_category=category)
        
        trending = queryset.order_by(
            '-like_count', '-view_count', '-rating'
        )[:limit]
        
        return [
            {
                'id': p.id,
                'name': p.name,
                'brand': {'name': p.brand.name if p.brand else ''},
                'price': {
                    'final': int(p.final_price) if p.final_price else 0,
                    'original': int(p.original_price) if p.original_price else 0
                },
                'images': {'main': p.main_image_url},
                'rating': {
                    'average': float(p.rating) if p.rating else 0,
                    'count': p.review_count or 0
                },
                'popularity': {
                    'like_count': p.like_count or 0,
                    'view_count': p.view_count or 0
                }
            }
            for p in trending
        ]

class SearchSuggestionsView(APIView):
    """검색 자동완성 API"""
    
    permission_classes = [AllowAny]
    
    def get(self, request):
        """검색 자동완성"""
        
        query = request.query_params.get('q', '').strip()
        
        if len(query) < 2:
            return Response({'suggestions': []})
        
        cache_key = f"suggestions:{query}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return Response(cached_result)
        
        suggestions = []
        
        try:
            # 브랜드 제안
            brands = Brand.objects.filter(
                name__icontains=query
            ).values_list('name', flat=True)[:5]
            
            for brand in brands:
                suggestions.append({
                    'text': brand,
                    'type': 'brand'
                })
            
            # 상품명 제안
            products = Product.objects.filter(
                name__icontains=query,
                is_available=True
            ).values_list('name', flat=True)[:5]
            
            for product in products:
                suggestions.append({
                    'text': product,
                    'type': 'product'
                })
            
            # 카테고리 제안
            categories = ['상의', '하의', '신발', '가방', '액세서리']
            for cat in categories:
                if query in cat:
                    suggestions.append({
                        'text': cat,
                        'type': 'category'
                    })
            
            result = {'suggestions': suggestions[:10]}
            
            # 캐시 저장 (10분)
            cache.set(cache_key, result, 600)
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Search suggestions error: {e}")
            return Response({'suggestions': []})