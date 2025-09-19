from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.db.models import Q, Count, Avg
from django.core.cache import cache
from .models import Product, Brand
from .serializers import (
    ProductListSerializer, 
    ProductDetailSerializer,
    BrandSerializer,
    ProductSearchSerializer
)
from .tasks import crawl_musinsa_products

class ProductViewSet(viewsets.ReadOnlyModelViewSet):
    """상품 조회 API"""
    
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return ProductDetailSerializer
        return ProductListSerializer
    
    def get_queryset(self):
        queryset = Product.objects.filter(is_available=True)
        
        # 필터링 파라미터
        category = self.request.query_params.get('category')
        gender = self.request.query_params.get('gender')
        min_price = self.request.query_params.get('min_price')
        max_price = self.request.query_params.get('max_price')
        brand_ids = self.request.query_params.getlist('brand')
        
        if category:
            queryset = queryset.filter(category=category)
        
        if gender:
            queryset = queryset.filter(gender=gender)
        
        if min_price:
            queryset = queryset.filter(
                Q(sale_price__gte=min_price) | 
                Q(sale_price__isnull=True, original_price__gte=min_price)
            )
        
        if max_price:
            queryset = queryset.filter(
                Q(sale_price__lte=max_price) | 
                Q(sale_price__isnull=True, original_price__lte=max_price)
            )
        
        if brand_ids:
            queryset = queryset.filter(brand__id__in=brand_ids)
        
        # 정렬
        sort_by = self.request.query_params.get('sort_by', 'popular')
        
        if sort_by == 'price_low':
            queryset = queryset.extra(
                select={'final_price': 'COALESCE(sale_price, original_price)'}
            ).order_by('final_price')
        elif sort_by == 'price_high':
            queryset = queryset.extra(
                select={'final_price': 'COALESCE(sale_price, original_price)'}
            ).order_by('-final_price')
        elif sort_by == 'popular':
            queryset = queryset.order_by('-like_count', '-review_count')
        elif sort_by == 'newest':
            queryset = queryset.order_by('-crawled_at')
        elif sort_by == 'rating':
            queryset = queryset.order_by('-rating', '-review_count')
        elif sort_by == 'discount':
            queryset = queryset.order_by('-discount_rate')
        
        return queryset
    
    @action(detail=False, methods=['post'])
    def search(self, request):
        """상품 검색"""
        
        serializer = ProductSearchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        query = serializer.validated_data.get('query', '')
        
        # 캐시 키 생성
        cache_key = f"product_search:{query}:{serializer.validated_data}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return Response(cached_result)
        
        # 검색 쿼리 구성
        queryset = self.get_queryset()
        
        if query:
            queryset = queryset.filter(
                Q(name__icontains=query) |
                Q(brand__name__icontains=query) |
                Q(brand__name_ko__icontains=query) |
                Q(tags__icontains=query) |
                Q(style_tags__icontains=query)
            )
        
        # 결과 제한
        queryset = queryset[:100]
        
        serializer = ProductListSerializer(queryset, many=True)
        result = {
            'count': len(serializer.data),
            'results': serializer.data
        }
        
        # 캐시 저장 (5분)
        cache.set(cache_key, result, 300)
        
        return Response(result)
    
    @action(detail=False, methods=['get'])
    def trending(self, request):
        """인기 상품"""
        
        cache_key = 'trending_products'
        cached = cache.get(cache_key)
        
        if cached:
            return Response(cached)
        
        # 최근 일주일 내 크롤링된 상품 중 인기순
        products = Product.objects.filter(
            is_available=True
        ).order_by('-like_count', '-review_count')[:20]
        
        serializer = ProductListSerializer(products, many=True)
        result = serializer.data
        
        cache.set(cache_key, result, 600)  # 10분 캐시
        
        return Response(result)
    
    @action(detail=False, methods=['get'])
    def deals(self, request):
        """할인 상품"""
        
        cache_key = 'deal_products'
        cached = cache.get(cache_key)
        
        if cached:
            return Response(cached)
        
        products = Product.objects.filter(
            is_available=True,
            discount_rate__gte=30
        ).order_by('-discount_rate', '-like_count')[:50]
        
        serializer = ProductListSerializer(products, many=True)
        result = serializer.data
        
        cache.set(cache_key, result, 600)  # 10분 캐시
        
        return Response(result)
    
    @action(detail=False, methods=['get'])
    def recommendations(self, request):
        """추천 상품 (사용자 맞춤)"""
        
        if not request.user.is_authenticated:
            # 비로그인 사용자는 인기 상품 반환
            return self.trending(request)
        
        # 사용자의 선호 스타일 기반 추천 (향후 구현)
        # 일단은 카테고리별 인기 상품 반환
        recommendations = {}
        
        for category in ['TOP', 'BOTTOM', 'OUTER', 'SHOES']:
            products = Product.objects.filter(
                category=category,
                is_available=True
            ).order_by('-rating', '-review_count')[:5]
            
            recommendations[category] = ProductListSerializer(products, many=True).data
        
        return Response(recommendations)
    
    @action(detail=True, methods=['get'])
    def similar(self, request, pk=None):
        """유사 상품"""
        
        product = self.get_object()
        
        # 같은 카테고리, 비슷한 가격대의 상품
        price_range = product.final_price * 0.3
        min_price = product.final_price - price_range
        max_price = product.final_price + price_range
        
        similar_products = Product.objects.filter(
            category=product.category,
            is_available=True
        ).exclude(id=product.id).filter(
            Q(sale_price__range=(min_price, max_price)) |
            Q(sale_price__isnull=True, original_price__range=(min_price, max_price))
        ).order_by('-rating')[:10]
        
        serializer = ProductListSerializer(similar_products, many=True)
        return Response(serializer.data)

class BrandViewSet(viewsets.ReadOnlyModelViewSet):
    """브랜드 조회 API"""
    
    queryset = Brand.objects.all()
    serializer_class = BrandSerializer
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['get'])
    def popular(self, request):
        """인기 브랜드"""
        
        # 상품 수가 많은 브랜드
        brands = Brand.objects.annotate(
            product_count=Count('product')
        ).order_by('-product_count')[:20]
        
        serializer = self.get_serializer(brands, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def products(self, request, pk=None):
        """브랜드별 상품"""
        
        brand = self.get_object()
        products = Product.objects.filter(
            brand=brand,
            is_available=True
        ).order_by('-like_count')
        
        page = self.paginate_queryset(products)
        if page is not None:
            serializer = ProductListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = ProductListSerializer(products, many=True)
        return Response(serializer.data)

class CrawlManagementView(viewsets.ViewSet):
    """크롤링 관리 API (관리자 전용)"""
    
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['post'])
    def start_crawl(self, request):
        """크롤링 시작"""
        
        if not request.user.is_staff:
            return Response(
                {'error': '관리자 권한이 필요합니다'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        category = request.data.get('category')
        max_pages = request.data.get('max_pages', 5)
        
        # Celery 태스크 실행
        task = crawl_musinsa_products.delay(category, max_pages)
        
        return Response({
            'message': '크롤링이 시작되었습니다',
            'task_id': task.id
        })
    
    @action(detail=False, methods=['get'])
    def status(self, request):
        """크롤링 상태 확인"""
        
        from .models import CrawlLog
        
        recent_logs = CrawlLog.objects.all()[:10]
        
        logs_data = [{
            'source': log.source,
            'category': log.category,
            'status': log.status,
            'products_crawled': log.products_crawled,
            'started_at': log.started_at,
            'completed_at': log.completed_at,
            'error_message': log.error_message
        } for log in recent_logs]
        
        return Response(logs_data)