from django.urls import path
from .views import HealthCheckView, AnalyzeClothingView
from .ai_views import AIStyleAnalysisView, AIOutfitRecommendationView, AIOutfitScoringView, AITrendAnalysisView
from .ar_views import ARBodyAnalysisView, ARSizeRecommendationView, ARVirtualTryOnView
from .search_views import ProductSearchView, ProductRecommendationView, TrendingProductsView, SearchSuggestionsView

urlpatterns = [
    # 기본 API
    path('health/', HealthCheckView.as_view(), name='health_check'),
    path('analyze/', AnalyzeClothingView.as_view(), name='analyze_clothing'),
    
    # AI 스타일 분석
    path('ai/style-analysis/', AIStyleAnalysisView.as_view(), name='ai_style_analysis'),
    path('ai/outfit-recommendation/', AIOutfitRecommendationView.as_view(), name='ai_outfit_recommendation'),
    path('ai/outfit-scoring/', AIOutfitScoringView.as_view(), name='ai_outfit_scoring'),
    path('ai/trend-analysis/', AITrendAnalysisView.as_view(), name='ai_trend_analysis'),
    
    # AR 가상 피팅
    path('ar/body-analysis/', ARBodyAnalysisView.as_view(), name='ar_body_analysis'),
    path('ar/size-recommendation/', ARSizeRecommendationView.as_view(), name='ar_size_recommendation'),
    path('ar/virtual-try-on/', ARVirtualTryOnView.as_view(), name='ar_virtual_try_on'),
    
    # 고성능 검색
    path('search/', ProductSearchView.as_view(), name='product_search'),
    path('search/suggestions/', SearchSuggestionsView.as_view(), name='search_suggestions'),
    path('products/<int:product_id>/recommendations/', ProductRecommendationView.as_view(), name='product_recommendations'),
    path('trending/', TrendingProductsView.as_view(), name='trending_products'),
]