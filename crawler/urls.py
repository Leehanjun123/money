from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ProductViewSet, BrandViewSet, CrawlManagementView

router = DefaultRouter()
router.register(r'products', ProductViewSet, basename='product')
router.register(r'brands', BrandViewSet, basename='brand')
router.register(r'crawl', CrawlManagementView, basename='crawl')

urlpatterns = [
    path('', include(router.urls)),
]