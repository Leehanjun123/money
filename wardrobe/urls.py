from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ClothingItemViewSet, OutfitViewSet, StylePreferenceViewSet

router = DefaultRouter()
router.register(r'items', ClothingItemViewSet, basename='clothingitem')
router.register(r'outfits', OutfitViewSet, basename='outfit')
router.register(r'preferences', StylePreferenceViewSet, basename='stylepreference')

urlpatterns = [
    path('', include(router.urls)),
]