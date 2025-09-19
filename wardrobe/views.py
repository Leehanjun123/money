from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import ClothingItem, Outfit, StylePreference
from .serializers import ClothingItemSerializer, OutfitSerializer, StylePreferenceSerializer

class ClothingItemViewSet(viewsets.ModelViewSet):
    serializer_class = ClothingItemSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return ClothingItem.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=False, methods=['get'])
    def by_category(self, request):
        category = request.query_params.get('category', None)
        if category:
            items = self.get_queryset().filter(category=category)
        else:
            items = self.get_queryset()
        serializer = self.get_serializer(items, many=True)
        return Response(serializer.data)

class OutfitViewSet(viewsets.ModelViewSet):
    serializer_class = OutfitSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Outfit.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=False, methods=['get'])
    def recommended(self, request):
        # AI 추천 로직 구현 예정
        outfits = self.get_queryset().filter(ai_generated=True)[:5]
        serializer = self.get_serializer(outfits, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def wear(self, request, pk=None):
        outfit = self.get_object()
        outfit.wear_count += 1
        outfit.save()
        return Response({'status': 'outfit worn', 'wear_count': outfit.wear_count})

class StylePreferenceViewSet(viewsets.ModelViewSet):
    serializer_class = StylePreferenceSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return StylePreference.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
