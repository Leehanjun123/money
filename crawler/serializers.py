from rest_framework import serializers
from .models import Product, Brand, PriceHistory

class BrandSerializer(serializers.ModelSerializer):
    class Meta:
        model = Brand
        fields = ['id', 'name', 'name_ko', 'logo_url', 'is_premium']

class PriceHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PriceHistory
        fields = ['original_price', 'sale_price', 'recorded_at']

class ProductListSerializer(serializers.ModelSerializer):
    brand = BrandSerializer(read_only=True)
    final_price = serializers.ReadOnlyField()
    
    class Meta:
        model = Product
        fields = [
            'id', 'product_id', 'name', 'brand', 'category',
            'original_price', 'sale_price', 'final_price', 'discount_rate',
            'main_image_url', 'rating', 'review_count', 'like_count',
            'is_available'
        ]

class ProductDetailSerializer(serializers.ModelSerializer):
    brand = BrandSerializer(read_only=True)
    final_price = serializers.ReadOnlyField()
    price_history = PriceHistorySerializer(many=True, read_only=True)
    
    class Meta:
        model = Product
        fields = '__all__'

class ProductSearchSerializer(serializers.Serializer):
    query = serializers.CharField(required=False)
    category = serializers.ChoiceField(
        choices=['TOP', 'BOTTOM', 'OUTER', 'DRESS', 'SHOES', 'BAG', 'ACC'],
        required=False
    )
    gender = serializers.ChoiceField(choices=['M', 'F', 'U'], required=False)
    min_price = serializers.IntegerField(required=False, min_value=0)
    max_price = serializers.IntegerField(required=False, min_value=0)
    brands = serializers.ListField(
        child=serializers.CharField(),
        required=False
    )
    sort_by = serializers.ChoiceField(
        choices=['price_low', 'price_high', 'popular', 'newest', 'rating', 'discount'],
        default='popular'
    )