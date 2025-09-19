from rest_framework import serializers
from .models import ClothingItem, Outfit, StylePreference

class ClothingItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClothingItem
        fields = '__all__'
        read_only_fields = ('user', 'created_at', 'updated_at', 'ai_analysis')

class OutfitSerializer(serializers.ModelSerializer):
    items = ClothingItemSerializer(many=True, read_only=True)
    item_ids = serializers.PrimaryKeyRelatedField(
        many=True, 
        write_only=True, 
        queryset=ClothingItem.objects.all(),
        source='items'
    )
    
    class Meta:
        model = Outfit
        fields = '__all__'
        read_only_fields = ('user', 'created_at', 'updated_at')

class StylePreferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StylePreference
        fields = '__all__'
        read_only_fields = ('user', 'created_at', 'updated_at')