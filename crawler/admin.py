from django.contrib import admin
from .models import Product, Brand, CrawlLog, PriceHistory

@admin.register(Brand)
class BrandAdmin(admin.ModelAdmin):
    list_display = ['name', 'name_ko', 'is_premium', 'created_at']
    search_fields = ['name', 'name_ko']
    list_filter = ['is_premium']

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'brand', 'category', 'original_price', 'sale_price', 'discount_rate', 'rating', 'is_available']
    list_filter = ['category', 'gender', 'source', 'is_available']
    search_fields = ['name', 'brand__name', 'product_id']
    readonly_fields = ['crawled_at', 'updated_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('brand')

@admin.register(CrawlLog)
class CrawlLogAdmin(admin.ModelAdmin):
    list_display = ['source', 'category', 'status', 'products_crawled', 'started_at', 'completed_at']
    list_filter = ['source', 'status']
    readonly_fields = ['started_at', 'completed_at']

@admin.register(PriceHistory)
class PriceHistoryAdmin(admin.ModelAdmin):
    list_display = ['product', 'original_price', 'sale_price', 'recorded_at']
    list_filter = ['recorded_at']
    search_fields = ['product__name']