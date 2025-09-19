from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.utils import timezone

class Brand(models.Model):
    name = models.CharField(max_length=100, unique=True)
    name_ko = models.CharField(max_length=100)
    logo_url = models.URLField(blank=True)
    description = models.TextField(blank=True)
    is_premium = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Product(models.Model):
    GENDER_CHOICES = [
        ('M', '남성'),
        ('F', '여성'),
        ('U', '공용'),
    ]
    
    CATEGORY_CHOICES = [
        ('TOP', '상의'),
        ('BOTTOM', '하의'),
        ('OUTER', '아우터'),
        ('DRESS', '원피스'),
        ('SHOES', '신발'),
        ('BAG', '가방'),
        ('ACC', '액세서리'),
    ]
    
    # 기본 정보
    product_id = models.CharField(max_length=100, unique=True)  # 쇼핑몰 고유 ID
    source = models.CharField(max_length=50)  # 'musinsa', '29cm', 'wconcept'
    name = models.CharField(max_length=200)
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    subcategory = models.CharField(max_length=100)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    
    # 가격 정보
    original_price = models.DecimalField(max_digits=10, decimal_places=0)
    sale_price = models.DecimalField(max_digits=10, decimal_places=0, null=True, blank=True)
    discount_rate = models.IntegerField(default=0)
    
    # 상품 정보
    description = models.TextField()
    material = models.CharField(max_length=200, blank=True)
    colors = models.JSONField(default=list)  # ['블랙', '화이트', '네이비']
    sizes = models.JSONField(default=list)   # ['S', 'M', 'L', 'XL']
    
    # 이미지
    main_image_url = models.URLField()
    image_urls = models.JSONField(default=list)
    
    # 메타 정보
    product_url = models.URLField()
    tags = models.JSONField(default=list)
    style_tags = models.JSONField(default=list)  # ['미니멀', '캐주얼', '스트릿']
    
    # 인기도/평점
    rating = models.FloatField(default=0.0)
    review_count = models.IntegerField(default=0)
    sales_count = models.IntegerField(default=0)
    view_count = models.IntegerField(default=0)
    like_count = models.IntegerField(default=0)
    
    # 재고 상태
    is_available = models.BooleanField(default=True)
    stock_status = models.CharField(max_length=50, default='in_stock')
    
    # 크롤링 정보
    crawled_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_checked = models.DateTimeField(default=timezone.now)
    
    class Meta:
        indexes = [
            models.Index(fields=['source', 'product_id']),
            models.Index(fields=['category', 'gender']),
            models.Index(fields=['brand']),
            models.Index(fields=['-like_count']),
            models.Index(fields=['-sales_count']),
            models.Index(fields=['sale_price']),
        ]
        ordering = ['-crawled_at']
    
    def __str__(self):
        return f"[{self.brand.name}] {self.name}"
    
    @property
    def final_price(self):
        return self.sale_price if self.sale_price else self.original_price

class PriceHistory(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='price_history')
    original_price = models.DecimalField(max_digits=10, decimal_places=0)
    sale_price = models.DecimalField(max_digits=10, decimal_places=0, null=True, blank=True)
    recorded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-recorded_at']

class CrawlLog(models.Model):
    source = models.CharField(max_length=50)
    category = models.CharField(max_length=50)
    status = models.CharField(max_length=20)  # 'success', 'failed', 'partial'
    products_crawled = models.IntegerField(default=0)
    products_updated = models.IntegerField(default=0)
    products_new = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField()
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"{self.source} - {self.started_at}"