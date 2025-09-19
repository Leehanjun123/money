from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class ClothingCategory(models.TextChoices):
    TOP = 'TOP', '상의'
    BOTTOM = 'BOTTOM', '하의'
    OUTER = 'OUTER', '아우터'
    DRESS = 'DRESS', '원피스'
    SHOES = 'SHOES', '신발'
    BAG = 'BAG', '가방'
    ACCESSORY = 'ACCESSORY', '액세서리'

class ClothingItem(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='clothing_items')
    category = models.CharField(max_length=20, choices=ClothingCategory.choices)
    brand = models.CharField(max_length=100, blank=True)
    name = models.CharField(max_length=200)
    color = models.CharField(max_length=50)
    material = models.CharField(max_length=100, blank=True)
    size = models.CharField(max_length=20, blank=True)
    purchase_price = models.DecimalField(max_digits=10, decimal_places=0, null=True, blank=True)
    purchase_date = models.DateField(null=True, blank=True)
    image = models.ImageField(upload_to='clothing_items/', null=True, blank=True)
    tags = models.JSONField(default=list)
    ai_analysis = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.name}"

class Outfit(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='outfits')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    items = models.ManyToManyField(ClothingItem)
    occasion = models.CharField(max_length=100)
    season = models.CharField(max_length=20)
    weather = models.CharField(max_length=50, blank=True)
    rating = models.IntegerField(default=0, validators=[MinValueValidator(0), MaxValueValidator(5)])
    wear_count = models.IntegerField(default=0)
    last_worn = models.DateTimeField(null=True, blank=True)
    ai_generated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.name}"

class StylePreference(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='style_preference')
    preferred_styles = models.JSONField(default=list)  # ['casual', 'business', 'streetwear']
    avoided_styles = models.JSONField(default=list)
    preferred_colors = models.JSONField(default=list)
    avoided_colors = models.JSONField(default=list)
    preferred_brands = models.JSONField(default=list)
    budget_min = models.DecimalField(max_digits=10, decimal_places=0, null=True, blank=True)
    budget_max = models.DecimalField(max_digits=10, decimal_places=0, null=True, blank=True)
    body_type = models.CharField(max_length=50, blank=True)
    height = models.IntegerField(null=True, blank=True)  # in cm
    weight = models.IntegerField(null=True, blank=True)  # in kg
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - Style Preferences"
