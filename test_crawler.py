#!/usr/bin/env python
"""크롤러 테스트 스크립트"""

import os
import sys
import django

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stylemate.settings')
sys.path.insert(0, os.path.dirname(__file__))
django.setup()

from crawler.musinsa_crawler import MusinsaCrawler
from crawler.models import Product, Brand
import json

def test_crawler():
    print("크롤러 테스트 시작...")
    
    crawler = MusinsaCrawler()
    
    # 샘플 상품 생성 (실제 크롤링 대신 테스트 데이터)
    test_products = [
        {
            'product_id': 'test_001',
            'source': 'musinsa',
            'name': '베이직 반팔 티셔츠',
            'brand': 'MUSINSA STANDARD',
            'product_url': 'https://www.musinsa.com/product/test_001',
            'main_image_url': 'https://image.musinsa.com/test.jpg',
            'original_price': 29900,
            'sale_price': 19900,
            'discount_rate': 33,
            'rating': 4.5,
            'review_count': 1234,
            'like_count': 5678,
        },
        {
            'product_id': 'test_002',
            'source': 'musinsa',
            'name': '와이드 데님 팬츠',
            'brand': 'COVERNAT',
            'product_url': 'https://www.musinsa.com/product/test_002',
            'main_image_url': 'https://image.musinsa.com/test2.jpg',
            'original_price': 89000,
            'sale_price': None,
            'discount_rate': 0,
            'rating': 4.8,
            'review_count': 456,
            'like_count': 2345,
        },
        {
            'product_id': 'test_003',
            'source': 'musinsa',
            'name': '오버사이즈 후드집업',
            'brand': 'THISISNEVERTHAT',
            'product_url': 'https://www.musinsa.com/product/test_003',
            'main_image_url': 'https://image.musinsa.com/test3.jpg',
            'original_price': 129000,
            'sale_price': 99000,
            'discount_rate': 23,
            'rating': 4.7,
            'review_count': 789,
            'like_count': 3456,
        }
    ]
    
    # 데이터베이스에 저장
    for product_data in test_products:
        # 브랜드 생성
        brand, _ = Brand.objects.get_or_create(
            name=product_data['brand'],
            defaults={'name_ko': product_data['brand']}
        )
        
        # 상품 생성
        product, created = Product.objects.update_or_create(
            product_id=product_data['product_id'],
            source=product_data['source'],
            defaults={
                'name': product_data['name'],
                'brand': brand,
                'category': 'TOP' if '티셔츠' in product_data['name'] or '후드' in product_data['name'] else 'BOTTOM',
                'gender': 'U',
                'original_price': product_data['original_price'],
                'sale_price': product_data.get('sale_price'),
                'discount_rate': product_data.get('discount_rate', 0),
                'main_image_url': product_data['main_image_url'],
                'product_url': product_data['product_url'],
                'rating': product_data.get('rating', 0),
                'review_count': product_data.get('review_count', 0),
                'like_count': product_data.get('like_count', 0),
                'description': f"{product_data['name']} - 테스트 상품입니다.",
                'colors': ['블랙', '화이트', '그레이'],
                'sizes': ['S', 'M', 'L', 'XL'],
                'is_available': True,
            }
        )
        
        if created:
            print(f"✅ 생성됨: {product.name}")
        else:
            print(f"📝 업데이트됨: {product.name}")
    
    # 저장된 상품 확인
    total_products = Product.objects.count()
    total_brands = Brand.objects.count()
    
    print(f"\n📊 데이터베이스 통계:")
    print(f"- 총 상품 수: {total_products}")
    print(f"- 총 브랜드 수: {total_brands}")
    
    # 상품 조회 테스트
    print("\n🔍 저장된 상품 목록:")
    for product in Product.objects.all()[:5]:
        print(f"- [{product.brand.name}] {product.name} - {product.final_price:,}원")

if __name__ == "__main__":
    test_crawler()