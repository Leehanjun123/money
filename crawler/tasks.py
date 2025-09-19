from celery import shared_task
from django.utils import timezone
from datetime import datetime, timedelta
import logging
from .musinsa_crawler import MusinsaCrawler
from .models import Product, Brand, CrawlLog, PriceHistory

logger = logging.getLogger(__name__)

@shared_task
def crawl_musinsa_products(category: str = None, max_pages: int = 5):
    """무신사 상품 크롤링 태스크"""
    
    crawl_log = CrawlLog.objects.create(
        source='musinsa',
        category=category or 'all',
        status='running',
        started_at=timezone.now()
    )
    
    try:
        crawler = MusinsaCrawler()
        
        if category:
            products = crawler.crawl_category(category, max_pages)
            results = {category: products}
        else:
            results = crawler.crawl_all_categories(max_pages_per_category=max_pages)
        
        # 크롤링 결과 저장
        total_new = 0
        total_updated = 0
        
        for category_name, products in results.items():
            for product_data in products:
                new_count, updated_count = save_product(product_data)
                total_new += new_count
                total_updated += updated_count
        
        # 크롤링 로그 업데이트
        crawl_log.status = 'success'
        crawl_log.products_crawled = total_new + total_updated
        crawl_log.products_new = total_new
        crawl_log.products_updated = total_updated
        crawl_log.completed_at = timezone.now()
        crawl_log.save()
        
        logger.info(f"Crawling completed: {total_new} new, {total_updated} updated")
        return {'new': total_new, 'updated': total_updated}
        
    except Exception as e:
        crawl_log.status = 'failed'
        crawl_log.error_message = str(e)
        crawl_log.completed_at = timezone.now()
        crawl_log.save()
        
        logger.error(f"Crawling failed: {e}")
        raise

def save_product(product_data: dict) -> tuple:
    """상품 데이터 저장"""
    
    # 브랜드 생성 또는 가져오기
    brand, _ = Brand.objects.get_or_create(
        name=product_data['brand'],
        defaults={'name_ko': product_data['brand']}
    )
    
    # 상품 생성 또는 업데이트
    product, created = Product.objects.update_or_create(
        product_id=product_data['product_id'],
        source=product_data['source'],
        defaults={
            'name': product_data['name'],
            'brand': brand,
            'category': detect_category(product_data['name']),
            'original_price': product_data['original_price'],
            'sale_price': product_data.get('sale_price'),
            'discount_rate': product_data.get('discount_rate', 0),
            'main_image_url': product_data['main_image_url'],
            'product_url': product_data['product_url'],
            'rating': product_data.get('rating', 0),
            'review_count': product_data.get('review_count', 0),
            'like_count': product_data.get('like_count', 0),
            'last_checked': timezone.now(),
        }
    )
    
    # 가격 히스토리 저장
    if not created:
        # 가격이 변경된 경우에만 히스토리 저장
        last_price = product.price_history.first()
        if not last_price or last_price.sale_price != product_data.get('sale_price'):
            PriceHistory.objects.create(
                product=product,
                original_price=product_data['original_price'],
                sale_price=product_data.get('sale_price')
            )
    
    return (1 if created else 0, 0 if created else 1)

def detect_category(name: str) -> str:
    """상품명에서 카테고리 추정"""
    
    name_lower = name.lower()
    
    if any(keyword in name_lower for keyword in ['티셔츠', '셔츠', '니트', '맨투맨', '후드']):
        return 'TOP'
    elif any(keyword in name_lower for keyword in ['자켓', '코트', '패딩', '점퍼']):
        return 'OUTER'
    elif any(keyword in name_lower for keyword in ['팬츠', '바지', '진', '슬랙스']):
        return 'BOTTOM'
    elif any(keyword in name_lower for keyword in ['원피스', '드레스']):
        return 'DRESS'
    elif any(keyword in name_lower for keyword in ['신발', '스니커즈', '구두', '부츠']):
        return 'SHOES'
    elif any(keyword in name_lower for keyword in ['가방', '백팩', '크로스백']):
        return 'BAG'
    else:
        return 'ACC'

@shared_task
def update_product_prices():
    """상품 가격 업데이트 태스크 (매일 실행)"""
    
    # 24시간 이내에 체크하지 않은 상품들
    outdated_products = Product.objects.filter(
        last_checked__lt=timezone.now() - timedelta(hours=24)
    )[:100]  # 한 번에 100개씩
    
    crawler = MusinsaCrawler()
    updated_count = 0
    
    for product in outdated_products:
        try:
            # 상품 페이지에서 최신 정보 가져오기
            product_data = crawler.get_product_detail(product.product_url)
            
            if product_data:
                # 가격 업데이트
                if product_data.get('sale_price') != product.sale_price:
                    PriceHistory.objects.create(
                        product=product,
                        original_price=product_data.get('original_price', product.original_price),
                        sale_price=product_data.get('sale_price')
                    )
                    
                    product.sale_price = product_data.get('sale_price')
                    product.discount_rate = crawler._calculate_discount_rate(
                        product.original_price,
                        product_data.get('sale_price')
                    )
                
                product.last_checked = timezone.now()
                product.save()
                updated_count += 1
                
        except Exception as e:
            logger.error(f"Failed to update product {product.product_id}: {e}")
    
    logger.info(f"Updated {updated_count} product prices")
    return updated_count

@shared_task
def clean_old_price_history():
    """오래된 가격 히스토리 정리 (30일 이상)"""
    
    cutoff_date = timezone.now() - timedelta(days=30)
    deleted_count = PriceHistory.objects.filter(recorded_at__lt=cutoff_date).delete()[0]
    
    logger.info(f"Deleted {deleted_count} old price history records")
    return deleted_count