from django.core.management.base import BaseCommand
from crawler.musinsa_crawler import MusinsaCrawler
from crawler.models import CrawlLog
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    """무신사 크롤링 실행 명령어"""
    
    help = 'Run Musinsa crawling for specified categories'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--categories',
            nargs='+',
            default=['001', '002', '003', '004', '005'],
            help='Categories to crawl (default: 상의, 하의, 신발, 가방, 액세서리)'
        )
        parser.add_argument(
            '--max-pages',
            type=int,
            default=50,
            help='Maximum pages to crawl per category'
        )
        parser.add_argument(
            '--delay',
            type=float,
            default=2.0,
            help='Delay between requests in seconds'
        )
        parser.add_argument(
            '--full-crawl',
            action='store_true',
            help='Perform full crawl (all categories, more pages)'
        )
    
    def handle(self, *args, **options):
        start_time = timezone.now()
        
        # 크롤링 로그 시작
        crawl_log = CrawlLog.objects.create(
            crawler_name='musinsa_main',
            start_time=start_time,
            status='running'
        )
        
        try:
            crawler = MusinsaCrawler()
            
            # 전체 크롤링 모드
            if options['full_crawl']:
                self.stdout.write(
                    self.style.SUCCESS('🚀 무신사 전체 크롤링 시작 - 시장 독점 모드!')
                )
                categories = list(crawler.CATEGORIES.keys())
                max_pages = 100
                delay = 1.5
            else:
                categories = options['categories']
                max_pages = options['max_pages']
                delay = options['delay']
            
            self.stdout.write(f'카테고리: {categories}')
            self.stdout.write(f'최대 페이지: {max_pages}')
            self.stdout.write(f'요청 간격: {delay}초')
            
            total_products = 0
            total_errors = 0
            
            # 카테고리별 크롤링
            for category_id in categories:
                if category_id not in crawler.CATEGORIES:
                    self.stdout.write(
                        self.style.ERROR(f'Unknown category: {category_id}')
                    )
                    continue
                
                category_name = crawler.CATEGORIES[category_id]['name']
                self.stdout.write(
                    self.style.SUCCESS(f'\n📦 {category_name} 카테고리 크롤링 시작...')
                )
                
                try:
                    # 카테고리 크롤링 실행
                    result = crawler.crawl_category(
                        category_id,
                        max_pages=max_pages,
                        delay=delay
                    )
                    
                    products_count = result.get('products_count', 0)
                    errors_count = result.get('errors_count', 0)
                    
                    total_products += products_count
                    total_errors += errors_count
                    
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'✅ {category_name}: {products_count}개 상품 수집, {errors_count}개 오류'
                        )
                    )
                    
                    # 서브카테고리별 크롤링
                    subcategories = crawler.CATEGORIES[category_id].get('subcategories', {})
                    for sub_id, sub_info in subcategories.items():
                        if isinstance(sub_info, dict) and 'name' in sub_info:
                            sub_name = sub_info['name']
                            
                            self.stdout.write(f'  🔸 {sub_name} 서브카테고리 크롤링...')
                            
                            sub_result = crawler.crawl_subcategory(
                                category_id, 
                                sub_id,
                                max_pages=min(max_pages // 2, 25),
                                delay=delay
                            )
                            
                            sub_products = sub_result.get('products_count', 0)
                            sub_errors = sub_result.get('errors_count', 0)
                            
                            total_products += sub_products
                            total_errors += sub_errors
                            
                            self.stdout.write(
                                f'    ✅ {sub_products}개 상품, {sub_errors}개 오류'
                            )
                    
                except Exception as e:
                    logger.error(f"Category {category_id} crawling failed: {e}")
                    self.stdout.write(
                        self.style.ERROR(f'❌ {category_name} 크롤링 실패: {e}')
                    )
                    total_errors += 1
            
            # 크롤링 완료
            end_time = timezone.now()
            duration = end_time - start_time
            
            crawl_log.end_time = end_time
            crawl_log.status = 'completed'
            crawl_log.products_crawled = total_products
            crawl_log.errors_count = total_errors
            crawl_log.save()
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'\n🎉 무신사 크롤링 완료!'
                    f'\n📊 총 {total_products}개 상품 수집'
                    f'\n⚠️ {total_errors}개 오류 발생'
                    f'\n⏱️ 소요시간: {duration}'
                )
            )
            
            # 시장 독점 메시지
            if options['full_crawl'] and total_products > 10000:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'\n🚀 StyleMate 시장 독점 모드 - {total_products}개 상품으로 무신사 제압!'
                    )
                )
        
        except Exception as e:
            # 크롤링 실패
            crawl_log.end_time = timezone.now()
            crawl_log.status = 'failed'
            crawl_log.error_message = str(e)
            crawl_log.save()
            
            logger.error(f"Crawling failed: {e}")
            self.stdout.write(
                self.style.ERROR(f'❌ 크롤링 실패: {e}')
            )