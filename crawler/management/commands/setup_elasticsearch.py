from django.core.management.base import BaseCommand
from crawler.elasticsearch_service import elasticsearch_service
from crawler.models import Product, Brand
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    """ElasticSearch 설정 및 초기 인덱싱 명령어"""
    
    help = 'Setup ElasticSearch indices and perform initial indexing'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--create-indices',
            action='store_true',
            help='Create ElasticSearch indices'
        )
        parser.add_argument(
            '--index-products',
            action='store_true',
            help='Index all products'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=500,
            help='Batch size for bulk indexing'
        )
        parser.add_argument(
            '--limit',
            type=int,
            help='Limit number of products to index'
        )
    
    def handle(self, *args, **options):
        if not elasticsearch_service.check_connection():
            self.stdout.write(
                self.style.ERROR('ElasticSearch connection failed. Check configuration.')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS('ElasticSearch connection established')
        )
        
        if options['create_indices']:
            self.create_indices()
        
        if options['index_products']:
            self.index_products(
                batch_size=options['batch_size'],
                limit=options.get('limit')
            )
    
    def create_indices(self):
        """인덱스 생성"""
        self.stdout.write('Creating ElasticSearch indices...')
        
        # 상품 인덱스 생성
        if elasticsearch_service.create_product_index():
            self.stdout.write(
                self.style.SUCCESS('Product index created successfully')
            )
        else:
            self.stdout.write(
                self.style.ERROR('Failed to create product index')
            )
        
        # 브랜드 인덱스 생성
        if elasticsearch_service.create_brand_index():
            self.stdout.write(
                self.style.SUCCESS('Brand index created successfully')
            )
        else:
            self.stdout.write(
                self.style.ERROR('Failed to create brand index')
            )
    
    def index_products(self, batch_size: int = 500, limit: int = None):
        """상품 인덱싱"""
        self.stdout.write('Starting product indexing...')
        
        queryset = Product.objects.select_related('brand').filter(is_available=True)
        
        if limit:
            queryset = queryset[:limit]
        
        total_count = queryset.count()
        self.stdout.write(f'Total products to index: {total_count}')
        
        # 배치 단위로 인덱싱
        processed = 0
        for i in range(0, total_count, batch_size):
            batch = list(queryset[i:i + batch_size])
            
            if elasticsearch_service.bulk_index_products(batch, batch_size):
                processed += len(batch)
                self.stdout.write(
                    f'Indexed {processed}/{total_count} products'
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'Failed to index batch {i}-{i + batch_size}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'Product indexing completed: {processed} products')
        )