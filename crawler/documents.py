from django_elasticsearch_dsl import Document, fields
from django_elasticsearch_dsl.registries import registry
from .models import Product, Brand

@registry.register_document
class ProductDocument(Document):
    """ElasticSearch 상품 검색 문서"""
    
    # 브랜드 관계 필드
    brand = fields.ObjectField(properties={
        'id': fields.IntegerField(),
        'name': fields.TextField(
            fields={'keyword': fields.KeywordField()}
        ),
        'name_ko': fields.TextField(
            fields={'keyword': fields.KeywordField()}
        ),
        'is_premium': fields.BooleanField(),
    })
    
    # 중첩 필드들을 개별 필드로 정의
    colors = fields.TextField(multi=True)
    sizes = fields.TextField(multi=True)
    tags = fields.TextField(multi=True)
    style_tags = fields.TextField(multi=True)
    
    # 자동 완성을 위한 suggest 필드
    name_suggest = fields.CompletionField()
    brand_suggest = fields.CompletionField()
    
    class Index:
        name = 'products'
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0,
            'analysis': {
                'analyzer': {
                    'korean_analyzer': {
                        'type': 'custom',
                        'tokenizer': 'standard',
                        'filter': ['lowercase', 'stop', 'snowball']
                    },
                    'edge_ngram_analyzer': {
                        'type': 'custom',
                        'tokenizer': 'edge_ngram_tokenizer',
                        'filter': ['lowercase']
                    }
                },
                'tokenizer': {
                    'edge_ngram_tokenizer': {
                        'type': 'edge_ngram',
                        'min_gram': 1,
                        'max_gram': 10,
                        'token_chars': ['letter', 'digit']
                    }
                }
            }
        }
    
    class Django:
        model = Product
        fields = [
            'id',
            'product_id',
            'source',
            'name',
            'category',
            'subcategory',
            'gender',
            'original_price',
            'sale_price',
            'discount_rate',
            'description',
            'material',
            'main_image_url',
            'product_url',
            'rating',
            'review_count',
            'sales_count',
            'view_count',
            'like_count',
            'is_available',
            'stock_status',
            'crawled_at',
            'updated_at',
        ]
        related_models = ['brand']
    
    def get_instances_from_related(self, related_instance):
        """브랜드가 업데이트되면 관련 상품도 업데이트"""
        if isinstance(related_instance, Brand):
            return related_instance.product_set.all()
    
    def prepare_name_suggest(self, instance):
        """상품명 자동완성 데이터"""
        return {
            'input': [instance.name, instance.name.lower()],
            'weight': instance.like_count + instance.review_count
        }
    
    def prepare_brand_suggest(self, instance):
        """브랜드명 자동완성 데이터"""
        return {
            'input': [instance.brand.name, instance.brand.name_ko],
            'weight': 10
        }

@registry.register_document  
class BrandDocument(Document):
    """브랜드 검색 문서"""
    
    product_count = fields.IntegerField()
    avg_rating = fields.FloatField()
    
    class Index:
        name = 'brands'
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0
        }
    
    class Django:
        model = Brand
        fields = [
            'id',
            'name',
            'name_ko',
            'logo_url',
            'description',
            'is_premium',
            'created_at'
        ]
    
    def prepare_product_count(self, instance):
        """브랜드별 상품 수"""
        return instance.product_set.count()
    
    def prepare_avg_rating(self, instance):
        """브랜드 평균 평점"""
        products = instance.product_set.all()
        if products:
            total_rating = sum(p.rating for p in products)
            return total_rating / len(products)
        return 0.0