from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from django.conf import settings
from django.core.management.base import BaseCommand
from crawler.models import Product, Brand
from typing import List, Dict, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ElasticsearchService:
    """ElasticSearch 서비스 클래스"""
    
    def __init__(self):
        self.es = Elasticsearch(
            [settings.ELASTICSEARCH_HOST],
            http_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD) if hasattr(settings, 'ELASTICSEARCH_USER') else None,
            http_compress=True,
            timeout=30,
            max_retries=10,
            retry_on_timeout=True
        ) if hasattr(settings, 'ELASTICSEARCH_HOST') else None
        
        self.products_index = 'stylemate_products'
        self.brands_index = 'stylemate_brands'
    
    def check_connection(self) -> bool:
        """ElasticSearch 연결 확인"""
        if not self.es:
            logger.warning("ElasticSearch not configured")
            return False
        
        try:
            return self.es.ping()
        except Exception as e:
            logger.error(f"ElasticSearch connection failed: {e}")
            return False
    
    def create_product_index(self):
        """상품 인덱스 생성"""
        if not self.es:
            return False
            
        mapping = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "korean_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "cjk_width",
                                "stop"
                            ]
                        },
                        "brand_analyzer": {
                            "type": "keyword"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "long"},
                    "name": {
                        "type": "text",
                        "analyzer": "korean_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "brand": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "long"},
                            "name": {
                                "type": "text",
                                "analyzer": "brand_analyzer",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "korean_name": {
                                "type": "text",
                                "analyzer": "korean_analyzer"
                            }
                        }
                    },
                    "category": {
                        "type": "object",
                        "properties": {
                            "large": {"type": "keyword"},
                            "medium": {"type": "keyword"},
                            "small": {"type": "keyword"}
                        }
                    },
                    "colors": {
                        "type": "keyword"
                    },
                    "sizes": {
                        "type": "keyword"
                    },
                    "style_tags": {
                        "type": "keyword"
                    },
                    "material": {"type": "text", "analyzer": "korean_analyzer"},
                    "fit": {"type": "keyword"},
                    "price": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "long"},
                            "final": {"type": "long"},
                            "discount_rate": {"type": "float"}
                        }
                    },
                    "rating": {
                        "type": "object",
                        "properties": {
                            "average": {"type": "float"},
                            "count": {"type": "long"}
                        }
                    },
                    "popularity": {
                        "type": "object",
                        "properties": {
                            "view_count": {"type": "long"},
                            "like_count": {"type": "long"},
                            "purchase_count": {"type": "long"}
                        }
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "korean_analyzer"
                    },
                    "images": {
                        "type": "object",
                        "properties": {
                            "main": {"type": "keyword"},
                            "list": {"type": "keyword"}
                        }
                    },
                    "is_available": {"type": "boolean"},
                    "gender": {"type": "keyword"},
                    "season": {"type": "keyword"},
                    "crawled_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "url": {"type": "keyword"},
                    "musinsa_id": {"type": "keyword"}
                }
            }
        }
        
        try:
            if self.es.indices.exists(index=self.products_index):
                self.es.indices.delete(index=self.products_index)
            
            self.es.indices.create(index=self.products_index, body=mapping)
            logger.info(f"Created {self.products_index} index")
            return True
        except Exception as e:
            logger.error(f"Failed to create product index: {e}")
            return False
    
    def create_brand_index(self):
        """브랜드 인덱스 생성"""
        if not self.es:
            return False
            
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "brand_analyzer": {
                            "type": "custom",
                            "tokenizer": "keyword",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "long"},
                    "name": {
                        "type": "text",
                        "analyzer": "brand_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "korean_name": {
                        "type": "text",
                        "analyzer": "korean_analyzer"
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "korean_analyzer"
                    },
                    "logo_image": {"type": "keyword"},
                    "product_count": {"type": "long"},
                    "average_price": {"type": "float"},
                    "is_active": {"type": "boolean"},
                    "created_at": {"type": "date"}
                }
            }
        }
        
        try:
            if self.es.indices.exists(index=self.brands_index):
                self.es.indices.delete(index=self.brands_index)
            
            self.es.indices.create(index=self.brands_index, body=mapping)
            logger.info(f"Created {self.brands_index} index")
            return True
        except Exception as e:
            logger.error(f"Failed to create brand index: {e}")
            return False
    
    def index_product(self, product: Product) -> bool:
        """단일 상품 인덱싱"""
        if not self.es:
            return False
            
        try:
            doc = {
                "id": product.id,
                "name": product.name,
                "brand": {
                    "id": product.brand.id,
                    "name": product.brand.name,
                    "korean_name": product.brand.korean_name or product.brand.name
                } if product.brand else None,
                "category": {
                    "large": product.large_category,
                    "medium": product.medium_category,
                    "small": product.small_category
                },
                "colors": product.colors or [],
                "sizes": product.sizes or [],
                "style_tags": product.style_tags or [],
                "material": product.material,
                "fit": product.fit,
                "price": {
                    "original": int(product.original_price) if product.original_price else 0,
                    "final": int(product.final_price) if product.final_price else 0,
                    "discount_rate": float(product.discount_rate) if product.discount_rate else 0
                },
                "rating": {
                    "average": float(product.rating) if product.rating else 0,
                    "count": product.review_count or 0
                },
                "popularity": {
                    "view_count": product.view_count or 0,
                    "like_count": product.like_count or 0,
                    "purchase_count": 0  # 추후 구현
                },
                "description": product.description or "",
                "images": {
                    "main": product.main_image_url,
                    "list": product.detail_images or []
                },
                "is_available": product.is_available,
                "gender": product.gender,
                "season": product.season,
                "crawled_at": product.crawled_at.isoformat() if product.crawled_at else None,
                "updated_at": product.updated_at.isoformat() if product.updated_at else None,
                "url": product.url,
                "musinsa_id": product.musinsa_id
            }
            
            self.es.index(
                index=self.products_index,
                id=product.id,
                body=doc
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index product {product.id}: {e}")
            return False
    
    def bulk_index_products(self, products: List[Product], batch_size: int = 500):
        """대량 상품 인덱싱"""
        if not self.es:
            return False
            
        def generate_docs():
            for product in products:
                doc = {
                    "_index": self.products_index,
                    "_id": product.id,
                    "_source": {
                        "id": product.id,
                        "name": product.name,
                        "brand": {
                            "id": product.brand.id,
                            "name": product.brand.name,
                            "korean_name": product.brand.korean_name or product.brand.name
                        } if product.brand else None,
                        "category": {
                            "large": product.large_category,
                            "medium": product.medium_category,
                            "small": product.small_category
                        },
                        "colors": product.colors or [],
                        "sizes": product.sizes or [],
                        "style_tags": product.style_tags or [],
                        "material": product.material,
                        "fit": product.fit,
                        "price": {
                            "original": int(product.original_price) if product.original_price else 0,
                            "final": int(product.final_price) if product.final_price else 0,
                            "discount_rate": float(product.discount_rate) if product.discount_rate else 0
                        },
                        "rating": {
                            "average": float(product.rating) if product.rating else 0,
                            "count": product.review_count or 0
                        },
                        "popularity": {
                            "view_count": product.view_count or 0,
                            "like_count": product.like_count or 0,
                            "purchase_count": 0
                        },
                        "description": product.description or "",
                        "images": {
                            "main": product.main_image_url,
                            "list": product.detail_images or []
                        },
                        "is_available": product.is_available,
                        "gender": product.gender,
                        "season": product.season,
                        "crawled_at": product.crawled_at.isoformat() if product.crawled_at else None,
                        "updated_at": product.updated_at.isoformat() if product.updated_at else None,
                        "url": product.url,
                        "musinsa_id": product.musinsa_id
                    }
                }
                yield doc
        
        try:
            success, failed = bulk(
                self.es,
                generate_docs(),
                chunk_size=batch_size,
                request_timeout=60
            )
            logger.info(f"Indexed {success} products successfully, {len(failed)} failed")
            return True
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return False
    
    def search_products(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """상품 검색"""
        if not self.es:
            return {"products": [], "total": 0}
            
        search_body = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": [],
                    "should": []
                }
            },
            "sort": [],
            "from": query.get("offset", 0),
            "size": query.get("limit", 20),
            "aggs": {
                "brands": {
                    "terms": {"field": "brand.name.keyword", "size": 20}
                },
                "categories": {
                    "terms": {"field": "category.medium", "size": 20}
                },
                "colors": {
                    "terms": {"field": "colors", "size": 20}
                },
                "price_ranges": {
                    "range": {
                        "field": "price.final",
                        "ranges": [
                            {"to": 50000},
                            {"from": 50000, "to": 100000},
                            {"from": 100000, "to": 200000},
                            {"from": 200000}
                        ]
                    }
                }
            }
        }
        
        # 텍스트 검색
        if query.get("keyword"):
            search_body["query"]["bool"]["must"].append({
                "multi_match": {
                    "query": query["keyword"],
                    "fields": [
                        "name^3",
                        "brand.name^2",
                        "brand.korean_name^2",
                        "description",
                        "style_tags"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
        
        # 필터 조건들
        if query.get("brand"):
            search_body["query"]["bool"]["filter"].append({
                "term": {"brand.name.keyword": query["brand"]}
            })
        
        if query.get("category"):
            search_body["query"]["bool"]["filter"].append({
                "term": {"category.medium": query["category"]}
            })
        
        if query.get("colors"):
            search_body["query"]["bool"]["filter"].append({
                "terms": {"colors": query["colors"]}
            })
        
        if query.get("min_price") or query.get("max_price"):
            price_range = {}
            if query.get("min_price"):
                price_range["gte"] = query["min_price"]
            if query.get("max_price"):
                price_range["lte"] = query["max_price"]
            
            search_body["query"]["bool"]["filter"].append({
                "range": {"price.final": price_range}
            })
        
        if query.get("is_available", True):
            search_body["query"]["bool"]["filter"].append({
                "term": {"is_available": True}
            })
        
        # 정렬
        sort_by = query.get("sort_by", "relevance")
        if sort_by == "price_low":
            search_body["sort"] = [{"price.final": "asc"}]
        elif sort_by == "price_high":
            search_body["sort"] = [{"price.final": "desc"}]
        elif sort_by == "popularity":
            search_body["sort"] = [
                {"popularity.like_count": "desc"},
                {"popularity.view_count": "desc"}
            ]
        elif sort_by == "rating":
            search_body["sort"] = [{"rating.average": "desc"}]
        elif sort_by == "latest":
            search_body["sort"] = [{"crawled_at": "desc"}]
        
        try:
            response = self.es.search(
                index=self.products_index,
                body=search_body
            )
            
            products = []
            for hit in response["hits"]["hits"]:
                product = hit["_source"]
                product["score"] = hit["_score"]
                products.append(product)
            
            return {
                "products": products,
                "total": response["hits"]["total"]["value"],
                "aggregations": response.get("aggregations", {}),
                "took": response["took"]
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"products": [], "total": 0, "error": str(e)}
    
    def get_product_recommendations(self, product_id: int, count: int = 10) -> List[Dict]:
        """상품 추천 (유사 상품)"""
        if not self.es:
            return []
            
        try:
            # 기준 상품 조회
            base_product = self.es.get(index=self.products_index, id=product_id)
            base_source = base_product["_source"]
            
            # More Like This 쿼리
            search_body = {
                "query": {
                    "bool": {
                        "must": {
                            "more_like_this": {
                                "fields": ["name", "brand.name", "style_tags", "category.medium"],
                                "like": [{"_index": self.products_index, "_id": product_id}],
                                "min_term_freq": 1,
                                "min_doc_freq": 1,
                                "max_query_terms": 12
                            }
                        },
                        "filter": [
                            {"term": {"is_available": True}},
                            {"range": {
                                "price.final": {
                                    "gte": base_source["price"]["final"] * 0.5,
                                    "lte": base_source["price"]["final"] * 2.0
                                }
                            }}
                        ],
                        "must_not": {
                            "term": {"id": product_id}
                        }
                    }
                },
                "size": count
            }
            
            response = self.es.search(
                index=self.products_index,
                body=search_body
            )
            
            recommendations = []
            for hit in response["hits"]["hits"]:
                product = hit["_source"]
                product["similarity_score"] = hit["_score"]
                recommendations.append(product)
            
            return recommendations
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return []
    
    def get_trending_products(self, category: str = None, limit: int = 20) -> List[Dict]:
        """인기 상품 조회"""
        if not self.es:
            return []
            
        search_body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"is_available": True}}
                    ]
                }
            },
            "sort": [
                {"popularity.like_count": "desc"},
                {"popularity.view_count": "desc"},
                {"rating.average": "desc"}
            ],
            "size": limit
        }
        
        if category:
            search_body["query"]["bool"]["filter"].append({
                "term": {"category.medium": category}
            })
        
        try:
            response = self.es.search(
                index=self.products_index,
                body=search_body
            )
            
            trending = []
            for hit in response["hits"]["hits"]:
                product = hit["_source"]
                trending.append(product)
            
            return trending
        except Exception as e:
            logger.error(f"Trending products query failed: {e}")
            return []

# 싱글톤 인스턴스
elasticsearch_service = ElasticsearchService()