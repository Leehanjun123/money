import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import List, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusinsaCrawler:
    """무신사 상품 크롤링 클래스"""
    
    BASE_URL = "https://www.musinsa.com"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    CATEGORIES = {
        '001': {'name': '상의', 'subcategories': {
            '001001': '반소매 티셔츠',
            '001002': '긴소매 티셔츠', 
            '001003': '셔츠/블라우스',
            '001004': '니트/스웨터',
            '001005': '후드/맨투맨',
        }},
        '002': {'name': '아우터', 'subcategories': {
            '002001': '코트',
            '002002': '자켓',
            '002003': '패딩',
            '002004': '가디건',
        }},
        '003': {'name': '바지', 'subcategories': {
            '003001': '데님',
            '003002': '슬랙스',
            '003003': '트레이닝',
            '003004': '숏팬츠',
        }},
        '018': {'name': '신발', 'subcategories': {
            '018001': '스니커즈',
            '018002': '구두',
            '018003': '부츠',
            '018004': '샌들/슬리퍼',
        }},
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def get_product_list(self, category: str = '001', page: int = 1, limit: int = 90) -> List[Dict]:
        """상품 리스트 가져오기"""
        
        url = f"{self.BASE_URL}/categories/{category}?page={page}&size={limit}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            products = []
            
            # 상품 리스트 파싱
            items = soup.select('li.li_box')
            
            for item in items:
                try:
                    product = self._parse_product_item(item)
                    if product:
                        products.append(product)
                except Exception as e:
                    logger.error(f"Error parsing product item: {e}")
                    continue
            
            logger.info(f"Crawled {len(products)} products from category {category}, page {page}")
            return products
            
        except Exception as e:
            logger.error(f"Error crawling product list: {e}")
            return []
    
    def _parse_product_item(self, item) -> Optional[Dict]:
        """개별 상품 아이템 파싱"""
        
        try:
            # 상품 ID와 링크
            link_elem = item.select_one('a.img-block')
            if not link_elem:
                return None
            
            product_url = self.BASE_URL + link_elem.get('href', '')
            product_id = product_url.split('/')[-1].split('?')[0]
            
            # 브랜드
            brand_elem = item.select_one('p.item_title a')
            brand = brand_elem.text.strip() if brand_elem else 'Unknown'
            
            # 상품명
            name_elem = item.select_one('p.list_info a')
            name = name_elem.get('title', '').strip() if name_elem else ''
            
            # 이미지
            img_elem = item.select_one('img.lazyload')
            main_image = img_elem.get('data-original', '') if img_elem else ''
            
            # 가격
            price_elem = item.select_one('p.price')
            original_price = 0
            sale_price = None
            
            if price_elem:
                # 할인 가격이 있는 경우
                sale_elem = price_elem.select_one('del')
                if sale_elem:
                    original_price = self._parse_price(sale_elem.text)
                    current_price_text = price_elem.text.replace(sale_elem.text, '').strip()
                    sale_price = self._parse_price(current_price_text)
                else:
                    original_price = self._parse_price(price_elem.text)
            
            # 평점과 리뷰
            rating_elem = item.select_one('span.rating')
            rating = float(rating_elem.text.strip()) if rating_elem else 0.0
            
            review_elem = item.select_one('span.count')
            review_count = self._parse_number(review_elem.text) if review_elem else 0
            
            # 좋아요
            like_elem = item.select_one('span.n_like')
            like_count = self._parse_number(like_elem.text) if like_elem else 0
            
            return {
                'product_id': product_id,
                'source': 'musinsa',
                'name': name,
                'brand': brand,
                'product_url': product_url,
                'main_image_url': main_image,
                'original_price': original_price,
                'sale_price': sale_price,
                'discount_rate': self._calculate_discount_rate(original_price, sale_price),
                'rating': rating,
                'review_count': review_count,
                'like_count': like_count,
                'crawled_at': datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error parsing product item: {e}")
            return None
    
    def get_product_detail(self, product_url: str) -> Optional[Dict]:
        """상품 상세 정보 가져오기"""
        
        try:
            response = self.session.get(product_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 상세 정보 파싱
            detail = {}
            
            # 색상 옵션
            color_elems = soup.select('select#color_option option')
            colors = [elem.text.strip() for elem in color_elems if elem.text.strip()]
            detail['colors'] = colors
            
            # 사이즈 옵션
            size_elems = soup.select('select#size_option option')
            sizes = [elem.text.strip() for elem in size_elems if elem.text.strip()]
            detail['sizes'] = sizes
            
            # 상품 설명
            desc_elem = soup.select_one('div.product_detail_contents')
            if desc_elem:
                detail['description'] = desc_elem.get_text(strip=True)
            
            # 소재 정보
            material_elem = soup.select_one('li:contains("소재")')
            if material_elem:
                detail['material'] = material_elem.get_text(strip=True).replace('소재', '').strip()
            
            # 추가 이미지
            img_elems = soup.select('div.product_img img')
            detail['image_urls'] = [img.get('src', '') for img in img_elems if img.get('src')]
            
            # 스타일 태그
            tag_elems = soup.select('div.style_tag a')
            detail['style_tags'] = [tag.text.strip() for tag in tag_elems]
            
            return detail
            
        except Exception as e:
            logger.error(f"Error getting product detail: {e}")
            return None
    
    def _parse_price(self, text: str) -> int:
        """가격 텍스트를 숫자로 변환"""
        import re
        numbers = re.findall(r'\d+', text.replace(',', ''))
        return int(''.join(numbers)) if numbers else 0
    
    def _parse_number(self, text: str) -> int:
        """숫자 텍스트를 정수로 변환"""
        import re
        numbers = re.findall(r'\d+', text.replace(',', ''))
        return int(''.join(numbers)) if numbers else 0
    
    def _calculate_discount_rate(self, original: int, sale: Optional[int]) -> int:
        """할인율 계산"""
        if sale and original > 0:
            return int((1 - sale / original) * 100)
        return 0
    
    def crawl_category(self, category: str, max_pages: int = 5) -> List[Dict]:
        """특정 카테고리의 상품들을 크롤링"""
        
        all_products = []
        
        for page in range(1, max_pages + 1):
            products = self.get_product_list(category, page)
            
            if not products:
                logger.info(f"No more products in category {category}")
                break
            
            all_products.extend(products)
            
            # 페이지 간 딜레이
            time.sleep(random.uniform(1, 3))
        
        logger.info(f"Total {len(all_products)} products crawled from category {category}")
        return all_products
    
    def crawl_all_categories(self, max_pages_per_category: int = 3) -> Dict[str, List[Dict]]:
        """모든 카테고리 크롤링"""
        
        results = {}
        
        for category_id, category_info in self.CATEGORIES.items():
            logger.info(f"Starting to crawl category: {category_info['name']}")
            
            products = self.crawl_category(category_id, max_pages_per_category)
            results[category_info['name']] = products
            
            # 카테고리 간 딜레이
            time.sleep(random.uniform(3, 5))
        
        return results


# 테스트 코드
if __name__ == "__main__":
    crawler = MusinsaCrawler()
    
    # 상의 카테고리 크롤링 테스트
    products = crawler.crawl_category('001', max_pages=1)
    
    if products:
        print(f"Successfully crawled {len(products)} products")
        print(f"Sample product: {json.dumps(products[0], indent=2, ensure_ascii=False)}")
    else:
        print("Failed to crawl products")