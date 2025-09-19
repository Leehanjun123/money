"""
Outfit Image Generator Service
실제 이미지를 조합해서 코디를 생성하는 서비스
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import random
import os

class OutfitImageGenerator:
    def __init__(self):
        self.initialized = True
        
        # 기본 의류 이미지 템플릿 (실제로는 DB에서 가져옴)
        self.clothing_templates = self._load_templates()
        
        # 배경 색상 팔레트
        self.backgrounds = {
            'minimal': '#F5F5F5',
            'warm': '#FFF8F0',
            'cool': '#F0F8FF',
            'dark': '#2C2C2C',
            'pastel': '#FFE4E1'
        }
        
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """의류 템플릿 로드 (실제로는 이미지 DB)"""
        return {
            'tops': [
                {'id': 't1', 'type': 'tshirt', 'color': 'white', 'style': 'casual'},
                {'id': 't2', 'type': 'shirt', 'color': 'blue', 'style': 'business'},
                {'id': 't3', 'type': 'hoodie', 'color': 'gray', 'style': 'streetwear'},
                {'id': 't4', 'type': 'blazer', 'color': 'navy', 'style': 'formal'},
                {'id': 't5', 'type': 'sweater', 'color': 'beige', 'style': 'casual'},
            ],
            'bottoms': [
                {'id': 'b1', 'type': 'jeans', 'color': 'blue', 'style': 'casual'},
                {'id': 'b2', 'type': 'chinos', 'color': 'khaki', 'style': 'smart_casual'},
                {'id': 'b3', 'type': 'shorts', 'color': 'black', 'style': 'sporty'},
                {'id': 'b4', 'type': 'pants', 'color': 'gray', 'style': 'business'},
                {'id': 'b5', 'type': 'skirt', 'color': 'black', 'style': 'elegant'},
            ],
            'shoes': [
                {'id': 's1', 'type': 'sneakers', 'color': 'white', 'style': 'casual'},
                {'id': 's2', 'type': 'loafers', 'color': 'brown', 'style': 'smart_casual'},
                {'id': 's3', 'type': 'boots', 'color': 'black', 'style': 'trendy'},
                {'id': 's4', 'type': 'heels', 'color': 'black', 'style': 'formal'},
                {'id': 's5', 'type': 'sandals', 'color': 'tan', 'style': 'casual'},
            ],
            'accessories': [
                {'id': 'a1', 'type': 'watch', 'color': 'silver', 'style': 'classic'},
                {'id': 'a2', 'type': 'bag', 'color': 'brown', 'style': 'elegant'},
                {'id': 'a3', 'type': 'hat', 'color': 'black', 'style': 'trendy'},
                {'id': 'a4', 'type': 'scarf', 'color': 'gray', 'style': 'classic'},
            ]
        }
    
    async def generate_outfit_image(
        self,
        outfit_data: Dict[str, Any],
        user_photo: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """코디 이미지 생성"""
        
        # 캔버스 생성
        canvas = self._create_canvas()
        
        # 사용자 사진이 있으면 분석
        if user_photo:
            user_items = await self._extract_user_items(user_photo)
        else:
            user_items = None
        
        # 코디 아이템 선택
        selected_items = self._select_outfit_items(outfit_data, user_items)
        
        # 아이템들을 캔버스에 배치
        outfit_image = self._compose_outfit(canvas, selected_items)
        
        # 스타일 정보 추가
        final_image = self._add_style_info(outfit_image, outfit_data)
        
        # 이미지를 base64로 인코딩
        image_base64 = self._image_to_base64(final_image)
        
        return {
            'image': image_base64,
            'items': selected_items,
            'style_notes': self._generate_style_notes(selected_items),
            'shopping_suggestions': self._generate_shopping_suggestions(selected_items)
        }
    
    def _create_canvas(self, size: Tuple[int, int] = (800, 1000)) -> Image.Image:
        """코디 캔버스 생성"""
        # 그라데이션 배경 생성
        canvas = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(canvas)
        
        # 부드러운 그라데이션 효과
        for i in range(size[1]):
            gray_value = 245 + int(10 * (i / size[1]))
            draw.rectangle([0, i, size[0], i+1], fill=(gray_value, gray_value, gray_value))
        
        return canvas
    
    async def _extract_user_items(self, photo: bytes) -> List[Dict]:
        """사용자 사진에서 의류 아이템 추출"""
        # 실제로는 AI 모델로 의류 세그멘테이션
        # 여기서는 간단한 시뮬레이션
        nparr = np.frombuffer(photo, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return []
        
        # 이미지 영역별로 아이템 추출
        h, w = img.shape[:2]
        
        items = []
        
        # 상의 영역
        top_region = img[0:int(h*0.4), :]
        items.append({
            'type': 'user_top',
            'image': top_region,
            'color': self._extract_dominant_color(top_region)
        })
        
        # 하의 영역
        bottom_region = img[int(h*0.4):int(h*0.8), :]
        items.append({
            'type': 'user_bottom',
            'image': bottom_region,
            'color': self._extract_dominant_color(bottom_region)
        })
        
        return items
    
    def _extract_dominant_color(self, img_region: np.ndarray) -> str:
        """주요 색상 추출"""
        # 평균 색상 계산
        avg_color = np.mean(img_region, axis=(0, 1))
        
        # BGR to RGB
        r, g, b = avg_color[2], avg_color[1], avg_color[0]
        
        # 색상 이름 매핑
        if r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > g and r > b:
            return 'red'
        elif g > r and g > b:
            return 'green'
        elif b > r and b > g:
            return 'blue'
        else:
            return 'mixed'
    
    def _select_outfit_items(
        self, 
        outfit_data: Dict[str, Any],
        user_items: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """코디 아이템 선택"""
        selected = []
        
        # 상의 선택
        if outfit_data.get('top'):
            top = self._find_matching_item('tops', outfit_data['top'])
            if top:
                selected.append({**top, 'category': 'top', 'position': 'upper'})
        
        # 하의 선택
        if outfit_data.get('bottom'):
            bottom = self._find_matching_item('bottoms', outfit_data['bottom'])
            if bottom:
                selected.append({**bottom, 'category': 'bottom', 'position': 'lower'})
        
        # 신발 선택
        if outfit_data.get('shoes'):
            shoes = self._find_matching_item('shoes', outfit_data['shoes'])
            if shoes:
                selected.append({**shoes, 'category': 'shoes', 'position': 'feet'})
        
        # 액세서리 선택 (선택적)
        if outfit_data.get('accessories'):
            for acc in outfit_data['accessories']:
                accessory = self._find_matching_item('accessories', acc)
                if accessory:
                    selected.append({**accessory, 'category': 'accessory', 'position': 'accent'})
        
        # 사용자 아이템이 있으면 일부 대체
        if user_items:
            selected = self._integrate_user_items(selected, user_items)
        
        return selected
    
    def _find_matching_item(self, category: str, criteria: Dict) -> Optional[Dict]:
        """조건에 맞는 아이템 찾기"""
        if category not in self.clothing_templates:
            return None
        
        items = self.clothing_templates[category]
        
        # 조건에 맞는 아이템 필터링
        matches = []
        for item in items:
            if criteria.get('type') and item['type'] == criteria['type']:
                matches.append(item)
            elif criteria.get('style') and item['style'] == criteria['style']:
                matches.append(item)
            elif criteria.get('color') and item['color'] == criteria['color']:
                matches.append(item)
        
        if matches:
            return random.choice(matches)
        
        # 매칭되는 것이 없으면 랜덤 선택
        return random.choice(items)
    
    def _integrate_user_items(self, selected: List[Dict], user_items: List[Dict]) -> List[Dict]:
        """사용자 아이템 통합"""
        # 일부 아이템을 사용자 아이템으로 대체
        for user_item in user_items[:1]:  # 최대 1개만 사용
            if user_item['type'] == 'user_top':
                # 상의를 사용자 아이템으로 대체
                selected = [item for item in selected if item.get('category') != 'top']
                selected.append({
                    'type': 'custom_top',
                    'category': 'top',
                    'position': 'upper',
                    'is_user_item': True,
                    'image_data': user_item['image']
                })
        
        return selected
    
    def _compose_outfit(self, canvas: Image.Image, items: List[Dict]) -> Image.Image:
        """아이템들을 캔버스에 배치"""
        draw = ImageDraw.Draw(canvas)
        
        # 캔버스 크기
        w, h = canvas.size
        
        # 각 아이템을 위치에 따라 배치
        for item in items:
            position = item.get('position')
            
            if position == 'upper':
                # 상의 그리기
                self._draw_clothing_item(draw, item, (w//2, h//4), size='large')
            elif position == 'lower':
                # 하의 그리기
                self._draw_clothing_item(draw, item, (w//2, h//2), size='large')
            elif position == 'feet':
                # 신발 그리기
                self._draw_clothing_item(draw, item, (w//2, 3*h//4), size='medium')
            elif position == 'accent':
                # 액세서리 그리기
                self._draw_clothing_item(draw, item, (3*w//4, h//4), size='small')
        
        return canvas
    
    def _draw_clothing_item(
        self, 
        draw: ImageDraw.Draw,
        item: Dict,
        position: Tuple[int, int],
        size: str = 'medium'
    ):
        """의류 아이템 그리기"""
        x, y = position
        
        # 크기 설정
        sizes = {
            'small': (80, 80),
            'medium': (150, 150),
            'large': (200, 250)
        }
        
        w, h = sizes[size]
        
        # 아이템 타입별 모양 그리기
        item_type = item.get('type', 'unknown')
        color = self._get_color_rgb(item.get('color', 'gray'))
        
        if 'shirt' in item_type or 'tshirt' in item_type:
            # 티셔츠 모양
            points = [
                (x - w//2, y - h//2),
                (x - w//3, y - h//2),
                (x - w//3, y - h//3),
                (x, y - h//3),
                (x + w//3, y - h//3),
                (x + w//3, y - h//2),
                (x + w//2, y - h//2),
                (x + w//2, y + h//2),
                (x - w//2, y + h//2)
            ]
            draw.polygon(points, fill=color, outline='black')
            
        elif 'pants' in item_type or 'jeans' in item_type:
            # 바지 모양
            draw.rectangle([x - w//3, y - h//2, x - 10, y + h//2], fill=color, outline='black')
            draw.rectangle([x + 10, y - h//2, x + w//3, y + h//2], fill=color, outline='black')
            draw.rectangle([x - w//3, y - h//2, x + w//3, y - h//3], fill=color, outline='black')
            
        elif 'skirt' in item_type:
            # 스커트 모양
            points = [
                (x - w//4, y - h//2),
                (x + w//4, y - h//2),
                (x + w//2, y + h//2),
                (x - w//2, y + h//2)
            ]
            draw.polygon(points, fill=color, outline='black')
            
        elif 'shoes' in item_type or 'sneakers' in item_type:
            # 신발 모양
            draw.ellipse([x - w//2, y - h//3, x + w//2, y + h//3], fill=color, outline='black')
            
        else:
            # 기본 사각형
            draw.rectangle([x - w//2, y - h//2, x + w//2, y + h//2], fill=color, outline='black')
        
        # 아이템 라벨
        try:
            # 기본 폰트 사용
            text = item.get('type', '').replace('_', ' ').title()
            draw.text((x, y + h//2 + 20), text, fill='black', anchor='mm')
        except:
            pass
    
    def _get_color_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """색상 이름을 RGB로 변환"""
        colors = {
            'white': (255, 255, 255),
            'black': (40, 40, 40),
            'gray': (128, 128, 128),
            'red': (220, 60, 60),
            'blue': (60, 120, 220),
            'navy': (30, 60, 120),
            'green': (60, 180, 60),
            'yellow': (250, 220, 60),
            'beige': (245, 220, 180),
            'brown': (139, 90, 43),
            'khaki': (189, 183, 107),
            'tan': (210, 180, 140)
        }
        
        return colors.get(color_name, (128, 128, 128))
    
    def _add_style_info(self, image: Image.Image, outfit_data: Dict) -> Image.Image:
        """스타일 정보 추가"""
        draw = ImageDraw.Draw(image)
        
        # 상단에 타이틀 추가
        title = outfit_data.get('occasion', 'Daily Look').upper()
        try:
            draw.text((image.width//2, 30), title, fill='black', anchor='mm')
        except:
            pass
        
        # 하단에 스타일 팁 추가
        if outfit_data.get('styling_tips'):
            y_pos = image.height - 50
            for tip in outfit_data['styling_tips'][:2]:
                try:
                    draw.text((image.width//2, y_pos), f"💡 {tip}", fill='gray', anchor='mm')
                    y_pos -= 25
                except:
                    pass
        
        return image
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """이미지를 base64로 인코딩"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _generate_style_notes(self, items: List[Dict]) -> List[str]:
        """스타일 노트 생성"""
        notes = []
        
        # 색상 조합 평가
        colors = [item.get('color') for item in items if item.get('color')]
        if len(set(colors)) <= 2:
            notes.append("✨ 색상 조합이 깔끔하고 세련되어 보여요")
        else:
            notes.append("🎨 다양한 색상이 포인트가 되는 룩이에요")
        
        # 스타일 평가
        styles = [item.get('style') for item in items if item.get('style')]
        if 'casual' in styles:
            notes.append("👕 편안하면서도 스타일리시한 캐주얼 룩")
        elif 'business' in styles:
            notes.append("💼 전문적이고 신뢰감 있는 비즈니스 스타일")
        elif 'trendy' in styles:
            notes.append("🔥 최신 트렌드를 반영한 패셔너블한 스타일")
        
        # 계절/날씨 팁
        notes.append("🌡️ 오늘 날씨에 딱 맞는 코디예요")
        
        return notes
    
    def _generate_shopping_suggestions(self, items: List[Dict]) -> List[Dict]:
        """쇼핑 제안 생성"""
        suggestions = []
        
        for item in items:
            suggestion = {
                'item': item.get('type', 'item'),
                'color': item.get('color', 'various'),
                'price_range': self._estimate_price_range(item),
                'brands': self._suggest_brands(item)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _estimate_price_range(self, item: Dict) -> str:
        """가격대 추정"""
        item_type = item.get('type', '')
        
        if 'shirt' in item_type or 'tshirt' in item_type:
            return "30,000 - 80,000원"
        elif 'pants' in item_type or 'jeans' in item_type:
            return "50,000 - 150,000원"
        elif 'shoes' in item_type:
            return "80,000 - 200,000원"
        elif 'blazer' in item_type:
            return "100,000 - 300,000원"
        else:
            return "가격 다양"
    
    def _suggest_brands(self, item: Dict) -> List[str]:
        """브랜드 제안"""
        style = item.get('style', 'casual')
        
        brand_map = {
            'casual': ['유니클로', '자라', 'H&M', '무신사 스탠다드'],
            'business': ['자라', '코스', '맥스마라', '휴고보스'],
            'streetwear': ['스투시', '슈프림', '아디다스', '나이키'],
            'trendy': ['자라', '코스', '아더에러', 'COS'],
            'elegant': ['마쉐', '미쏘', '올리비아로렌', '질스튜어트']
        }
        
        return brand_map.get(style, ['다양한 브랜드'])