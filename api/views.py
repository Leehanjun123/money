from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.parsers import MultiPartParser
import cv2
import numpy as np
from PIL import Image
import io

class HealthCheckView(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request):
        return Response({
            'status': 'healthy',
            'service': 'Style Mate API',
            'version': '1.0.0'
        })

class AnalyzeClothingView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]
    
    def post(self, request):
        if 'image' not in request.FILES:
            return Response(
                {'error': '이미지 파일이 필요합니다'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES['image']
        
        # 이미지 분석 로직 (간단한 예시)
        try:
            image = Image.open(image_file)
            image_array = np.array(image)
            
            # 색상 분석
            avg_color = np.mean(image_array, axis=(0, 1))
            dominant_color = self.get_color_name(avg_color)
            
            # AI 분석 결과 (향후 고도화 예정)
            analysis_result = {
                'type': self.detect_clothing_type(image_array),
                'color': dominant_color,
                'style': self.detect_style(image_array),
                'recommendations': self.get_recommendations()
            }
            
            return Response(analysis_result)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get_color_name(self, rgb):
        # 간단한 색상 매핑
        r, g, b = rgb[:3]
        if r > 200 and g > 200 and b > 200:
            return '화이트'
        elif r < 50 and g < 50 and b < 50:
            return '블랙'
        elif r > g and r > b:
            return '레드'
        elif g > r and g > b:
            return '그린'
        elif b > r and b > g:
            return '블루'
        else:
            return '혼합색'
    
    def detect_clothing_type(self, image):
        # 향후 YOLO 등으로 구현
        return '상의'
    
    def detect_style(self, image):
        # 향후 스타일 분류 모델로 구현
        return '캐주얼'
    
    def get_recommendations(self):
        return [
            {'item': '청바지', 'reason': '캐주얼 상의와 잘 어울림'},
            {'item': '스니커즈', 'reason': '편안한 스타일 완성'},
            {'item': '백팩', 'reason': '캐주얼 룩에 실용성 추가'}
        ]
