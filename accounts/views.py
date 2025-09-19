from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate

class RegisterView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        username = request.data.get('username')
        email = request.data.get('email')
        password = request.data.get('password')
        
        if not username or not email or not password:
            return Response(
                {'error': '모든 필드를 입력해주세요'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if User.objects.filter(username=username).exists():
            return Response(
                {'error': '이미 존재하는 사용자명입니다'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if User.objects.filter(email=email).exists():
            return Response(
                {'error': '이미 존재하는 이메일입니다'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'message': '회원가입 성공',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token)
            }
        }, status=status.HTTP_201_CREATED)

class ProfileView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        return Response({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'date_joined': user.date_joined
        })
    
    def put(self, request):
        user = request.user
        email = request.data.get('email')
        
        if email:
            if User.objects.filter(email=email).exclude(id=user.id).exists():
                return Response(
                    {'error': '이미 존재하는 이메일입니다'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            user.email = email
            user.save()
        
        return Response({
            'message': '프로필 업데이트 성공',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })

class SocialLoginView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        provider = request.data.get('provider')  # 'google' or 'kakao'
        token = request.data.get('token')
        
        # 소셜 로그인 로직 구현 예정
        # 실제로는 provider별 SDK를 사용하여 토큰 검증
        
        return Response({
            'message': f'{provider} 로그인 구현 예정',
            'status': 'pending'
        })
