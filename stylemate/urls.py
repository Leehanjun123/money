"""
URL configuration for stylemate project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

def health_check(request):
    try:
        # 간단한 헬스체크 - 데이터베이스 연결 등 확인하지 않음
        return JsonResponse({
            'status': 'healthy',
            'service': 'StyleMate Backend',
            'version': '1.0.0'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

def api_test(request):
    return JsonResponse({
        'message': 'StyleMate API is working!',
        'status': 'success',
        'version': '1.0.0'
    })

urlpatterns = [
    path('', health_check, name='health_check'),
    path('admin/', admin.site.urls),
    
    # API URLs (temporarily disabled for Railway deployment)  
    # path('api/', include('api.urls')),
    # path('api/accounts/', include('accounts.urls')),
    # path('api/wardrobe/', include('wardrobe.urls')),
    # path('api/shop/', include('crawler.urls')),
    
    # Temporary API test endpoint
    path('api/', api_test, name='api_test'),
    
    # JWT Authentication
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    # Social Authentication
    path('accounts/', include('allauth.urls')),
    
    # API Documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
