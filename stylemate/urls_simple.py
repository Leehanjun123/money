"""
간단한 URL 설정 - Railway 배포 테스트용
"""
from django.contrib import admin
from django.urls import path
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({
        'status': 'healthy',
        'service': 'StyleMate Backend Simple',
        'version': '1.0.0'
    })

def api_test(request):
    return JsonResponse({
        'message': 'StyleMate API is working!',
        'status': 'success'
    })

urlpatterns = [
    path('', health_check, name='health_check'),
    path('admin/', admin.site.urls),
    path('api/', api_test, name='api_test'),
]