from django.urls import path
from .views import RegisterView, ProfileView, SocialLoginView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('social-login/', SocialLoginView.as_view(), name='social_login'),
]