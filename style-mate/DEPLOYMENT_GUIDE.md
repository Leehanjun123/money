# Style Mate 배포 가이드 🚀

## 1. Railway 백엔드 배포

### 사전 준비
1. Railway 계정 생성 (이미 유료 결제 완료)
2. GitHub 레포지토리 생성

### 배포 단계

```bash
# 1. Git 초기화
cd style-mate/backend
git init
git add .
git commit -m "Initial backend commit"

# 2. GitHub 레포지토리 연결
git remote add origin https://github.com/YOUR_USERNAME/style-mate-backend.git
git push -u origin main

# 3. Railway에서 배포
# Railway 대시보드에서:
# - New Project → Deploy from GitHub repo
# - style-mate-backend 선택
# - 환경변수 설정 (아래 참고)
```

### 필수 환경변수 설정
```env
DATABASE_URL=postgresql://postgres:password@db.railway.internal:5432/railway
REDIS_URL=redis://default:password@redis.railway.internal:6379
SECRET_KEY=your-secret-key-here
OPENWEATHER_API_KEY=your-api-key
ENVIRONMENT=production
```

### Railway 서비스 추가
```bash
# PostgreSQL 추가
railway add postgresql

# Redis 추가
railway add redis
```

## 2. Flutter 앱 빌드

### iOS 빌드
```bash
cd style-mate/frontend

# 의존성 설치
flutter pub get

# iOS 빌드
flutter build ios --release

# Xcode에서 Archive 후 TestFlight 업로드
```

### Android 빌드
```bash
# AAB 빌드 (Google Play용)
flutter build appbundle --release

# APK 빌드 (테스트용)
flutter build apk --release
```

### API URL 설정
`lib/services/api_service.dart` 파일에서:
```dart
// Production URL로 변경
static const String baseUrl = 'https://stylemate.up.railway.app/api';
```

## 3. 로컬 테스트 방법

### 백엔드 테스트
```bash
cd style-mate/backend

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# 로컬 실행
uvicorn main:app --reload --port 8000

# API 테스트
curl http://localhost:8000/
```

### Flutter 앱 테스트
```bash
cd style-mate/frontend

# 의존성 설치
flutter pub get

# 개발 모드 실행
flutter run

# 특정 디바이스 선택
flutter devices  # 디바이스 목록 확인
flutter run -d iPhone_15_Pro  # iOS 시뮬레이터
flutter run -d chrome  # 웹 브라우저
```

## 4. AI 모델 최적화

### 모델 다운로드 (첫 실행 시)
```python
# 백엔드 서버 첫 실행 시 자동으로 다운로드
# YOLOv8n: ~6MB
# Segformer-b0: ~14MB
```

### 성능 모니터링
```bash
# Railway 대시보드에서:
# - Metrics → CPU/Memory 사용량 확인
# - Logs → 실시간 로그 확인

# 로컬 테스트
python -m pytest tests/ -v
```

## 5. 베타 테스트 배포

### TestFlight (iOS)
1. App Store Connect 로그인
2. 새 앱 생성
3. TestFlight 탭에서 빌드 업로드
4. 베타 테스터 초대 링크 생성

### Google Play Console (Android)
1. Play Console 로그인
2. 앱 생성
3. Internal Testing 트랙에 AAB 업로드
4. 테스터 이메일 추가

### 베타 테스터 모집
- 에브리타임/블라인드에 게시
- 피드백 폼 준비 (Google Forms)
- Discord/Slack 채널 운영

## 6. 모니터링 & 분석

### 에러 추적
```bash
# Sentry 설정 (선택사항)
pip install sentry-sdk
```

### 사용자 분석
- Google Analytics 설정
- 사용자 행동 패턴 추적
- A/B 테스트 진행

## 7. 체크리스트

### 배포 전 체크리스트
- [ ] 환경변수 모두 설정했나?
- [ ] API URL 프로덕션으로 변경했나?
- [ ] 이미지 업로드 크기 제한 설정했나? (10MB)
- [ ] Redis 캐싱 동작하나?
- [ ] CORS 설정 확인했나?
- [ ] SSL 인증서 적용됐나?
- [ ] 에러 로깅 설정했나?

### 성능 목표
- [ ] 이미지 분석: < 3초
- [ ] API 응답: < 1초
- [ ] 앱 시작: < 2초
- [ ] 메모리 사용: < 200MB

## 8. 트러블슈팅

### Railway 배포 실패
```bash
# 로그 확인
railway logs

# 빌드 다시 시도
railway up --detach
```

### Flutter 빌드 에러
```bash
# 캐시 클리어
flutter clean
flutter pub get
flutter build ios --release
```

### AI 모델 로딩 실패
```python
# 수동 다운로드
from ultralytics import YOLO
YOLO('yolov8n.pt')  # 자동 다운로드
```

## 9. 비용 관리

### Railway 예상 비용
- Hobby Plan: $5/월 (이미 결제)
- 추가 사용량: ~$10-20/월
- 총 예상: $15-25/월

### 비용 절감 팁
1. 이미지 리사이징 (클라이언트)
2. 적극적인 캐싱 활용
3. 오프피크 시간대 배치 작업

## 10. 연락처

문제 발생 시:
- GitHub Issues 생성
- Railway Support 문의
- Flutter Discord 커뮤니티

---

🎉 **성공적인 런칭을 위해 화이팅!**