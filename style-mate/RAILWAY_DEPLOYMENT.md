# 🚀 Railway 배포 가이드 - Style Mate Backend

## 📋 사전 준비 완료 상태

✅ **완료된 작업:**
- Python FastAPI 백엔드 구현
- OpenCV 기반 이미지 분석
- 코디 추천 알고리즘
- Git 저장소 초기화 및 커밋
- Railway 설정 파일 준비

## 🎯 Railway 배포 단계

### 1. GitHub 저장소 생성

```bash
# GitHub에서 새 저장소 생성 후
cd /Users/leehanjun/Desktop/money/style-mate/backend

# Remote 추가
git remote add origin https://github.com/YOUR_USERNAME/stylemate-backend.git

# Push
git branch -M main
git push -u origin main
```

### 2. Railway 프로젝트 생성

1. [Railway 대시보드](https://railway.app/dashboard) 접속
2. **New Project** 클릭
3. **Deploy from GitHub repo** 선택
4. `stylemate-backend` 저장소 선택

### 3. 환경변수 설정

Railway 대시보드에서 다음 환경변수 추가:

```env
# 필수 환경변수
ENVIRONMENT=production
PORT=8000
SECRET_KEY=your-very-secure-secret-key-here

# 선택적 환경변수 (필요시)
OPENWEATHER_API_KEY=your-api-key
MAX_UPLOAD_SIZE=10485760
```

### 4. 데이터베이스 추가 (선택사항)

```bash
# Railway CLI 사용
railway add postgresql

# 또는 대시보드에서 직접 추가
# + New → Database → PostgreSQL
```

### 5. 배포 모니터링

- **Deployment Logs**: 실시간 로그 확인
- **Metrics**: CPU/Memory 사용량 모니터링
- **Domain**: 자동 생성된 도메인 확인

## 🔧 배포 후 테스트

### API 엔드포인트 테스트

```bash
# 도메인 확인 (예: stylemate.up.railway.app)
RAILWAY_URL="https://your-app.up.railway.app"

# 헬스 체크
curl $RAILWAY_URL/

# API 문서 확인
open $RAILWAY_URL/docs

# 이미지 분석 테스트
curl -X POST $RAILWAY_URL/api/analyze \
  -F "file=@test_image.jpg"

# 코디 추천 테스트
curl -X POST $RAILWAY_URL/api/coordinate \
  -H "Content-Type: application/json" \
  -d '{"occasion":"casual"}'
```

## 📊 성능 최적화

### 현재 최적화 상태
- ✅ OpenCV 경량 버전 사용
- ✅ 메모리 캐싱 구현
- ✅ 이미지 크기 제한 (10MB)
- ✅ 비동기 처리

### 예상 성능
- 응답 시간: < 2초
- 동시 사용자: 100+
- 메모리 사용: < 512MB
- 월 비용: $5-15 (Hobby Plan)

## 🐛 트러블슈팅

### 빌드 실패 시
```bash
# requirements.txt 확인
pip freeze > requirements.txt

# Railway 로그 확인
railway logs
```

### 메모리 초과 시
- 이미지 처리 크기 축소
- 캐시 크기 제한 조정
- Worker 프로세스 수 감소

### 속도 개선
- CDN 활용 (Cloudflare)
- 이미지 사전 처리
- 응답 압축 활성화

## 📱 Flutter 앱 연동

Flutter 앱의 `api_service.dart` 수정:

```dart
// Production URL로 변경
static const String baseUrl = 'https://your-app.up.railway.app/api';
```

## 📈 모니터링

### Railway 대시보드
- CPU 사용률
- 메모리 사용량
- 네트워크 트래픽
- 에러 로그

### 추가 모니터링 (선택사항)
- Sentry (에러 추적)
- Google Analytics (사용자 분석)
- Datadog (상세 메트릭)

## ✅ 배포 체크리스트

- [ ] GitHub 저장소 생성
- [ ] Railway 프로젝트 연결
- [ ] 환경변수 설정
- [ ] 배포 성공 확인
- [ ] API 엔드포인트 테스트
- [ ] Flutter 앱 URL 업데이트
- [ ] 성능 모니터링 설정

## 🎉 배포 완료!

배포가 완료되면:
1. API 문서: `https://your-app.up.railway.app/docs`
2. 헬스체크: `https://your-app.up.railway.app/`
3. Flutter 앱과 연동하여 테스트

## 📞 지원

문제 발생 시:
- Railway Discord 커뮤니티
- GitHub Issues
- Railway 지원 티켓

---

**현재 상태: 로컬 테스트 완료 ✅ | Railway 배포 준비 완료 🚀**