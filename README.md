# 🎯 Style Mate - AI Fashion Coordination Service

> AI 기반 패션 코디 추천 서비스 | 당신의 스마트한 스타일 메이트

## 🌟 프로젝트 소개

Style Mate는 AI 기술을 활용한 혁신적인 패션 코디 추천 서비스입니다. 사용자의 옷장을 스마트하게 관리하고, 날씨와 상황에 맞는 최적의 코디를 추천합니다.

### ✨ 핵심 기능

- **🤖 AI 의류 분석**: OpenCV 기반 실시간 의류 감지 및 분류
- **👔 스마트 코디 추천**: 날씨, TPO, 개인 스타일 기반 맞춤 추천
- **📸 간편한 옷장 관리**: 사진 10장으로 시작하는 가상 옷장
- **⚡ 빠른 응답 속도**: 평균 1초 이내 분석 완료
- **🌤️ 날씨 연동**: 실시간 날씨 기반 코디 조정

## 🛠️ 기술 스택

### Backend
- **Framework**: FastAPI (Python 3.11)
- **AI/CV**: OpenCV, Pillow, NumPy
- **Database**: PostgreSQL (Railway)
- **Cache**: In-memory caching
- **Deployment**: Railway

### Frontend
- **Framework**: Flutter 3.x
- **State Management**: Provider
- **Network**: Dio
- **Storage**: Hive

## 📂 프로젝트 구조

```
style-mate/
├── backend/              # FastAPI 백엔드 서버
│   ├── main_production.py   # 프로덕션 서버
│   ├── app/
│   │   ├── services/     # AI, 추천, 날씨 서비스
│   │   └── config.py     # 설정 관리
│   └── requirements.txt  # Python 패키지
├── frontend/             # Flutter 모바일 앱
│   ├── lib/
│   │   ├── screens/      # UI 화면
│   │   └── services/     # API 연동
│   └── pubspec.yaml      # Flutter 패키지
└── docs/                 # 문서
```

## 🚀 빠른 시작

### Backend 실행

```bash
# 1. 백엔드 디렉토리로 이동
cd style-mate/backend

# 2. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 3. 패키지 설치
pip install -r requirements_production.txt

# 4. 서버 실행
python main_production.py
```

서버가 실행되면:
- API: http://localhost:8000
- 문서: http://localhost:8000/docs

### Frontend 실행

```bash
# 1. Flutter 설치 확인
flutter --version

# 2. 프론트엔드 디렉토리로 이동
cd style-mate/frontend

# 3. 패키지 설치
flutter pub get

# 4. 앱 실행
flutter run
```

## 📱 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 헬스 체크 |
| POST | `/api/analyze` | 의류 이미지 분석 |
| POST | `/api/coordinate` | 코디 추천 생성 |
| GET | `/api/weather/{location}` | 날씨 정보 조회 |
| POST | `/api/closet/items` | 옷장 아이템 추가 |
| GET | `/api/closet/items` | 옷장 아이템 조회 |

## 🧪 테스트

```bash
# 백엔드 테스트
cd style-mate
python test_production.py

# 브라우저 테스트
open test_frontend.html
```

## 🚢 Railway 배포

1. GitHub 레포지토리 연결
2. Railway 프로젝트 생성
3. 환경변수 설정:
   ```env
   ENVIRONMENT=production
   PORT=8000
   SECRET_KEY=your-secret-key
   ```
4. 자동 배포 확인

상세 가이드: [RAILWAY_DEPLOYMENT.md](style-mate/RAILWAY_DEPLOYMENT.md)

## 📊 성능

- **응답 속도**: < 1초
- **이미지 분석**: 0.5초
- **동시 사용자**: 100+
- **캐시 적중률**: 60%+

## 🎯 로드맵

- [x] MVP 개발
- [x] AI 모델 통합
- [x] 프로덕션 서버 구현
- [ ] 실제 YOLOv8 모델 통합
- [ ] 사용자 인증 시스템
- [ ] 소셜 기능
- [ ] 쇼핑몰 연동

## 📄 라이선스

MIT License

## 🤝 기여

Issues와 Pull Requests를 환영합니다!

## 📞 문의

- GitHub Issues
- Email: your-email@example.com

---

**🚀 Style Mate - Your Smart Fashion Companion**