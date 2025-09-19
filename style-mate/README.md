# Style Mate - AI 기반 패션 코디 추천 앱

## 🎯 Vision
업계 최고의 AI 패션 코디 추천 서비스

## 🏗️ Architecture

```
style-mate/
├── frontend/          # Flutter 모바일 앱
├── backend/           # Python FastAPI + Railway
├── ai-models/         # AI 모델 서빙
├── infrastructure/    # Docker, Redis 설정
└── docs/             # 문서
```

## 🛠️ Tech Stack

### Frontend
- Flutter 3.x
- Provider (상태관리)
- Camera/Gallery 플러그인
- Local Storage (Hive)

### Backend  
- Python FastAPI
- PostgreSQL (Railway)
- Redis (캐싱)
- AWS S3 / Cloudinary (이미지)

### AI/ML
- YOLOv8n (의류 감지)
- Segformer-b0 (세그멘테이션)
- FashionCLIP (스타일 분석)
- TensorFlow Lite (모바일 최적화)

### Infrastructure
- Railway (백엔드 호스팅)
- Cloudflare (CDN)
- GitHub Actions (CI/CD)

## 📱 Core Features

1. **Smart Closet**
   - 사진 10장으로 시작
   - 자동 배경 제거
   - 카테고리 자동 분류

2. **AI Coordination**
   - 날씨 기반 추천
   - TPO별 스타일링
   - 개인 취향 학습

3. **Performance**
   - 1-3초 내 분석
   - 오프라인 모드 지원
   - 실시간 동기화

## 🚀 Development Phases

### Phase 1: MVP (2주)
- [ ] 기본 옷장 관리
- [ ] YOLOv8n 의류 인식
- [ ] 날씨 연동 추천

### Phase 2: Enhancement (2주)
- [ ] 멀티 모델 앙상블
- [ ] 스타일 학습
- [ ] 소셜 기능

### Phase 3: Scale (1개월)
- [ ] 쇼핑 연동
- [ ] 프리미엄 기능
- [ ] 글로벌 확장

## 📊 Success Metrics
- 분석 정확도 > 90%
- 응답 시간 < 3초
- DAU/MAU > 40%
- 앱스토어 평점 > 4.5