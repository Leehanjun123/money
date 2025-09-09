# Railway 배포 가이드 🚂

## 1. Railway에 프로젝트 배포하기

### 필수 환경변수 설정
Railway 대시보드에서 다음 환경변수들을 설정하세요:

```bash
# 거래소 API 키
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET=your_coinbase_secret_here

# AI API (선택사항)
OPENAI_API_KEY=your_openai_api_key_here

# 데이터베이스 (자동으로 Railway가 생성)
DATABASE_URL=postgresql://... (Railway가 자동 제공)
REDIS_URL=redis://... (Railway가 자동 제공)
```

## 2. Railway 서비스 추가

### PostgreSQL 추가
```bash
railway add postgresql
```

### Redis 추가 (선택사항)
```bash
railway add redis
```

## 3. 배포 파일 확인

배포에 필요한 파일들이 모두 준비되었습니다:
- ✅ `admin_dashboard.py` - 메인 관리 대시보드
- ✅ `railway_trader_fixed.py` - 트레이딩 봇 로직  
- ✅ `requirements.txt` - Python 패키지 의존성
- ✅ `Procfile_fixed` - Railway 시작 명령어
- ✅ `railway.json` - Railway 설정

## 4. 배포 명령어

```bash
# Railway CLI 설치
npm install -g @railway/cli

# 로그인
railway login

# 프로젝트 연결
railway link

# 배포
railway up
```

## 5. 관리자 대시보드 접속

배포가 완료되면 Railway에서 제공하는 URL로 접속할 수 있습니다:

- 📊 **메인 대시보드**: `https://your-app.railway.app/`
- ⚙️ **설정 페이지**: `https://your-app.railway.app/settings`  
- 📋 **로그 페이지**: `https://your-app.railway.app/logs`
- 🏥 **헬스체크**: `https://your-app.railway.app/health`

## 6. 주요 기능

### 실시간 모니터링
- 🔄 봇 시작/중지/재시작
- 📈 실시간 수익 추적
- 🎯 전략별 성과 분석
- ⚠️ 실시간 알림

### 트레이딩 전략
- 📊 **차익거래**: 거래소 간 가격차 활용
- 📉 **변동성 거래**: 급격한 가격 변동 활용  
- 📈 **추세 추종**: 기술적 지표 기반 매매
- 🔄 **그리드 거래**: 구간별 자동 매매

### 리스크 관리
- 💰 **포지션 크기 제한**: 최대 투자금액 설정
- 📈 **수익 목표**: 자동 익절 설정
- 📉 **손절 한계**: 손실 제한 설정
- 🔔 **실시간 알림**: 중요 이벤트 알림

## 7. 문제 해결

### 데이터베이스 연결 오류
- Railway PostgreSQL 서비스가 활성화되어 있는지 확인
- DATABASE_URL 환경변수가 올바르게 설정되어 있는지 확인

### API 키 오류
- 거래소 API 키가 올바르게 입력되었는지 확인
- API 키에 거래 권한이 부여되어 있는지 확인

### 성능 최적화
- 무료 플랜의 경우 리소스 제한이 있을 수 있음
- 필요시 Railway Pro 플랜으로 업그레이드 고려

## 8. 보안 주의사항

⚠️ **중요**: 
- API 키를 소스코드에 직접 작성하지 마세요
- 모든 민감한 정보는 Railway 환경변수로 설정하세요
- 정기적으로 API 키를 갱신하세요
- 거래 권한을 최소한으로 제한하세요

---

🎉 **축하합니다!** 이제 24/7 자동 트레이딩 봇이 Railway에서 실행됩니다!