# 🚀 Railway 배포 가이드

## 현재 상태
- ✅ CCXT 라이브러리 통합 완료 (100+ 거래소 지원)
- ✅ 실시간 가격 조회 작동 확인
- ✅ WebSocket HTTPS/WSS 호환
- ✅ Railway 헬스체크 최적화

## 배포 파일
- `railway_production_bot.py` - 메인 봇 (CCXT 통합)
- `requirements.txt` - 의존성 (CCXT 포함)
- `railway.toml` - Railway 설정

## Railway 환경변수 설정

### 필수 설정
```
PORT=8000
```

### Binance API (선택사항)
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret_key
USE_TESTNET=false
```
⚠️ API 키 없어도 시뮬레이션 모드로 작동

## 배포 명령어
```bash
git add .
git commit -m "CCXT 통합 완료"
git push
```

## 작동 모드
1. **시뮬레이션** - API 키 없을 때
2. **테스트넷** - USE_TESTNET=true
3. **실거래** - API 키 설정 시

## 현재 기능
- BTC/ETH 실시간 가격
- 자동 거래 전략
- 웹 대시보드
- WebSocket 실시간 업데이트

## 성능 수준
- 현재: 50/100 (CCXT 통합)
- 목표: 70/100 (ML 추가 필요)