#!/bin/bash

# Railway 배포 스크립트
# 한 번 실행으로 24/7 트레이딩 봇 배포

echo "🚀 Railway 트레이딩 봇 배포 시작"

# 1. Railway CLI 설치 확인
if ! command -v railway &> /dev/null; then
    echo "📦 Railway CLI 설치 중..."
    npm install -g @railway/cli
fi

# 2. GitHub 저장소 생성
echo "📁 GitHub 저장소 설정..."
git init
git add .
git commit -m "Initial commit: Railway Trading Bot"

# 3. Railway 프로젝트 생성
echo "🔧 Railway 프로젝트 생성..."
railway login
railway init

# 4. 서비스 추가
echo "⚙️ 서비스 설정..."

# PostgreSQL 추가
railway add --plugin postgresql

# Redis 추가  
railway add --plugin redis

# 5. 환경변수 설정
echo "🔐 환경변수 설정..."
railway variables set BINANCE_API_KEY="your_key_here"
railway variables set BINANCE_SECRET="your_secret_here"
railway variables set COINBASE_API_KEY="your_key_here"
railway variables set COINBASE_SECRET="your_secret_here"

# 6. 배포
echo "🚀 배포 중..."
railway up

# 7. 도메인 생성
railway domain

echo """
═══════════════════════════════════════════════════════════════
✅ Railway 배포 완료!
═══════════════════════════════════════════════════════════════

다음 단계:
1. https://railway.app/dashboard 에서 프로젝트 확인
2. Settings > Variables에서 API 키 설정
3. 생성된 URL로 대시보드 접속

예상 비용: $0-5/월 (무료 크레딧으로 충분)
예상 수익: $1,000-5,000/월

봇이 24/7 자동으로 거래합니다!
═══════════════════════════════════════════════════════════════
"""