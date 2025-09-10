#!/bin/bash

# Freqtrade - 전문가급 오픈소스 트레이딩 봇 설치
# 100+ 전략, 백테스팅, 실거래, ML 지원

echo "🚀 Freqtrade 설치 시작..."

# 1. Freqtrade 클론
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade

# 2. 설치
./setup.sh -i

# 3. 설정 파일 생성
freqtrade create-userdir --userdir user_data

# 4. 전략 다운로드
freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT

echo "✅ Freqtrade 설치 완료!"
echo "📊 백테스팅: freqtrade backtesting --strategy SampleStrategy"
echo "💰 실거래: freqtrade trade --strategy SampleStrategy"