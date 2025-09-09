#!/bin/bash

# HYPERSONIC TRADER 설치 스크립트
# 초격차 트레이딩 시스템 완전 자동 설치

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           HYPERSONIC TRADER - 초격차 시스템 설치                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# 1. 시스템 체크
echo "🔍 시스템 체크 중..."

# GPU 체크
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 감지됨"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️  GPU 없음 - CPU 모드로 실행됩니다"
fi

# Python 버전 체크
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if (( $(echo "$PYTHON_VERSION >= 3.10" | bc -l) )); then
    echo "✅ Python $PYTHON_VERSION 확인"
else
    echo "❌ Python 3.10+ 필요합니다"
    exit 1
fi

# 2. 가상환경 생성
echo "🔧 가상환경 생성 중..."
python3 -m venv hypersonic_env
source hypersonic_env/bin/activate

# 3. 기본 패키지 업그레이드
pip install --upgrade pip setuptools wheel

# 4. CUDA 체크 및 PyTorch 설치
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU 가속 PyTorch 설치 중..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 CPU PyTorch 설치 중..."
    pip install torch torchvision torchaudio
fi

# 5. 나머지 의존성 설치
echo "📦 의존성 패키지 설치 중..."
pip install -r requirements.txt

# 6. Redis 설치 체크
if command -v redis-server &> /dev/null; then
    echo "✅ Redis 설치됨"
else
    echo "📥 Redis 설치 중..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install redis
        brew services start redis
    else
        # Linux
        sudo apt-get update
        sudo apt-get install -y redis-server
        sudo systemctl start redis
    fi
fi

# 7. 환경 변수 설정
echo "🔐 환경 변수 설정 중..."
cat > .env << EOL
# API Keys (실제 키로 교체하세요)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Exchange API Keys
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret

# Network Settings
REDIS_HOST=localhost
REDIS_PORT=6379

# Trading Settings
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
RISK_LIMIT=0.02
ENABLE_PAPER_TRADING=true

# MEV Settings
ETHEREUM_RPC=wss://mainnet.infura.io/ws/v3/your_infura_key
FLASHBOTS_RELAY=https://relay.flashbots.net
EOL

echo "⚠️  .env 파일을 열어 API 키를 설정하세요"

# 8. 데이터 디렉토리 생성
echo "📁 데이터 디렉토리 생성 중..."
mkdir -p data/historical
mkdir -p data/models
mkdir -p logs
mkdir -p backtest_results

# 9. 시스템 서비스 생성 (선택사항)
echo "🔧 시스템 서비스 생성..."
cat > hypersonic.service << EOL
[Unit]
Description=Hypersonic Trader
After=network.target redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/hypersonic_env/bin"
ExecStart=$(pwd)/hypersonic_env/bin/python hypersonic_implementation.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

echo "✅ 서비스 파일 생성됨 (sudo systemctl enable $(pwd)/hypersonic.service 로 활성화)"

# 10. 테스트 실행
echo "🧪 시스템 테스트 중..."
python3 << EOF
import torch
import ccxt
import redis
import fastapi

print("✅ PyTorch:", torch.__version__)
if torch.cuda.is_available():
    print("✅ CUDA 사용 가능:", torch.cuda.get_device_name(0))
else:
    print("⚠️  CUDA 사용 불가 - CPU 모드")

print("✅ CCXT 버전:", ccxt.__version__)
print("✅ FastAPI 준비됨")

# Redis 연결 테스트
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("✅ Redis 연결 성공")
except:
    print("❌ Redis 연결 실패 - redis-server 실행 확인")

print("\n🎉 모든 컴포넌트 준비 완료!")
EOF

# 11. 실행 스크립트 생성
cat > start_hypersonic.sh << 'EOL'
#!/bin/bash
source hypersonic_env/bin/activate

echo "🚀 Hypersonic Trader 시작 중..."

# Redis 시작
redis-server --daemonize yes

# 메인 시스템 시작
python hypersonic_implementation.py &
MAIN_PID=$!

# 대시보드 URL 출력
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Hypersonic Trader 실행 중"
echo "  대시보드: http://localhost:8000"
echo "  API 문서: http://localhost:8000/docs"
echo "  WebSocket: ws://localhost:8000/ws"
echo "  종료: Ctrl+C"
echo "════════════════════════════════════════════════════════"
echo ""

# 종료 처리
trap "kill $MAIN_PID; redis-cli shutdown; exit" INT TERM
wait $MAIN_PID
EOL

chmod +x start_hypersonic.sh

# 12. 간단한 전략 예제 생성
cat > example_strategy.py << 'EOL'
"""
Hypersonic Trader 전략 예제
"""

import asyncio
from hypersonic_implementation import HypersonicTradingSystem

async def momentum_strategy(system):
    """
    간단한 모멘텀 전략 예제
    """
    while True:
        # 시장 데이터 가져오기
        market_data = await system._collect_market_data()
        
        # 20일 이동평균 계산
        ma20 = market_data['prices'][-20:].mean()
        current_price = market_data['prices'][-1]
        
        # 매매 신호
        if current_price > ma20 * 1.02:  # 2% 이상 상승
            print(f"📈 매수 신호: {current_price}")
            # 매수 주문
            order = {
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "market",
                "amount": 0.1
            }
            await system.network.optimized_order_routing(order)
            
        elif current_price < ma20 * 0.98:  # 2% 이상 하락
            print(f"📉 매도 신호: {current_price}")
            # 매도 주문
            order = {
                "symbol": "BTC/USDT",
                "side": "sell",
                "type": "market",
                "amount": 0.1
            }
            await system.network.optimized_order_routing(order)
        
        await asyncio.sleep(60)  # 1분마다 체크

if __name__ == "__main__":
    system = HypersonicTradingSystem()
    asyncio.run(momentum_strategy(system))
EOL

# 13. 완료 메시지
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ 설치 완료!                                  ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  다음 단계:                                                       ║"
echo "║  1. .env 파일을 열어 API 키 설정                                  ║"
echo "║  2. ./start_hypersonic.sh 로 시스템 시작                          ║"
echo "║  3. http://localhost:8000 접속                                   ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  예제 전략 실행:                                                  ║"
echo "║  python example_strategy.py                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"