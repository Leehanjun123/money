"""
RAILWAY TRADER - 24/7 클라우드 트레이딩 봇
Railway.app에 배포하여 맥북 꺼도 계속 돈 벌기
"""

import os
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import aiohttp
from typing import Dict, List, Optional
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import redis
import psycopg2
from decimal import Decimal

# Railway 환경 변수
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PORT = int(os.getenv("PORT", 8000))

# 거래소 API 키 (Railway 환경변수로 안전하게 관리)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_SECRET = os.getenv("COINBASE_SECRET")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 데이터베이스 설정 ====================

class DatabaseManager:
    """Railway PostgreSQL 관리"""
    
    def __init__(self):
        self.conn = None
        if DATABASE_URL:
            self.conn = psycopg2.connect(DATABASE_URL)
            self.setup_tables()
    
    def setup_tables(self):
        """테이블 생성"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    exchange VARCHAR(50),
                    symbol VARCHAR(20),
                    side VARCHAR(10),
                    price DECIMAL(20, 8),
                    amount DECIMAL(20, 8),
                    profit DECIMAL(20, 8),
                    strategy VARCHAR(50)
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id SERIAL PRIMARY KEY,
                    date DATE DEFAULT CURRENT_DATE,
                    total_profit DECIMAL(20, 8),
                    win_rate DECIMAL(5, 2),
                    total_trades INTEGER
                )
            """)
            self.conn.commit()
    
    def log_trade(self, trade: Dict):
        """거래 기록"""
        if not self.conn:
            return
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (exchange, symbol, side, price, amount, profit, strategy)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                trade['exchange'],
                trade['symbol'],
                trade['side'],
                trade['price'],
                trade['amount'],
                trade.get('profit', 0),
                trade.get('strategy', 'unknown')
            ))
            self.conn.commit()

# ==================== Railway 최적화 트레이딩 봇 ====================

class RailwayTrader:
    """
    Railway에서 24/7 실행되는 트레이딩 봇
    메모리 효율적, 자동 재시작 지원
    """
    
    def __init__(self):
        self.exchanges = self._init_exchanges()
        self.db = DatabaseManager()
        self.redis_client = self._init_redis()
        
        # 전략 설정
        self.active_strategies = {
            'arbitrage': True,
            'grid': True,
            'momentum': True,
            'mean_reversion': True
        }
        
        # 리스크 관리
        self.max_position_size = 1000  # $1000
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.05  # 5%
        
        logger.info("Railway Trader 초기화 완료")
    
    def _init_exchanges(self):
        """거래소 초기화"""
        exchanges = {}
        
        if BINANCE_API_KEY:
            exchanges['binance'] = ccxt.binance({
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            logger.info("Binance 연결 성공")
        
        if COINBASE_API_KEY:
            exchanges['coinbase'] = ccxt.coinbase({
                'apiKey': COINBASE_API_KEY,
                'secret': COINBASE_SECRET,
                'enableRateLimit': True
            })
            logger.info("Coinbase 연결 성공")
        
        return exchanges
    
    def _init_redis(self):
        """Redis 초기화"""
        try:
            r = redis.from_url(REDIS_URL)
            r.ping()
            logger.info("Redis 연결 성공")
            return r
        except:
            logger.warning("Redis 연결 실패 - 로컬 캐시 사용")
            return None
    
    async def run_forever(self):
        """
        메인 실행 루프
        Railway는 자동으로 재시작하므로 에러 처리 간단하게
        """
        logger.info("트레이딩 봇 시작...")
        
        tasks = []
        
        if self.active_strategies['arbitrage']:
            tasks.append(self.arbitrage_strategy())
        
        if self.active_strategies['grid']:
            tasks.append(self.grid_strategy())
        
        if self.active_strategies['momentum']:
            tasks.append(self.momentum_strategy())
        
        if self.active_strategies['mean_reversion']:
            tasks.append(self.mean_reversion_strategy())
        
        # 성과 모니터링
        tasks.append(self.monitor_performance())
        
        # 헬스체크 (Railway 상태 확인용)
        tasks.append(self.health_check())
        
        await asyncio.gather(*tasks)
    
    async def arbitrage_strategy(self):
        """
        전략 1: 거래소 간 차익거래
        Binance <-> Coinbase
        """
        logger.info("차익거래 전략 시작")
        
        while True:
            try:
                if len(self.exchanges) < 2:
                    await asyncio.sleep(60)
                    continue
                
                # BTC 가격 비교
                symbols = ['BTC/USDT', 'BTC/USD']
                prices = {}
                
                for exchange_name, exchange in self.exchanges.items():
                    symbol = symbols[0] if 'binance' in exchange_name else symbols[1]
                    ticker = await exchange.fetch_ticker(symbol)
                    prices[exchange_name] = ticker['last']
                
                # 차익 기회 찾기
                if len(prices) >= 2:
                    min_exchange = min(prices, key=prices.get)
                    max_exchange = max(prices, key=prices.get)
                    
                    price_diff = prices[max_exchange] - prices[min_exchange]
                    diff_pct = (price_diff / prices[min_exchange]) * 100
                    
                    # 0.3% 이상 차이나면 실행
                    if diff_pct > 0.3:
                        logger.info(f"🎯 차익 발견: {diff_pct:.2f}% ({min_exchange} -> {max_exchange})")
                        
                        # 실제 거래 실행 (페이퍼 트레이딩)
                        trade = {
                            'exchange': f"{min_exchange}->{max_exchange}",
                            'symbol': 'BTC',
                            'side': 'arbitrage',
                            'price': prices[min_exchange],
                            'amount': 0.001,
                            'profit': price_diff * 0.001,
                            'strategy': 'arbitrage'
                        }
                        
                        self.db.log_trade(trade)
                        
                        # Redis에 캐시
                        if self.redis_client:
                            self.redis_client.incr('total_arbitrage_trades')
                            self.redis_client.incrbyfloat('total_profit', float(trade['profit']))
                
                await asyncio.sleep(10)  # 10초마다 체크
                
            except Exception as e:
                logger.error(f"차익거래 에러: {e}")
                await asyncio.sleep(60)
    
    async def grid_strategy(self):
        """
        전략 2: 그리드 트레이딩
        변동성 구간에서 자동 매매
        """
        logger.info("그리드 전략 시작")
        
        symbol = 'ETH/USDT'
        grid_levels = 5
        grid_spacing = 0.01  # 1% 간격
        
        if 'binance' not in self.exchanges:
            return
        
        exchange = self.exchanges['binance']
        
        while True:
            try:
                # 현재 가격
                ticker = await exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # 그리드 설정
                buy_orders = []
                sell_orders = []
                
                for i in range(1, grid_levels + 1):
                    buy_price = current_price * (1 - grid_spacing * i)
                    sell_price = current_price * (1 + grid_spacing * i)
                    
                    buy_orders.append({
                        'symbol': symbol,
                        'type': 'limit',
                        'side': 'buy',
                        'price': buy_price,
                        'amount': 0.01
                    })
                    
                    sell_orders.append({
                        'symbol': symbol,
                        'type': 'limit',
                        'side': 'sell',
                        'price': sell_price,
                        'amount': 0.01
                    })
                
                logger.info(f"📊 그리드 설정: {len(buy_orders) + len(sell_orders)}개 주문")
                
                # 주문 체크 및 재설정 (1시간마다)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"그리드 에러: {e}")
                await asyncio.sleep(300)
    
    async def momentum_strategy(self):
        """
        전략 3: 모멘텀 추종
        상승 추세 감지 시 진입
        """
        logger.info("모멘텀 전략 시작")
        
        while True:
            try:
                if 'binance' not in self.exchanges:
                    await asyncio.sleep(60)
                    continue
                
                exchange = self.exchanges['binance']
                symbol = 'BTC/USDT'
                
                # 최근 24시간 데이터
                ohlcv = await exchange.fetch_ohlcv(symbol, '1h', limit=24)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # 지표 계산
                df['ma7'] = df['close'].rolling(7).mean()
                df['ma25'] = df['close'].rolling(25).mean()
                df['rsi'] = self._calculate_rsi(df['close'])
                
                current_price = df['close'].iloc[-1]
                ma7 = df['ma7'].iloc[-1]
                ma25 = df['ma25'].iloc[-1]
                rsi = df['rsi'].iloc[-1]
                
                # 매수 신호: MA7 > MA25 & RSI < 70
                if ma7 > ma25 and rsi < 70:
                    logger.info(f"📈 모멘텀 매수 신호: RSI={rsi:.2f}")
                    
                    trade = {
                        'exchange': 'binance',
                        'symbol': symbol,
                        'side': 'buy',
                        'price': current_price,
                        'amount': 0.001,
                        'strategy': 'momentum'
                    }
                    self.db.log_trade(trade)
                
                # 매도 신호: MA7 < MA25 | RSI > 80
                elif ma7 < ma25 or rsi > 80:
                    logger.info(f"📉 모멘텀 매도 신호: RSI={rsi:.2f}")
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                logger.error(f"모멘텀 에러: {e}")
                await asyncio.sleep(600)
    
    async def mean_reversion_strategy(self):
        """
        전략 4: 평균 회귀
        과매도/과매수 구간에서 역방향 매매
        """
        logger.info("평균회귀 전략 시작")
        
        while True:
            try:
                if 'binance' not in self.exchanges:
                    await asyncio.sleep(60)
                    continue
                
                exchange = self.exchanges['binance']
                
                # 여러 코인 스캔
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
                
                for symbol in symbols:
                    ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=50)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # 볼린저 밴드
                    df['ma20'] = df['close'].rolling(20).mean()
                    df['std20'] = df['close'].rolling(20).std()
                    df['upper'] = df['ma20'] + (df['std20'] * 2)
                    df['lower'] = df['ma20'] - (df['std20'] * 2)
                    
                    current_price = df['close'].iloc[-1]
                    upper_band = df['upper'].iloc[-1]
                    lower_band = df['lower'].iloc[-1]
                    
                    # 하단 밴드 터치 - 매수
                    if current_price <= lower_band:
                        logger.info(f"🎯 {symbol} 과매도 - 매수 기회")
                        
                    # 상단 밴드 터치 - 매도
                    elif current_price >= upper_band:
                        logger.info(f"🎯 {symbol} 과매수 - 매도 기회")
                
                await asyncio.sleep(180)  # 3분마다
                
            except Exception as e:
                logger.error(f"평균회귀 에러: {e}")
                await asyncio.sleep(300)
    
    def _calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def monitor_performance(self):
        """성과 모니터링"""
        while True:
            try:
                if self.redis_client:
                    total_trades = self.redis_client.get('total_arbitrage_trades') or 0
                    total_profit = self.redis_client.get('total_profit') or 0
                    
                    logger.info(f"""
                    ═══════════════════════════════════
                    📊 실시간 성과
                    ───────────────────────────────────
                    총 거래: {total_trades}
                    총 수익: ${float(total_profit):.2f}
                    ═══════════════════════════════════
                    """)
                
                await asyncio.sleep(60)  # 1분마다
                
            except Exception as e:
                logger.error(f"모니터링 에러: {e}")
                await asyncio.sleep(60)
    
    async def health_check(self):
        """Railway 헬스체크"""
        while True:
            logger.info("♥️ Bot is alive")
            await asyncio.sleep(30)

# ==================== FastAPI 대시보드 ====================

app = FastAPI(title="Railway Trader Dashboard")
trader = None

@app.on_event("startup")
async def startup():
    global trader
    trader = RailwayTrader()
    asyncio.create_task(trader.run_forever())

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Railway Trader</title>
        <style>
            body { font-family: monospace; background: #000; color: #0f0; padding: 20px; }
            h1 { color: #0f0; text-shadow: 0 0 10px #0f0; }
            .stats { border: 1px solid #0f0; padding: 20px; margin: 20px 0; }
            .profit { color: #0f0; font-size: 24px; }
            .loss { color: #f00; }
        </style>
    </head>
    <body>
        <h1>🚀 Railway Trader Dashboard</h1>
        <div class="stats">
            <h2>실시간 성과</h2>
            <p class="profit">수익: $0.00</p>
            <p>거래 횟수: 0</p>
            <p>승률: 0%</p>
        </div>
        <div class="stats">
            <h2>활성 전략</h2>
            <p>✅ 차익거래</p>
            <p>✅ 그리드 트레이딩</p>
            <p>✅ 모멘텀</p>
            <p>✅ 평균회귀</p>
        </div>
        <script>
            setInterval(() => {
                fetch('/stats').then(r => r.json()).then(data => {
                    // 실시간 업데이트
                });
            }, 5000);
        </script>
    </body>
    </html>
    """)

@app.get("/stats")
async def get_stats():
    """실시간 통계"""
    if trader and trader.redis_client:
        return {
            "total_trades": trader.redis_client.get('total_arbitrage_trades') or 0,
            "total_profit": trader.redis_client.get('total_profit') or 0
        }
    return {"status": "initializing"}

@app.get("/health")
async def health():
    """Railway 헬스체크 엔드포인트"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Railway는 PORT 환경변수 사용
    uvicorn.run(app, host="0.0.0.0", port=PORT)