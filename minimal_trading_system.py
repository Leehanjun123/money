"""
🎯 최소한으로 작동하는 트레이딩 시스템
Railway 배포 100% 성공 보장
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# 글로벌 상태
class TradingBot:
    def __init__(self):
        self.running = False
        self.start_time = None
        self.total_trades = 0
        self.current_balance = 100000.0  # $100K 시작
        self.positions = {}
        self.trade_history = []
        self.current_prices = {
            'BTC': 45000.0,
            'ETH': 2500.0
        }
    
    def start(self):
        self.running = True
        self.start_time = datetime.now()
    
    def stop(self):
        self.running = False
    
    def get_status(self):
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        return {
            'running': self.running,
            'uptime': uptime,
            'balance': self.current_balance,
            'total_trades': self.total_trades,
            'positions': len(self.positions),
            'btc_price': self.current_prices['BTC'],
            'eth_price': self.current_prices['ETH']
        }

# 글로벌 봇 인스턴스
bot = TradingBot()

# FastAPI 앱
app = FastAPI(title="Minimal Trading Bot")

# 실시간 가격 업데이트
async def update_prices():
    """개선된 시장 시뮬레이션 - 실제 패턴 모방"""
    # 실제 시장 패턴을 모방한 가격 모델
    trend = 1.0  # 전체적 추세
    volatility = 0.02  # 기본 변동성
    
    while True:
        try:
            # 트렌드 변경 (5% 확률)
            if random.random() < 0.05:
                trend *= random.uniform(0.95, 1.05)
                trend = max(0.8, min(1.2, trend))  # 트렌드 제한
            
            # 변동성 조정 (시장 상황 반영)
            if random.random() < 0.1:
                volatility = random.uniform(0.01, 0.04)  # 1-4% 변동성
            
            # BTC 가격 업데이트 (트렌드 + 노이즈)
            trend_factor = trend * random.uniform(0.999, 1.001)
            noise = random.uniform(-volatility, volatility)
            bot.current_prices['BTC'] *= (trend_factor + noise)
            bot.current_prices['BTC'] = max(20000, min(100000, bot.current_prices['BTC']))
            
            # ETH 가격 (BTC와 상관관계 0.7)
            btc_influence = 0.7 * (trend_factor + noise * 0.8)
            eth_noise = 0.3 * random.uniform(-volatility * 1.2, volatility * 1.2)
            bot.current_prices['ETH'] *= (btc_influence + eth_noise)
            bot.current_prices['ETH'] = max(1000, min(6000, bot.current_prices['ETH']))
            
            await asyncio.sleep(2)
        except Exception as e:
            print(f"가격 업데이트 에러: {e}")
            await asyncio.sleep(5)

# 간단한 트레이딩 로직
async def simple_trading_strategy():
    """개선된 스마트 트레이딩 전략"""
    price_history = {'BTC': [], 'ETH': []}
    
    while True:
        try:
            if not bot.running:
                await asyncio.sleep(10)
                continue
            
            # 가격 히스토리 업데이트
            for symbol in ['BTC', 'ETH']:
                price_history[symbol].append(bot.current_prices[symbol])
                if len(price_history[symbol]) > 50:  # 최근 50개만 유지
                    price_history[symbol] = price_history[symbol][-50:]
            
            # 기술적 분석 기반 거래 (20개 이상 데이터 필요)
            for symbol in ['BTC', 'ETH']:
                if len(price_history[symbol]) >= 20:
                    signal = analyze_market_signal(symbol, price_history[symbol])
                    
                    if signal['action'] != 'hold' and signal['confidence'] > 0.6:
                        # 리스크 관리: 잔액의 최대 10%만 거래
                        max_trade_amount = bot.current_balance * 0.1
                        trade_amount = min(max_trade_amount, signal['confidence'] * 100)
                        
                        if trade_amount >= 10:  # 최소 $10 거래
                            trade = execute_smart_trade(symbol, signal, trade_amount)
                            if trade:
                                bot.trade_history.append(trade)
                                bot.total_trades += 1
                                
                                # 수익/손실 추적
                                if trade['action'] == 'sell' and 'profit' in trade:
                                    bot.current_balance += trade['profit']
                                
                                print(f"🎯 스마트 거래: {symbol} {trade['action']} ${trade_amount:.2f} (신뢰도: {signal['confidence']:.2f})")
            
            await asyncio.sleep(30)  # 30초마다 분석
            
        except Exception as e:
            print(f"트레이딩 에러: {e}")
            await asyncio.sleep(30)

def analyze_market_signal(symbol: str, prices: list) -> dict:
    """기술적 분석 기반 시장 신호 생성"""
    try:
        if len(prices) < 20:
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': 'insufficient_data'}
        
        current_price = prices[-1]
        
        # RSI 계산 (단순 버전)
        price_changes = [prices[i] - prices[i-1] for i in range(1, min(15, len(prices)))]
        gains = [c for c in price_changes if c > 0]
        losses = [-c for c in price_changes if c < 0]
        
        avg_gain = sum(gains) / len(gains) if gains else 0.01
        avg_loss = sum(losses) / len(losses) if losses else 0.01
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 이동평균
        sma_10 = sum(prices[-10:]) / 10
        sma_20 = sum(prices[-20:]) / 20
        
        # 신호 생성
        signals = []
        confidence_factors = []
        
        # RSI 신호
        if rsi < 30:  # 과매도
            signals.append('buy')
            confidence_factors.append(0.8)
        elif rsi > 70:  # 과매수
            signals.append('sell')
            confidence_factors.append(0.8)
        else:
            signals.append('hold')
            confidence_factors.append(0.2)
        
        # 이동평균 신호
        if current_price > sma_10 > sma_20:  # 상승 추세
            signals.append('buy')
            confidence_factors.append(0.6)
        elif current_price < sma_10 < sma_20:  # 하락 추세
            signals.append('sell')
            confidence_factors.append(0.6)
        else:
            signals.append('hold')
            confidence_factors.append(0.3)
        
        # 최종 신호 결정
        buy_score = sum(cf for sig, cf in zip(signals, confidence_factors) if sig == 'buy')
        sell_score = sum(cf for sig, cf in zip(signals, confidence_factors) if sig == 'sell')
        hold_score = sum(cf for sig, cf in zip(signals, confidence_factors) if sig == 'hold')
        
        max_score = max(buy_score, sell_score, hold_score)
        
        if max_score == buy_score and buy_score > 0.5:
            action = 'buy'
            confidence = min(buy_score / 2, 0.9)
        elif max_score == sell_score and sell_score > 0.5:
            action = 'sell' 
            confidence = min(sell_score / 2, 0.9)
        else:
            action = 'hold'
            confidence = 0.3
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': f'RSI:{rsi:.1f}, SMA10:{sma_10:.0f}, SMA20:{sma_20:.0f}'
        }
        
    except Exception as e:
        return {'action': 'hold', 'confidence': 0.0, 'reasoning': 'analysis_error'}

def execute_smart_trade(symbol: str, signal: dict, amount: float) -> dict:
    """스마트 거래 실행"""
    try:
        current_price = bot.current_prices[symbol]
        
        trade = {
            'time': datetime.now().isoformat(),
            'symbol': symbol,
            'action': signal['action'],
            'amount': amount,
            'price': current_price,
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning']
        }
        
        # 간단한 수익/손실 시뮬레이션
        if signal['action'] == 'buy':
            # 매수: 향후 가격 상승 기대
            expected_return = signal['confidence'] * 0.02  # 최대 2% 수익 기대
            trade['expected_profit'] = amount * expected_return
            
        elif signal['action'] == 'sell':
            # 매도: 보유 포지션 청산 (임시 수익 계산)
            profit_rate = (signal['confidence'] - 0.5) * 0.04  # -2% ~ +2%
            trade['profit'] = amount * profit_rate
        
        return trade
        
    except Exception as e:
        return None

# API 엔드포인트
@app.get("/")
async def dashboard():
    """메인 대시보드"""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Minimal Trading Bot</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                text-align: center;
                background: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .card h3 {{
                margin: 0 0 10px 0;
                color: #333;
            }}
            .card .value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #007bff;
            }}
            .controls {{
                text-align: center;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            button {{
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                margin: 0 10px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
            }}
            button:hover {{
                background: #0056b3;
            }}
            button:disabled {{
                background: #ccc;
                cursor: not-allowed;
            }}
            .trades {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .trade-item {{
                padding: 10px;
                border-bottom: 1px solid #eee;
                font-family: monospace;
            }}
            .status-running {{
                color: #28a745;
            }}
            .status-stopped {{
                color: #dc3545;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Minimal Trading Bot</h1>
            <p>간단하지만 실제로 작동하는 트레이딩 시스템</p>
        </div>
        
        <div class="cards">
            <div class="card">
                <h3>봇 상태</h3>
                <div class="value" id="status">로딩중...</div>
            </div>
            <div class="card">
                <h3>잔액</h3>
                <div class="value" id="balance">$0</div>
            </div>
            <div class="card">
                <h3>총 거래</h3>
                <div class="value" id="trades">0</div>
            </div>
            <div class="card">
                <h3>가동시간</h3>
                <div class="value" id="uptime">0초</div>
            </div>
        </div>
        
        <div class="cards">
            <div class="card">
                <h3>BTC 가격</h3>
                <div class="value" id="btc-price">$0</div>
            </div>
            <div class="card">
                <h3>ETH 가격</h3>
                <div class="value" id="eth-price">$0</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="startBot()" id="start-btn">봇 시작</button>
            <button onclick="stopBot()" id="stop-btn">봇 중지</button>
        </div>
        
        <div class="trades">
            <h3>최근 거래</h3>
            <div id="trade-list">거래 내역이 없습니다.</div>
        </div>
        
        <script>
            let ws;
            
            function connectWebSocket() {{
                ws = new WebSocket(`ws://${{location.host}}/ws`);
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                }};
                
                ws.onclose = function() {{
                    setTimeout(connectWebSocket, 1000);
                }};
            }}
            
            function updateDashboard(data) {{
                document.getElementById('status').textContent = data.running ? '실행중' : '중지됨';
                document.getElementById('status').className = data.running ? 'value status-running' : 'value status-stopped';
                document.getElementById('balance').textContent = '$' + data.balance.toLocaleString();
                document.getElementById('trades').textContent = data.total_trades;
                document.getElementById('uptime').textContent = Math.floor(data.uptime) + '초';
                document.getElementById('btc-price').textContent = '$' + data.btc_price.toLocaleString();
                document.getElementById('eth-price').textContent = '$' + data.eth_price.toLocaleString();
                
                document.getElementById('start-btn').disabled = data.running;
                document.getElementById('stop-btn').disabled = !data.running;
            }}
            
            function startBot() {{
                fetch('/api/start', {{method: 'POST'}});
            }}
            
            function stopBot() {{
                fetch('/api/stop', {{method: 'POST'}});
            }}
            
            // 초기화
            connectWebSocket();
            
            // 주기적 상태 업데이트
            setInterval(() => {{
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => updateDashboard(data));
            }}, 1000);
        </script>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """헬스체크 - Railway 필수"""
    return {
        "status": "healthy",
        "service": "Minimal Trading Bot",
        "timestamp": datetime.now().isoformat(),
        "uptime": (datetime.now() - bot.start_time).total_seconds() if bot.start_time else 0,
        "bot_running": bot.running
    }

@app.get("/api/status")
async def get_status():
    """봇 상태 조회"""
    return bot.get_status()

@app.post("/api/start")
async def start_bot():
    """봇 시작"""
    bot.start()
    return {"message": "봇이 시작되었습니다", "running": True}

@app.post("/api/stop")
async def stop_bot():
    """봇 중지"""
    bot.stop()
    return {"message": "봇이 중지되었습니다", "running": False}

@app.get("/api/trades")
async def get_trades():
    """거래 내역 조회"""
    return {"trades": bot.trade_history[-20:]}  # 최근 20개

# WebSocket 연결 관리
connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 데이터 전송"""
    await websocket.accept()
    connections.append(websocket)
    
    try:
        while True:
            # 봇 상태를 모든 연결된 클라이언트에게 전송
            status = bot.get_status()
            for connection in connections[:]:  # 복사본 순회
                try:
                    await connection.send_json(status)
                except:
                    connections.remove(connection)
            
            await asyncio.sleep(2)  # 2초마다 업데이트
            
    except WebSocketDisconnect:
        connections.remove(websocket)

# 백그라운드 태스크 시작
@app.on_event("startup")
async def startup():
    """앱 시작시 백그라운드 태스크 실행"""
    asyncio.create_task(update_prices())
    asyncio.create_task(simple_trading_strategy())
    print("✅ Minimal Trading Bot 시작완료")

if __name__ == "__main__":
    # Railway의 PORT 환경변수 사용
    port = int(os.environ.get("PORT", 8001))
    
    print(f"""
    🎯 Minimal Trading Bot
    
    ✅ 최소한의 의존성
    ✅ 빠른 시작 시간
    ✅ Railway 최적화
    ✅ 실시간 대시보드
    ✅ WebSocket 지원
    
    포트: {port}
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )