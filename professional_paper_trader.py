"""
🏆 전문가급 Paper Trading 시스템
- 실제 API 사용하되 가상 거래만 실행
- 실전과 동일한 거래 환경 시뮬레이션
- 완벽한 리스크 관리 통합
- 실시간 성과 모니터링
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import ccxt.async_support as ccxt
from dataclasses import dataclass, asdict
from enum import Enum

from professional_risk_manager import ProfessionalRiskManager, Position, RiskMetrics
from citadel_data_engine import CitadelDataEngine, MarketSignal, SignalStrength

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: OrderType
    price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    strategy: str
    signal_strength: float
    filled_price: Optional[float] = None
    filled_quantity: float = 0
    commission: float = 0

@dataclass
class TradingSession:
    session_id: str
    start_time: datetime
    initial_capital: float
    current_capital: float
    total_trades: int
    profitable_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    active: bool

class ProfessionalPaperTrader:
    """
    업계 최고 수준의 Paper Trading 시스템
    - 실제 거래소 API 연결 (가격 정보만 사용)
    - 완벽한 슬리패지/수수료 시뮬레이션
    - 리스크 관리 시스템 통합
    - 실시간 성과 분석
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 api_keys: Dict[str, Dict] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 핵심 시스템 통합
        self.risk_manager = ProfessionalRiskManager(initial_capital)
        self.data_engine = CitadelDataEngine()
        
        # 거래소 연결 (Read-Only)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.api_keys = api_keys or {}
        
        # Paper Trading 상태
        self.paper_positions: Dict[str, Position] = {}
        self.paper_orders: Dict[str, PaperOrder] = {}
        self.order_history: List[PaperOrder] = []
        self.session: Optional[TradingSession] = None
        
        # 거래 파라미터
        self.commission_rates = {
            'binance': 0.001,  # 0.1%
            'coinbase': 0.005  # 0.5%
        }
        self.slippage_rates = {
            'binance': 0.0005,  # 0.05%
            'coinbase': 0.001   # 0.1%
        }
        
        # 안전장치
        self.daily_loss_limit = 0.03  # 3%
        self.position_size_limit = 0.1  # 10%
        self.max_open_orders = 20
        
        # 성과 추적
        self.equity_history: List[float] = [initial_capital]
        self.trade_history: List[Dict] = []
        self.last_equity_update = datetime.now()

    async def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            # 거래소 연결 (읽기 전용)
            await self._initialize_exchanges()
            
            # 데이터 엔진 시작
            await self._start_data_feeds()
            
            # 새로운 트레이딩 세션 시작
            await self._start_new_session()
            
            logger.info("🚀 Professional Paper Trader 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"초기화 에러: {e}")
            return False

    async def _initialize_exchanges(self):
        """거래소 연결 (읽기 전용)"""
        try:
            # 바이낸스 연결
            if 'binance' in self.api_keys:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.api_keys['binance']['api_key'],
                    'secret': self.api_keys['binance']['secret'],
                    'sandbox': True,  # 테스트넷 사용
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                await self.exchanges['binance'].load_markets()
                logger.info("✅ 바이낸스 연결 완료 (읽기 전용)")
            
            # 코인베이스 연결
            if 'coinbase' in self.api_keys:
                self.exchanges['coinbase'] = ccxt.coinbasepro({
                    'apiKey': self.api_keys['coinbase']['api_key'],
                    'secret': self.api_keys['coinbase']['secret'],
                    'passphrase': self.api_keys['coinbase']['passphrase'],
                    'sandbox': True,  # 테스트넷 사용
                    'enableRateLimit': True
                })
                await self.exchanges['coinbase'].load_markets()
                logger.info("✅ 코인베이스 연결 완료 (읽기 전용)")
                
        except Exception as e:
            logger.error(f"거래소 연결 에러: {e}")

    async def _start_data_feeds(self):
        """실시간 데이터 피드 시작"""
        try:
            # 주요 심볼들의 가격 데이터 수집 시작
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            
            for symbol in symbols:
                asyncio.create_task(self._collect_price_data(symbol))
            
            # 실시간 신호 분석 시작
            asyncio.create_task(self._analyze_market_signals())
            
            logger.info("📡 실시간 데이터 피드 시작")
            
        except Exception as e:
            logger.error(f"데이터 피드 시작 에러: {e}")

    async def _collect_price_data(self, symbol: str):
        """실시간 가격 데이터 수집"""
        while True:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    if exchange_name == 'binance':
                        ticker = await exchange.fetch_ticker(symbol)
                        ohlcv = await exchange.fetch_ohlcv(symbol, '1m', limit=100)
                        
                        # 데이터 엔진에 업데이트
                        await self.data_engine.update_price_data(symbol, ohlcv)
                        
                        # 리스크 매니저에 가격 업데이트
                        await self.risk_manager.update_prices({symbol: ticker['last']})
                        
                await asyncio.sleep(30)  # 30초마다 업데이트
                
            except Exception as e:
                logger.error(f"가격 데이터 수집 에러 {symbol}: {e}")
                await asyncio.sleep(60)

    async def _analyze_market_signals(self):
        """실시간 시장 신호 분석"""
        while True:
            try:
                if not self.session or not self.session.active:
                    await asyncio.sleep(60)
                    continue
                
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                
                for symbol in symbols:
                    # Citadel 수준의 신호 분석
                    signal = await self.data_engine.analyze_symbol(symbol)
                    
                    if signal and signal.confidence > 0.7:
                        # 고신뢰도 신호에 대해 거래 검토
                        await self._evaluate_trading_signal(signal)
                
                await asyncio.sleep(60)  # 1분마다 분석
                
            except Exception as e:
                logger.error(f"신호 분석 에러: {e}")
                await asyncio.sleep(120)

    async def _evaluate_trading_signal(self, signal: MarketSignal):
        """거래 신호 평가 및 주문 생성"""
        try:
            # 1. 리스크 관리 체크
            emergency_stop, reason = await self.risk_manager.emergency_shutdown_check()
            if emergency_stop:
                logger.warning(f"⛔ 긴급 중단 신호: {reason}")
                return
            
            # 2. 포지션 사이징
            position_size = await self.risk_manager.calculate_position_size(
                signal.symbol, signal.signal_type, signal.confidence
            )
            
            # 3. 상관관계 리스크 체크
            if not await self.risk_manager.check_correlation_risk(signal.symbol):
                logger.info(f"⚠️ 상관관계 리스크로 인한 신호 무시: {signal.symbol}")
                return
            
            # 4. 주문 생성
            if signal.strength in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH]:
                await self._create_paper_order(
                    symbol=signal.symbol,
                    side='buy',
                    quantity=position_size,
                    order_type=OrderType.MARKET,
                    strategy=signal.signal_type,
                    signal_strength=signal.confidence
                )
            elif signal.strength in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH]:
                # 기존 포지션이 있을 때만 매도
                if signal.symbol in self.paper_positions:
                    current_position = self.paper_positions[signal.symbol].size
                    sell_quantity = min(current_position, position_size)
                    
                    if sell_quantity > 0:
                        await self._create_paper_order(
                            symbol=signal.symbol,
                            side='sell',
                            quantity=sell_quantity,
                            order_type=OrderType.MARKET,
                            strategy=signal.signal_type,
                            signal_strength=signal.confidence
                        )
                        
        except Exception as e:
            logger.error(f"거래 신호 평가 에러: {e}")

    async def _create_paper_order(self, symbol: str, side: str, quantity: float,
                                order_type: OrderType, price: Optional[float] = None,
                                strategy: str = "unknown", signal_strength: float = 0) -> str:
        """Paper Trading 주문 생성"""
        try:
            # 주문 제한 체크
            if len(self.paper_orders) >= self.max_open_orders:
                logger.warning("⚠️ 최대 주문 수 초과")
                return ""
            
            # 주문 ID 생성
            order_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.paper_orders)}"
            
            # Paper Order 생성
            paper_order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                strategy=strategy,
                signal_strength=signal_strength
            )
            
            # 주문 저장
            self.paper_orders[order_id] = paper_order
            
            # 즉시 체결 시뮬레이션 (마켓 주문의 경우)
            if order_type == OrderType.MARKET:
                asyncio.create_task(self._simulate_order_fill(order_id))
            
            logger.info(f"📝 Paper 주문 생성: {symbol} {side} {quantity} ({strategy})")
            return order_id
            
        except Exception as e:
            logger.error(f"Paper 주문 생성 에러: {e}")
            return ""

    async def _simulate_order_fill(self, order_id: str):
        """주문 체결 시뮬레이션"""
        try:
            if order_id not in self.paper_orders:
                return
            
            order = self.paper_orders[order_id]
            
            # 현재 시장 가격 가져오기
            current_price = await self._get_current_market_price(order.symbol)
            if not current_price:
                order.status = OrderStatus.REJECTED
                return
            
            # 슬리페지 적용
            exchange_name = 'binance'  # 기본값
            slippage_rate = self.slippage_rates.get(exchange_name, 0.001)
            commission_rate = self.commission_rates.get(exchange_name, 0.001)
            
            if order.side == 'buy':
                filled_price = current_price * (1 + slippage_rate)
            else:
                filled_price = current_price * (1 - slippage_rate)
            
            # 수수료 계산
            trade_value = order.quantity * filled_price
            commission = trade_value * commission_rate
            
            # 자금 충분성 체크 (매수의 경우)
            if order.side == 'buy':
                required_capital = trade_value + commission
                if required_capital > self.current_capital:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"❌ 자금 부족으로 주문 거부: {order.symbol}")
                    return
                
                self.current_capital -= required_capital
            else:
                # 매도의 경우 보유량 체크
                current_holding = self.paper_positions.get(order.symbol)
                if not current_holding or current_holding.size < order.quantity:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"❌ 보유량 부족으로 주문 거부: {order.symbol}")
                    return
                
                self.current_capital += (trade_value - commission)
            
            # 주문 체결 처리
            order.status = OrderStatus.FILLED
            order.filled_price = filled_price
            order.filled_quantity = order.quantity
            order.commission = commission
            
            # 포지션 업데이트
            await self._update_paper_position(order)
            
            # 리스크 매니저에 포지션 추가
            await self.risk_manager.add_position(
                order.symbol, 
                order.quantity if order.side == 'buy' else -order.quantity,
                filled_price, 
                order.strategy
            )
            
            # 거래 히스토리 기록
            self._record_trade(order)
            
            # 주문을 히스토리로 이동
            self.order_history.append(order)
            del self.paper_orders[order_id]
            
            logger.info(f"✅ 주문 체결: {order.symbol} {order.side} {order.quantity} @ {filled_price:.4f}")
            
        except Exception as e:
            logger.error(f"주문 체결 시뮬레이션 에러: {e}")

    async def _get_current_market_price(self, symbol: str) -> Optional[float]:
        """현재 시장 가격 조회"""
        try:
            for exchange in self.exchanges.values():
                ticker = await exchange.fetch_ticker(symbol)
                return ticker['last']
            return None
        except Exception as e:
            logger.error(f"시장 가격 조회 에러 {symbol}: {e}")
            return None

    async def _update_paper_position(self, order: PaperOrder):
        """Paper Trading 포지션 업데이트"""
        try:
            symbol = order.symbol
            
            if symbol not in self.paper_positions:
                # 새로운 포지션 생성
                if order.side == 'buy':
                    self.paper_positions[symbol] = Position(
                        symbol=symbol,
                        size=order.filled_quantity,
                        entry_price=order.filled_price,
                        current_price=order.filled_price,
                        timestamp=order.timestamp,
                        strategy=order.strategy
                    )
            else:
                # 기존 포지션 수정
                existing_position = self.paper_positions[symbol]
                
                if order.side == 'buy':
                    # 포지션 증가
                    total_value = (existing_position.size * existing_position.entry_price + 
                                 order.filled_quantity * order.filled_price)
                    total_size = existing_position.size + order.filled_quantity
                    new_avg_price = total_value / total_size
                    
                    existing_position.size = total_size
                    existing_position.entry_price = new_avg_price
                else:
                    # 포지션 감소
                    existing_position.size -= order.filled_quantity
                    
                    if existing_position.size <= 0:
                        del self.paper_positions[symbol]
            
        except Exception as e:
            logger.error(f"포지션 업데이트 에러: {e}")

    def _record_trade(self, order: PaperOrder):
        """거래 기록"""
        try:
            trade_record = {
                'timestamp': order.timestamp.isoformat(),
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.filled_quantity,
                'price': order.filled_price,
                'commission': order.commission,
                'strategy': order.strategy,
                'signal_strength': order.signal_strength,
                'trade_value': order.filled_quantity * order.filled_price
            }
            
            self.trade_history.append(trade_record)
            
            # 세션 통계 업데이트
            if self.session:
                self.session.total_trades += 1
                
                # P&L 계산 (간단 버전)
                if order.side == 'sell' and order.symbol in self.paper_positions:
                    # 실제로는 FIFO/LIFO 등 복잡한 계산 필요
                    position = self.paper_positions[order.symbol]
                    trade_pnl = (order.filled_price - position.entry_price) * order.filled_quantity
                    self.session.total_pnl += trade_pnl
                    
                    if trade_pnl > 0:
                        self.session.profitable_trades += 1
            
        except Exception as e:
            logger.error(f"거래 기록 에러: {e}")

    async def _start_new_session(self):
        """새로운 트레이딩 세션 시작"""
        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.session = TradingSession(
                session_id=session_id,
                start_time=datetime.now(),
                initial_capital=self.current_capital,
                current_capital=self.current_capital,
                total_trades=0,
                profitable_trades=0,
                total_pnl=0,
                max_drawdown=0,
                sharpe_ratio=0,
                active=True
            )
            
            logger.info(f"🎯 새로운 트레이딩 세션 시작: {session_id}")
            
        except Exception as e:
            logger.error(f"세션 시작 에러: {e}")

    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        try:
            total_position_value = 0
            position_details = []
            
            for symbol, position in self.paper_positions.items():
                position_value = position.size * position.current_price
                total_position_value += position_value
                
                position_details.append({
                    'symbol': symbol,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'value': position_value,
                    'pnl': position.pnl,
                    'strategy': position.strategy
                })
            
            total_equity = self.current_capital + total_position_value
            total_return = (total_equity - self.initial_capital) / self.initial_capital
            
            return {
                'session_id': self.session.session_id if self.session else None,
                'initial_capital': self.initial_capital,
                'current_cash': self.current_capital,
                'position_value': total_position_value,
                'total_equity': total_equity,
                'total_return': total_return,
                'total_trades': len(self.order_history),
                'active_orders': len(self.paper_orders),
                'positions': position_details,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 요약 에러: {e}")
            return {}

    def get_performance_metrics(self) -> Dict:
        """성과 지표"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_trade_pnl': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'profit_factor': 0
                }
            
            # 기본 통계
            total_trades = len(self.trade_history)
            profitable_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # P&L 통계
            trade_pnls = [trade.get('pnl', 0) for trade in self.trade_history]
            avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0
            
            # 드로우다운
            equity_curve = self.equity_history
            if len(equity_curve) > 1:
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (np.array(equity_curve) - running_max) / running_max
                max_drawdown = abs(np.min(drawdown))
            else:
                max_drawdown = 0
            
            # 샤프 비율 (간단 계산)
            if len(equity_curve) > 10:
                returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
                excess_return = np.mean(returns) - 0.02/252  # 일일 무위험 수익률
                sharpe_ratio = excess_return / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # 수익 팩터
            gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_trade_pnl': avg_trade_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }
            
        except Exception as e:
            logger.error(f"성과 지표 계산 에러: {e}")
            return {}

    async def stop_trading(self):
        """트레이딩 중지"""
        try:
            if self.session:
                self.session.active = False
                self.session.current_capital = self.current_capital
                
            # 모든 pending 주문 취소
            for order_id in list(self.paper_orders.keys()):
                self.paper_orders[order_id].status = OrderStatus.CANCELLED
                self.order_history.append(self.paper_orders[order_id])
                del self.paper_orders[order_id]
            
            logger.info("🛑 Paper Trading 중지")
            
        except Exception as e:
            logger.error(f"트레이딩 중지 에러: {e}")

    async def get_risk_report(self) -> Dict:
        """리스크 리포트"""
        try:
            risk_metrics = await self.risk_manager.update_portfolio_metrics()
            risk_summary = self.risk_manager.get_risk_summary()
            
            return {
                'risk_metrics': asdict(risk_metrics),
                'risk_summary': risk_summary,
                'emergency_status': await self.risk_manager.emergency_shutdown_check()
            }
            
        except Exception as e:
            logger.error(f"리스크 리포트 에러: {e}")
            return {}

    def export_trading_log(self) -> Dict:
        """거래 로그 내보내기"""
        try:
            return {
                'session': asdict(self.session) if self.session else None,
                'portfolio_summary': self.get_portfolio_summary(),
                'performance_metrics': self.get_performance_metrics(),
                'trade_history': self.trade_history,
                'order_history': [asdict(order) for order in self.order_history],
                'equity_curve': self.equity_history,
                'export_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"거래 로그 내보내기 에러: {e}")
            return {}