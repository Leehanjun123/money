"""
🏆 업계 최고 수준 리스크 관리 시스템
Renaissance Technologies, Citadel, Two Sigma의 노하우 통합
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    timestamp: datetime
    strategy: str
    
    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    @property
    def value(self) -> float:
        return self.size * self.current_price

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk 95%
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    total_exposure: float
    leverage: float

class ProfessionalRiskManager:
    """
    Renaissance Technologies 수준의 리스크 관리
    - 실시간 VaR 계산
    - 포트폴리오 최적화
    - 동적 포지션 사이징
    - 상관관계 기반 리스크 제어
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.returns_history: List[float] = []
        self.equity_curve: List[float] = [initial_capital]
        
        # Renaissance-style 리스크 파라미터
        self.max_portfolio_var = 0.02  # 2% VaR 한도
        self.max_individual_weight = 0.05  # 개별 종목 5% 한도
        self.max_sector_weight = 0.20  # 섹터별 20% 한도
        self.max_leverage = 2.0  # 최대 레버리지 2배
        self.correlation_threshold = 0.7  # 상관관계 임계값
        
        # 동적 스톱 로스 (Citadel 방식)
        self.daily_loss_limit = 0.03  # 일일 손실 3% 한도
        self.weekly_loss_limit = 0.08  # 주간 손실 8% 한도
        self.monthly_loss_limit = 0.15  # 월간 손실 15% 한도
        
        # 포지션 사이징 (Kelly Criterion 기반)
        self.win_rate = 0.55  # 초기 승률 55%
        self.avg_win = 0.025  # 평균 수익 2.5%
        self.avg_loss = 0.015  # 평균 손실 1.5%
        
    async def calculate_position_size(self, symbol: str, strategy: str, 
                                    confidence: float) -> float:
        """
        Kelly Criterion + Renaissance 방식의 포지션 사이징
        """
        try:
            # Kelly Criterion
            kelly_fraction = ((self.win_rate * self.avg_win) - 
                            ((1 - self.win_rate) * self.avg_loss)) / self.avg_win
            
            # 안전 계수 적용 (Kelly의 25%만 사용)
            safe_kelly = kelly_fraction * 0.25
            
            # 변동성 조정
            volatility = await self._calculate_volatility(symbol)
            vol_adjusted_size = safe_kelly / max(volatility, 0.01)
            
            # 신뢰도 조정
            confidence_adjusted = vol_adjusted_size * confidence
            
            # 최대 한도 적용
            max_position_value = self.current_capital * self.max_individual_weight
            position_size = min(confidence_adjusted * self.current_capital, max_position_value)
            
            return max(position_size, 100)  # 최소 $100
            
        except Exception as e:
            logger.error(f"포지션 사이징 에러: {e}")
            return self.current_capital * 0.01  # 1% 기본값

    async def _calculate_volatility(self, symbol: str) -> float:
        """변동성 계산"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return 0.02  # 기본 변동성 2%
        
        prices = np.array(self.price_history[symbol][-20:])
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252)  # 연율화

    async def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Value at Risk 계산 (Renaissance 방식)
        몬테카르로 시뮬레이션 + 히스토리컬 방법 조합
        """
        try:
            if len(self.returns_history) < 30:
                return 0.0
            
            returns = np.array(self.returns_history[-252:])  # 1년 데이터
            
            # 히스토리컬 VaR
            historical_var = np.percentile(returns, (1 - confidence) * 100)
            
            # 몬테카르로 VaR
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # 10,000번 시뮬레이션
            simulated_returns = np.random.normal(mean_return, std_return, 10000)
            monte_carlo_var = np.percentile(simulated_returns, (1 - confidence) * 100)
            
            # 두 방법의 가중평균 (히스토리컬 70%, 몬테카르로 30%)
            final_var = 0.7 * historical_var + 0.3 * monte_carlo_var
            
            return abs(final_var) * self.current_capital
            
        except Exception as e:
            logger.error(f"VaR 계산 에러: {e}")
            return self.current_capital * 0.05

    async def check_correlation_risk(self, new_symbol: str) -> bool:
        """
        상관관계 리스크 체크 (Two Sigma 방식)
        """
        try:
            if not self.positions or new_symbol not in self.price_history:
                return True
            
            new_prices = np.array(self.price_history[new_symbol][-30:])
            
            for existing_symbol in self.positions.keys():
                if existing_symbol in self.price_history:
                    existing_prices = np.array(self.price_history[existing_symbol][-30:])
                    
                    if len(new_prices) == len(existing_prices):
                        correlation = np.corrcoef(new_prices, existing_prices)[0, 1]
                        
                        if abs(correlation) > self.correlation_threshold:
                            logger.warning(f"높은 상관관계 감지: {new_symbol} vs {existing_symbol} = {correlation:.3f}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"상관관계 체크 에러: {e}")
            return True

    async def dynamic_stop_loss(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Citadel 스타일의 동적 스톱로스
        변동성과 시장 조건에 따라 동적 조정
        """
        try:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            volatility = await self._calculate_volatility(symbol)
            
            # 기본 스톱로스: 2 * 변동성
            base_stop = 2 * volatility
            
            # 시장 상황에 따른 조정
            market_stress = await self._calculate_market_stress()
            adjusted_stop = base_stop * (1 + market_stress)
            
            # 최대 스톱로스 5%로 제한
            final_stop = min(adjusted_stop, 0.05)
            
            if position.size > 0:  # 롱 포지션
                stop_price = position.entry_price * (1 - final_stop)
            else:  # 숏 포지션
                stop_price = position.entry_price * (1 + final_stop)
            
            return stop_price
            
        except Exception as e:
            logger.error(f"동적 스톱로스 계산 에러: {e}")
            return None

    async def _calculate_market_stress(self) -> float:
        """시장 스트레스 지수 계산"""
        try:
            if len(self.returns_history) < 20:
                return 0.0
            
            recent_returns = np.array(self.returns_history[-20:])
            recent_vol = np.std(recent_returns)
            long_term_vol = np.std(self.returns_history[-100:]) if len(self.returns_history) >= 100 else recent_vol
            
            stress_ratio = recent_vol / max(long_term_vol, 0.001)
            return min(max(stress_ratio - 1, 0), 1.0)  # 0-1 범위로 제한
            
        except:
            return 0.0

    async def update_portfolio_metrics(self) -> RiskMetrics:
        """포트폴리오 리스크 메트릭스 업데이트"""
        try:
            var_95 = await self.calculate_var(0.95)
            max_dd = await self._calculate_max_drawdown()
            sharpe = await self._calculate_sharpe_ratio()
            sortino = await self._calculate_sortino_ratio()
            beta = await self._calculate_beta()
            alpha = await self._calculate_alpha()
            
            total_exposure = sum(abs(pos.value) for pos in self.positions.values())
            leverage = total_exposure / self.current_capital if self.current_capital > 0 else 0
            
            return RiskMetrics(
                var_95=var_95,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                beta=beta,
                alpha=alpha,
                total_exposure=total_exposure,
                leverage=leverage
            )
            
        except Exception as e:
            logger.error(f"메트릭스 업데이트 에러: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    async def _calculate_max_drawdown(self) -> float:
        """최대 드로우다운 계산"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return abs(np.min(drawdown))

    async def _calculate_sharpe_ratio(self) -> float:
        """샤프 비율 계산"""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_return = np.mean(returns) - 0.02/252  # 2% 무위험 수익률
        return excess_return / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    async def _calculate_sortino_ratio(self) -> float:
        """소르티노 비율 계산"""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_return = np.mean(returns) - 0.02/252
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 0.001
        return excess_return / downside_dev * np.sqrt(252)

    async def _calculate_beta(self) -> float:
        """베타 계산 (시장 대비)"""
        # 간단한 베타 추정치 (실제로는 벤치마크 필요)
        if len(self.returns_history) < 60:
            return 1.0
        
        returns = np.array(self.returns_history[-60:])
        market_returns = np.random.normal(0.0003, 0.012, 60)  # 가상 시장 수익률
        
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance > 0 else 1.0

    async def _calculate_alpha(self) -> float:
        """알파 계산"""
        sharpe = await self._calculate_sharpe_ratio()
        beta = await self._calculate_beta()
        market_return = 0.08  # 8% 연간 시장 수익률 가정
        risk_free_rate = 0.02  # 2% 무위험 수익률
        
        portfolio_return = np.mean(self.returns_history) * 252 if self.returns_history else 0
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        
        return portfolio_return - expected_return

    async def emergency_shutdown_check(self) -> Tuple[bool, str]:
        """
        긴급 중단 체크 (Circuit Breaker)
        Renaissance Technologies의 리스크 컨트롤 방식
        """
        try:
            reasons = []
            
            # 1. 일일 손실 한도 체크
            if len(self.equity_curve) >= 2:
                daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
                if daily_return < -self.daily_loss_limit:
                    reasons.append(f"일일 손실 한도 초과: {daily_return:.2%}")
            
            # 2. VaR 한도 체크
            current_var = await self.calculate_var()
            if current_var > self.max_portfolio_var * self.current_capital:
                reasons.append(f"VaR 한도 초과: {current_var:.0f}")
            
            # 3. 레버리지 체크
            total_exposure = sum(abs(pos.value) for pos in self.positions.values())
            leverage = total_exposure / self.current_capital if self.current_capital > 0 else 0
            if leverage > self.max_leverage:
                reasons.append(f"레버리지 한도 초과: {leverage:.2f}x")
            
            # 4. 시장 스트레스 체크
            market_stress = await self._calculate_market_stress()
            if market_stress > 0.8:  # 스트레스 지수 80% 초과
                reasons.append(f"극심한 시장 스트레스: {market_stress:.2%}")
            
            should_shutdown = len(reasons) > 0
            reason_text = "; ".join(reasons) if reasons else ""
            
            return should_shutdown, reason_text
            
        except Exception as e:
            logger.error(f"긴급 중단 체크 에러: {e}")
            return True, f"리스크 관리 시스템 오류: {str(e)}"

    async def add_position(self, symbol: str, size: float, price: float, strategy: str):
        """포지션 추가"""
        self.positions[symbol] = Position(
            symbol=symbol,
            size=size,
            entry_price=price,
            current_price=price,
            timestamp=datetime.now(),
            strategy=strategy
        )
        logger.info(f"포지션 추가: {symbol} {size} @ {price}")

    async def update_prices(self, prices: Dict[str, float]):
        """가격 업데이트"""
        for symbol, price in prices.items():
            # 가격 히스토리 업데이트
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            
            # 최근 100개 데이터만 유지
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # 포지션 현재가 업데이트
            if symbol in self.positions:
                self.positions[symbol].current_price = price
        
        # 포트폴리오 밸류 업데이트
        total_value = self.current_capital
        for pos in self.positions.values():
            total_value += pos.pnl / 100 * pos.size
        
        if len(self.equity_curve) > 0:
            daily_return = (total_value - self.equity_curve[-1]) / self.equity_curve[-1]
            self.returns_history.append(daily_return)
            
            # 최근 1000개 리턴만 유지
            if len(self.returns_history) > 1000:
                self.returns_history = self.returns_history[-1000:]
        
        self.equity_curve.append(total_value)
        self.current_capital = total_value

    def get_risk_summary(self) -> Dict:
        """리스크 요약 정보"""
        metrics = asyncio.create_task(self.update_portfolio_metrics())
        
        return {
            'current_capital': self.current_capital,
            'total_positions': len(self.positions),
            'daily_pnl': self.returns_history[-1] if self.returns_history else 0,
            'win_rate': self.win_rate,
            'risk_level': self._get_risk_level(),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'size': pos.size,
                    'pnl': pos.pnl,
                    'strategy': pos.strategy
                }
                for pos in self.positions.values()
            ]
        }
    
    def _get_risk_level(self) -> str:
        """현재 리스크 수준 평가"""
        try:
            if not self.positions:
                return RiskLevel.LOW.value
            
            total_exposure = sum(abs(pos.value) for pos in self.positions.values())
            leverage = total_exposure / self.current_capital if self.current_capital > 0 else 0
            
            if leverage > 1.5:
                return RiskLevel.CRITICAL.value
            elif leverage > 1.0:
                return RiskLevel.HIGH.value
            elif leverage > 0.5:
                return RiskLevel.MEDIUM.value
            else:
                return RiskLevel.LOW.value
                
        except:
            return RiskLevel.MEDIUM.value