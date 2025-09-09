"""
🏆 업계 최고 수준 통합 트레이딩 시스템
Renaissance Technologies + Citadel + Two Sigma의 최고 기술 결합

⚠️ 안전한 Paper Trading 모드로 실행
- 실제 API 사용하되 가상 거래만 실행
- 완벽한 리스크 관리 및 백테스팅
- 실전 배포 전 충분한 검증
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from dotenv import load_dotenv

from professional_risk_manager import ProfessionalRiskManager
from citadel_data_engine import CitadelDataEngine
from two_sigma_backtester import TwoSigmaBacktester
from professional_paper_trader import ProfessionalPaperTrader

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/professional_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalTradingSystem:
    """
    🏆 업계 최고 수준 통합 트레이딩 시스템
    
    핵심 기능:
    1. Renaissance 수준 리스크 관리
    2. Citadel 수준 실시간 데이터 분석
    3. Two Sigma 수준 백테스팅
    4. 안전한 Paper Trading
    5. 실시간 모니터링 및 알림
    """
    
    def __init__(self):
        self.system_status = "initializing"
        self.start_time = datetime.now()
        
        # API 키 설정
        self.api_keys = {
            'binance': {
                'api_key': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET')
            },
            'coinbase': {
                'api_key': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'passphrase': os.getenv('COINBASE_PASSPHRASE')
            }
        }
        
        # 시스템 설정
        self.initial_capital = 100000  # $100,000 가상 자본
        self.paper_trading_mode = True  # 안전 모드
        
        # 핵심 시스템 컴포넌트
        self.risk_manager = None
        self.data_engine = None
        self.backtester = None
        self.paper_trader = None
        
        # 성과 추적
        self.system_metrics = {
            'uptime': 0,
            'total_signals': 0,
            'executed_trades': 0,
            'current_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'system_health': 'good'
        }
        
        # 알림 시스템
        self.alerts = []
        self.max_alerts = 1000

    async def initialize_system(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("🚀 Professional Trading System 초기화 시작...")
            
            # 1. 리스크 관리 시스템 초기화
            self.risk_manager = ProfessionalRiskManager(self.initial_capital)
            logger.info("✅ Renaissance 수준 리스크 관리 시스템 로드")
            
            # 2. 데이터 분석 엔진 초기화
            self.data_engine = CitadelDataEngine()
            logger.info("✅ Citadel 수준 데이터 분석 엔진 로드")
            
            # 3. 백테스팅 시스템 초기화
            self.backtester = TwoSigmaBacktester(self.initial_capital)
            logger.info("✅ Two Sigma 수준 백테스팅 시스템 로드")
            
            # 4. Paper Trading 시스템 초기화
            self.paper_trader = ProfessionalPaperTrader(
                initial_capital=self.initial_capital,
                api_keys=self.api_keys
            )
            
            # Paper Trader 초기화
            paper_init_success = await self.paper_trader.initialize()
            if not paper_init_success:
                raise Exception("Paper Trading 초기화 실패")
            
            logger.info("✅ 전문가급 Paper Trading 시스템 로드")
            
            # 5. 실시간 모니터링 시작
            asyncio.create_task(self._start_system_monitoring())
            asyncio.create_task(self._start_performance_tracking())
            
            self.system_status = "running"
            await self._add_alert("🎯 Professional Trading System 완전 가동", "success")
            
            logger.info("🎉 업계 최고 수준 트레이딩 시스템 가동 완료!")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 에러: {e}")
            self.system_status = "failed"
            await self._add_alert(f"❌ 시스템 초기화 실패: {str(e)}", "error")
            return False

    async def run_comprehensive_backtest(self, days: int = 90) -> Dict:
        """포괄적인 백테스팅 실행"""
        try:
            logger.info(f"📊 {days}일간 포괄적인 백테스팅 시작...")
            
            # 백테스팅 기간 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 샘플 가격 데이터 생성 (실제로는 거래소에서 가져옴)
            price_data = await self._generate_sample_price_data(start_date, end_date)
            
            # 전략 함수 정의
            strategy_func = self._professional_trading_strategy
            
            # 백테스트 실행
            backtest_id = await self.backtester.run_full_backtest(
                strategy_func=strategy_func,
                price_data=price_data,
                start_date=start_date,
                end_date=end_date,
                strategy_params={
                    'risk_level': 'medium',
                    'max_position_size': 0.1,
                    'signal_threshold': 0.7
                }
            )
            
            # 결과 분석
            summary = self.backtester.get_backtest_summary(backtest_id)
            detailed_metrics = self.backtester.get_detailed_metrics(backtest_id)
            
            await self._add_alert(f"📈 백테스팅 완료: {summary['annualized_return']} 연수익률", "info")
            
            return {
                'backtest_id': backtest_id,
                'summary': summary,
                'detailed_metrics': detailed_metrics,
                'recommendation': self._generate_backtest_recommendation(detailed_metrics)
            }
            
        except Exception as e:
            logger.error(f"백테스팅 에러: {e}")
            return {'error': str(e)}

    async def _professional_trading_strategy(self, current_prices: Dict, 
                                          historical_data: Dict, params: Dict) -> List[Dict]:
        """전문가급 트레이딩 전략"""
        signals = []
        
        try:
            for symbol in current_prices.keys():
                # Citadel 수준 시장 분석
                market_signal = await self.data_engine.analyze_symbol(symbol)
                
                if not market_signal or market_signal.confidence < params.get('signal_threshold', 0.7):
                    continue
                
                # Renaissance 수준 포지션 사이징
                position_size = await self.risk_manager.calculate_position_size(
                    symbol, market_signal.signal_type, market_signal.confidence
                )
                
                # 신호 생성
                if market_signal.strength.value > 0:  # Bullish
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': position_size / current_prices[symbol]['close'],
                        'strategy': market_signal.signal_type,
                        'confidence': market_signal.confidence
                    })
                elif market_signal.strength.value < 0:  # Bearish
                    signals.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': position_size / current_prices[symbol]['close'],
                        'strategy': market_signal.signal_type,
                        'confidence': market_signal.confidence
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"전략 실행 에러: {e}")
            return []

    async def _generate_sample_price_data(self, start_date: datetime, 
                                        end_date: datetime) -> Dict:
        """샘플 가격 데이터 생성 (실제로는 거래소 API 사용)"""
        try:
            import pandas as pd
            import numpy as np
            
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            price_data = {}
            
            # 일일 데이터 생성
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for symbol in symbols:
                # 랜덤 워크로 가격 시뮬레이션
                initial_price = 50000 if 'BTC' in symbol else (3000 if 'ETH' in symbol else 300)
                returns = np.random.normal(0.001, 0.02, len(dates))  # 일일 0.1% 평균, 2% 변동성
                
                prices = [initial_price]
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                # OHLCV 데이터 생성
                df_data = []
                for i, (date, price) in enumerate(zip(dates, prices)):
                    high = price * (1 + abs(np.random.normal(0, 0.01)))
                    low = price * (1 - abs(np.random.normal(0, 0.01)))
                    open_price = prices[i-1] if i > 0 else price
                    volume = np.random.uniform(1000, 10000)
                    
                    df_data.append({
                        'timestamp': date,
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': price,
                        'volume': volume
                    })
                
                price_data[symbol] = pd.DataFrame(df_data)
            
            return price_data
            
        except Exception as e:
            logger.error(f"샘플 데이터 생성 에러: {e}")
            return {}

    def _generate_backtest_recommendation(self, metrics: Dict) -> Dict:
        """백테스트 결과 기반 추천"""
        try:
            if not metrics:
                return {'recommendation': 'insufficient_data'}
            
            performance = metrics.get('performance_metrics', {})
            risk = metrics.get('risk_metrics', {})
            trading = metrics.get('trading_metrics', {})
            
            # 성과 평가
            annualized_return = performance.get('annualized_return', 0)
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            max_drawdown = risk.get('max_drawdown', 1)
            win_rate = trading.get('win_rate', 0)
            
            # 점수 계산
            score = 0
            
            # 수익률 점수 (0-30점)
            if annualized_return > 0.20:  # 20% 이상
                score += 30
            elif annualized_return > 0.10:  # 10% 이상
                score += 20
            elif annualized_return > 0.05:  # 5% 이상
                score += 10
            
            # 샤프 비율 점수 (0-25점)
            if sharpe_ratio > 2.0:
                score += 25
            elif sharpe_ratio > 1.5:
                score += 20
            elif sharpe_ratio > 1.0:
                score += 15
            elif sharpe_ratio > 0.5:
                score += 10
            
            # 드로우다운 점수 (0-25점)
            if max_drawdown < 0.05:  # 5% 미만
                score += 25
            elif max_drawdown < 0.10:  # 10% 미만
                score += 20
            elif max_drawdown < 0.15:  # 15% 미만
                score += 15
            elif max_drawdown < 0.20:  # 20% 미만
                score += 10
            
            # 승률 점수 (0-20점)
            if win_rate > 0.60:  # 60% 이상
                score += 20
            elif win_rate > 0.55:  # 55% 이상
                score += 15
            elif win_rate > 0.50:  # 50% 이상
                score += 10
            
            # 추천 등급
            if score >= 80:
                grade = "EXCELLENT"
                recommendation = "실전 배포 강력 추천"
                risk_level = "LOW"
            elif score >= 60:
                grade = "GOOD"
                recommendation = "추가 최적화 후 소액 실전 테스트 권장"
                risk_level = "MEDIUM"
            elif score >= 40:
                grade = "FAIR"
                recommendation = "전략 개선 필요, Paper Trading 지속"
                risk_level = "HIGH"
            else:
                grade = "POOR"
                recommendation = "전략 전면 재검토 필요"
                risk_level = "VERY_HIGH"
            
            return {
                'grade': grade,
                'score': score,
                'recommendation': recommendation,
                'risk_level': risk_level,
                'key_metrics': {
                    'annualized_return': f"{annualized_return:.2%}",
                    'sharpe_ratio': f"{sharpe_ratio:.2f}",
                    'max_drawdown': f"{max_drawdown:.2%}",
                    'win_rate': f"{win_rate:.2%}"
                },
                'next_steps': self._get_next_steps(grade, score)
            }
            
        except Exception as e:
            logger.error(f"추천 생성 에러: {e}")
            return {'recommendation': 'error', 'error': str(e)}

    def _get_next_steps(self, grade: str, score: int) -> List[str]:
        """다음 단계 추천"""
        if grade == "EXCELLENT":
            return [
                "1-2개월 추가 Paper Trading으로 안정성 확인",
                "소액($1,000-$5,000)으로 실전 테스트 시작",
                "정기적인 성과 모니터링 및 리스크 관리",
                "점진적인 자본 증액 고려"
            ]
        elif grade == "GOOD":
            return [
                "전략 파라미터 추가 최적화",
                "다양한 시장 조건에서 백테스팅",
                "리스크 관리 규칙 강화",
                "2-3개월 Paper Trading 연장"
            ]
        elif grade == "FAIR":
            return [
                "신호 생성 로직 개선",
                "포지션 사이징 알고리즘 재검토",
                "손절/익절 규칙 최적화",
                "추가 기술적 지표 도입 검토"
            ]
        else:
            return [
                "전략 아키텍처 전면 재설계",
                "다른 시장/자산으로 테스트",
                "외부 전문가 컨설팅 고려",
                "기본 이론 및 방법론 재학습"
            ]

    async def _start_system_monitoring(self):
        """시스템 실시간 모니터링"""
        while True:
            try:
                # 시스템 상태 체크
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.system_metrics['uptime'] = uptime
                
                # Paper Trader 상태 체크
                if self.paper_trader:
                    portfolio = self.paper_trader.get_portfolio_summary()
                    performance = self.paper_trader.get_performance_metrics()
                    
                    self.system_metrics['current_pnl'] = portfolio.get('total_return', 0)
                    self.system_metrics['executed_trades'] = performance.get('total_trades', 0)
                    self.system_metrics['sharpe_ratio'] = performance.get('sharpe_ratio', 0)
                    self.system_metrics['max_drawdown'] = performance.get('max_drawdown', 0)
                
                # 리스크 관리 상태 체크
                if self.risk_manager:
                    emergency_stop, reason = await self.risk_manager.emergency_shutdown_check()
                    if emergency_stop:
                        await self._add_alert(f"🚨 긴급 중단 신호: {reason}", "critical")
                        await self.emergency_shutdown()
                
                # 시스템 건강도 평가
                await self._evaluate_system_health()
                
                await asyncio.sleep(30)  # 30초마다 체크
                
            except Exception as e:
                logger.error(f"시스템 모니터링 에러: {e}")
                await asyncio.sleep(60)

    async def _start_performance_tracking(self):
        """성과 추적"""
        while True:
            try:
                if self.paper_trader:
                    # 성과 지표 업데이트
                    performance = self.paper_trader.get_performance_metrics()
                    portfolio = self.paper_trader.get_portfolio_summary()
                    
                    # 성과 로그 기록
                    performance_log = {
                        'timestamp': datetime.now().isoformat(),
                        'total_return': portfolio.get('total_return', 0),
                        'total_trades': performance.get('total_trades', 0),
                        'win_rate': performance.get('win_rate', 0),
                        'sharpe_ratio': performance.get('sharpe_ratio', 0),
                        'max_drawdown': performance.get('max_drawdown', 0)
                    }
                    
                    # 성과 알림
                    total_return = portfolio.get('total_return', 0)
                    if total_return > 0.05:  # 5% 이상 수익
                        await self._add_alert(f"🎉 우수한 성과: {total_return:.2%} 수익률", "success")
                    elif total_return < -0.03:  # 3% 이상 손실
                        await self._add_alert(f"⚠️ 손실 발생: {total_return:.2%} 손실률", "warning")
                
                await asyncio.sleep(300)  # 5분마다 추적
                
            except Exception as e:
                logger.error(f"성과 추적 에러: {e}")
                await asyncio.sleep(600)

    async def _evaluate_system_health(self):
        """시스템 건강도 평가"""
        try:
            health_score = 100
            
            # 성과 기반 평가
            current_pnl = self.system_metrics.get('current_pnl', 0)
            max_drawdown = self.system_metrics.get('max_drawdown', 0)
            
            if max_drawdown > 0.10:  # 10% 이상 드로우다운
                health_score -= 30
            elif max_drawdown > 0.05:  # 5% 이상 드로우다운
                health_score -= 15
            
            if current_pnl < -0.05:  # 5% 이상 손실
                health_score -= 25
            
            # 시스템 안정성 평가
            uptime = self.system_metrics.get('uptime', 0)
            if uptime < 3600:  # 1시간 미만
                health_score -= 10
            
            # 건강도 등급
            if health_score >= 90:
                self.system_metrics['system_health'] = 'excellent'
            elif health_score >= 75:
                self.system_metrics['system_health'] = 'good'
            elif health_score >= 60:
                self.system_metrics['system_health'] = 'fair'
            else:
                self.system_metrics['system_health'] = 'poor'
                await self._add_alert(f"⚠️ 시스템 건강도 저하: {health_score}점", "warning")
                
        except Exception as e:
            logger.error(f"시스템 건강도 평가 에러: {e}")

    async def emergency_shutdown(self):
        """긴급 시스템 중단"""
        try:
            logger.warning("🚨 긴급 시스템 중단 시작")
            
            # Paper Trader 중지
            if self.paper_trader:
                await self.paper_trader.stop_trading()
            
            # 모든 포지션 정리 (Paper Trading이므로 안전)
            self.system_status = "emergency_stopped"
            
            await self._add_alert("🛑 긴급 시스템 중단 완료", "critical")
            logger.warning("🛑 긴급 시스템 중단 완료")
            
        except Exception as e:
            logger.error(f"긴급 중단 에러: {e}")

    async def _add_alert(self, message: str, level: str = "info"):
        """알림 추가"""
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'level': level
            }
            
            self.alerts.append(alert)
            
            # 최대 알림 수 제한
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            # 로그 출력
            if level == "critical":
                logger.critical(message)
            elif level == "error":
                logger.error(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "success":
                logger.info(f"✅ {message}")
            else:
                logger.info(message)
                
        except Exception as e:
            logger.error(f"알림 추가 에러: {e}")

    def get_system_status(self) -> Dict:
        """시스템 상태 정보"""
        try:
            portfolio_summary = {}
            performance_metrics = {}
            
            if self.paper_trader:
                portfolio_summary = self.paper_trader.get_portfolio_summary()
                performance_metrics = self.paper_trader.get_performance_metrics()
            
            return {
                'system_status': self.system_status,
                'uptime_hours': self.system_metrics.get('uptime', 0) / 3600,
                'system_health': self.system_metrics.get('system_health', 'unknown'),
                'paper_trading_mode': self.paper_trading_mode,
                'portfolio_summary': portfolio_summary,
                'performance_metrics': performance_metrics,
                'recent_alerts': self.alerts[-10:],  # 최근 10개 알림
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 에러: {e}")
            return {'error': str(e)}

    async def generate_daily_report(self) -> Dict:
        """일일 리포트 생성"""
        try:
            # 백테스팅 실행
            backtest_result = await self.run_comprehensive_backtest(days=30)
            
            # Paper Trading 성과
            portfolio = self.paper_trader.get_portfolio_summary() if self.paper_trader else {}
            performance = self.paper_trader.get_performance_metrics() if self.paper_trader else {}
            
            # 리스크 리포트
            risk_report = await self.paper_trader.get_risk_report() if self.paper_trader else {}
            
            daily_report = {
                'report_date': datetime.now().isoformat(),
                'system_overview': {
                    'status': self.system_status,
                    'uptime_hours': self.system_metrics.get('uptime', 0) / 3600,
                    'health': self.system_metrics.get('system_health', 'unknown')
                },
                'portfolio_status': portfolio,
                'performance_metrics': performance,
                'backtest_results': backtest_result,
                'risk_analysis': risk_report,
                'recent_alerts': self.alerts[-20:],
                'recommendations': backtest_result.get('recommendation', {})
            }
            
            await self._add_alert("📊 일일 리포트 생성 완료", "info")
            return daily_report
            
        except Exception as e:
            logger.error(f"일일 리포트 생성 에러: {e}")
            return {'error': str(e)}


# 메인 실행 함수
async def main():
    """메인 실행 함수"""
    print("""
    🏆 업계 최고 수준 트레이딩 시스템
    
    ⚠️  중요: Paper Trading 모드로 실행
    - 실제 돈이 위험에 노출되지 않습니다
    - 실제 API를 사용하되 가상 거래만 실행
    - 충분한 검증 후 실전 배포 고려
    
    🔧 시스템 구성:
    - Renaissance Technologies 수준 리스크 관리
    - Citadel 수준 실시간 데이터 분석  
    - Two Sigma 수준 백테스팅
    - 전문가급 Paper Trading
    """)
    
    # 시스템 초기화
    trading_system = ProfessionalTradingSystem()
    
    success = await trading_system.initialize_system()
    if not success:
        print("❌ 시스템 초기화 실패")
        return
    
    print("✅ 시스템 초기화 완료! Paper Trading 모드로 실행 중...")
    
    # 백테스팅 실행
    print("\n📊 30일간 백테스팅 실행 중...")
    backtest_result = await trading_system.run_comprehensive_backtest(days=30)
    
    if 'error' not in backtest_result:
        recommendation = backtest_result.get('recommendation', {})
        print(f"\n🎯 백테스팅 결과:")
        print(f"등급: {recommendation.get('grade', 'N/A')}")
        print(f"점수: {recommendation.get('score', 0)}/100")
        print(f"추천사항: {recommendation.get('recommendation', 'N/A')}")
    
    # 시스템 실행 (Paper Trading)
    print("\n🎯 Paper Trading 시작! (Ctrl+C로 중지)")
    
    try:
        # 무한 루프로 시스템 실행
        while True:
            await asyncio.sleep(60)
            
            # 주기적 상태 출력
            status = trading_system.get_system_status()
            portfolio = status.get('portfolio_summary', {})
            
            if portfolio:
                total_return = portfolio.get('total_return', 0)
                print(f"💰 현재 수익률: {total_return:.2%} | 건강도: {status.get('system_health', 'unknown')}")
            
    except KeyboardInterrupt:
        print("\n⏹️  사용자 중지 요청")
        await trading_system.emergency_shutdown()
        print("🛑 시스템 안전 중지 완료")


if __name__ == "__main__":
    asyncio.run(main())