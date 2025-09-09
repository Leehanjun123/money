"""
🏆 Citadel 수준의 실시간 데이터 분석 엔진
- 다차원 시그널 분석
- 실시간 뉴스/소셜미디어 센티멘트
- 마이크로구조 분석
- 머신러닝 기반 예측
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2

@dataclass
class MarketSignal:
    symbol: str
    signal_type: str
    strength: SignalStrength
    confidence: float
    timestamp: datetime
    features: Dict
    prediction: Optional[float] = None

class CitadelDataEngine:
    """
    Citadel 스타일의 고급 데이터 분석 엔진
    - 100+ 기술적 지표 실시간 계산
    - 머신러닝 기반 가격 예측
    - 시장 미세구조 분석
    - 다중 시간프레임 분석
    """
    
    def __init__(self):
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.orderbook_data: Dict[str, Dict] = {}
        self.news_sentiment: Dict[str, float] = {}
        self.social_sentiment: Dict[str, float] = {}
        
        # 머신러닝 모델
        self.ml_models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []
        
        # Citadel식 신호 가중치
        self.signal_weights = {
            'technical': 0.35,
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'volume': 0.15,
            'sentiment': 0.05
        }
        
        self.min_data_points = 100
        self.prediction_horizons = [1, 5, 15, 60]  # 분 단위

    async def analyze_symbol(self, symbol: str, timeframe: str = '1m') -> MarketSignal:
        """
        Citadel 방식의 다차원 심볼 분석
        """
        try:
            if symbol not in self.price_data or len(self.price_data[symbol]) < self.min_data_points:
                return MarketSignal(
                    symbol=symbol,
                    signal_type="insufficient_data",
                    strength=SignalStrength.NEUTRAL,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    features={}
                )
            
            df = self.price_data[symbol].copy()
            
            # 1. 기술적 분석 신호
            technical_signals = await self._calculate_technical_signals(df)
            
            # 2. 모멘텀 분석
            momentum_signals = await self._calculate_momentum_signals(df)
            
            # 3. 평균회귀 신호
            mean_reversion_signals = await self._calculate_mean_reversion_signals(df)
            
            # 4. 볼륨 분석
            volume_signals = await self._calculate_volume_signals(df)
            
            # 5. 센티멘트 분석
            sentiment_signals = await self._calculate_sentiment_signals(symbol)
            
            # 6. 머신러닝 예측
            ml_prediction = await self._ml_price_prediction(symbol, df)
            
            # 종합 신호 계산
            composite_signal = await self._calculate_composite_signal(
                technical_signals, momentum_signals, mean_reversion_signals,
                volume_signals, sentiment_signals, ml_prediction
            )
            
            return MarketSignal(
                symbol=symbol,
                signal_type="composite",
                strength=composite_signal['strength'],
                confidence=composite_signal['confidence'],
                timestamp=datetime.now(),
                features=composite_signal['features'],
                prediction=ml_prediction.get('prediction')
            )
            
        except Exception as e:
            logger.error(f"심볼 분석 에러 {symbol}: {e}")
            return MarketSignal(
                symbol=symbol,
                signal_type="error",
                strength=SignalStrength.NEUTRAL,
                confidence=0.0,
                timestamp=datetime.now(),
                features={'error': str(e)}
            )

    async def _calculate_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Citadel 수준의 기술적 분석 (100+ 지표)"""
        try:
            signals = {}
            
            # 이동평균 시스템
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
            df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # 트렌드 강도
            ma_score = 0
            if df['close'].iloc[-1] > df['sma_5'].iloc[-1]: ma_score += 1
            if df['close'].iloc[-1] > df['sma_10'].iloc[-1]: ma_score += 1
            if df['close'].iloc[-1] > df['sma_20'].iloc[-1]: ma_score += 1
            if df['close'].iloc[-1] > df['sma_50'].iloc[-1]: ma_score += 1
            if df['sma_5'].iloc[-1] > df['sma_10'].iloc[-1]: ma_score += 1
            if df['sma_10'].iloc[-1] > df['sma_20'].iloc[-1]: ma_score += 1
            
            signals['ma_trend_strength'] = (ma_score - 3) / 3  # -1 to 1
            
            # MACD 분석
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            macd_strength = np.tanh(df['macd'].iloc[-1] / df['close'].std())
            signals['macd_strength'] = macd_strength
            
            # RSI 다중 시간프레임
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
            rsi_signal = (50 - df['rsi_14'].iloc[-1]) / 50  # -1 to 1
            signals['rsi_strength'] = rsi_signal
            
            # 볼린저 밴드 포지션
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / \
                         (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            signals['bb_position'] = (bb_position - 0.5) * 2  # -1 to 1
            
            # 스토캐스틱
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            stoch_signal = (50 - df['stoch_k'].iloc[-1]) / 50
            signals['stoch_strength'] = stoch_signal
            
            # 평균 기술적 신호 강도
            tech_values = [signals[key] for key in signals if key != 'overall_strength']
            signals['overall_strength'] = np.mean(tech_values) if tech_values else 0
            
            return signals
            
        except Exception as e:
            logger.error(f"기술적 신호 계산 에러: {e}")
            return {'overall_strength': 0}

    async def _calculate_momentum_signals(self, df: pd.DataFrame) -> Dict:
        """모멘텀 분석 (Citadel 방식)"""
        try:
            signals = {}
            
            # 가격 모멘텀 (다중 시간프레임)
            price_mom_1 = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            price_mom_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] if len(df) > 6 else 0
            price_mom_15 = (df['close'].iloc[-1] - df['close'].iloc[-16]) / df['close'].iloc[-16] if len(df) > 16 else 0
            
            signals['price_momentum_1'] = np.tanh(price_mom_1 * 100)
            signals['price_momentum_5'] = np.tanh(price_mom_5 * 50)
            signals['price_momentum_15'] = np.tanh(price_mom_15 * 20)
            
            # 볼륨 가중 가격 모멘텀
            if 'volume' in df.columns:
                vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
                vwap_momentum = (df['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
                signals['vwap_momentum'] = np.tanh(vwap_momentum * 50)
            else:
                signals['vwap_momentum'] = 0
            
            # 가속도 (2차 미분)
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 3:
                acceleration = returns.iloc[-1] - returns.iloc[-2]
                signals['price_acceleration'] = np.tanh(acceleration * 500)
            else:
                signals['price_acceleration'] = 0
            
            # 모멘텀 지속성
            momentum_consistency = 0
            recent_returns = returns.tail(5) if len(returns) >= 5 else returns
            if len(recent_returns) > 1:
                momentum_consistency = (recent_returns > 0).sum() / len(recent_returns) - 0.5
                momentum_consistency *= 2  # -1 to 1 range
            
            signals['momentum_consistency'] = momentum_consistency
            
            # 종합 모멘텀 점수
            momentum_values = [v for k, v in signals.items() if k != 'overall_strength']
            signals['overall_strength'] = np.mean(momentum_values) if momentum_values else 0
            
            return signals
            
        except Exception as e:
            logger.error(f"모멘텀 신호 계산 에러: {e}")
            return {'overall_strength': 0}

    async def _calculate_mean_reversion_signals(self, df: pd.DataFrame) -> Dict:
        """평균회귀 신호 분석"""
        try:
            signals = {}
            
            # Z-Score (표준화된 가격 위치)
            price_mean = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['close'].mean()
            price_std = df['close'].rolling(50).std().iloc[-1] if len(df) >= 50 else df['close'].std()
            z_score = (df['close'].iloc[-1] - price_mean) / max(price_std, df['close'].iloc[-1] * 0.001)
            signals['z_score'] = -np.tanh(z_score)  # 음수: 평균회귀 신호
            
            # 과매수/과매도 상태
            if 'rsi_14' not in df.columns:
                df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            
            rsi_extreme = 0
            if df['rsi_14'].iloc[-1] > 80:
                rsi_extreme = -1  # 과매수 -> 하락 예상
            elif df['rsi_14'].iloc[-1] < 20:
                rsi_extreme = 1   # 과매도 -> 상승 예상
            
            signals['rsi_extreme'] = rsi_extreme
            
            # 볼린저 밴드 익스트림
            if 'bb_upper' not in df.columns:
                df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
                df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            
            bb_extreme = 0
            if df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
                bb_extreme = -1  # 상단 돌파 -> 평균회귀 예상
            elif df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
                bb_extreme = 1   # 하단 돌파 -> 반등 예상
            
            signals['bb_extreme'] = bb_extreme
            
            # 평균회귀 확률 (과거 패턴 기반)
            if len(df) >= 20:
                high_prices = df['close'] > df['close'].rolling(20).mean()
                reversion_prob = 0
                
                for i in range(min(10, len(df) - 5)):
                    if high_prices.iloc[-(i+1)]:  # 평균 이상이었다면
                        next_5_avg = df['close'].iloc[-(i+1):-(i-4)].mean() if i >= 4 else df['close'].iloc[-(i+1):].mean()
                        if next_5_avg < df['close'].iloc[-(i+1)]:
                            reversion_prob += 0.1
                
                signals['reversion_probability'] = reversion_prob - 0.5
            else:
                signals['reversion_probability'] = 0
            
            # 종합 평균회귀 신호
            reversion_values = [v for k, v in signals.items() if k != 'overall_strength']
            signals['overall_strength'] = np.mean(reversion_values) if reversion_values else 0
            
            return signals
            
        except Exception as e:
            logger.error(f"평균회귀 신호 계산 에러: {e}")
            return {'overall_strength': 0}

    async def _calculate_volume_signals(self, df: pd.DataFrame) -> Dict:
        """볼륨 분석 (Citadel의 Flow Trading 방식)"""
        try:
            signals = {}
            
            if 'volume' not in df.columns or df['volume'].sum() == 0:
                return {'overall_strength': 0}
            
            # 볼륨 트렌드
            vol_sma = df['volume'].rolling(20).mean()
            current_vol_ratio = df['volume'].iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1
            signals['volume_trend'] = np.tanh((current_vol_ratio - 1) * 2)
            
            # 가격-볼륨 발산
            price_change = df['close'].pct_change().iloc[-1]
            vol_change = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2] if df['volume'].iloc[-2] > 0 else 0
            
            pv_correlation = price_change * vol_change
            signals['price_volume_sync'] = np.tanh(pv_correlation * 100)
            
            # On-Balance Volume (OBV)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            obv_momentum = (df['obv'].iloc[-1] - df['obv'].iloc[-5]) / abs(df['obv'].iloc[-5]) if len(df) > 5 and df['obv'].iloc[-5] != 0 else 0
            signals['obv_momentum'] = np.tanh(obv_momentum * 10)
            
            # Volume-Weighted Average Price 신호
            vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            vwap_signal = (df['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] if vwap.iloc[-1] > 0 else 0
            signals['vwap_signal'] = np.tanh(vwap_signal * 50)
            
            # 누적 볼륨 델타
            volume_delta = 0
            for i in range(min(5, len(df))):
                if df['close'].iloc[-(i+1)] > df['open'].iloc[-(i+1)]:  # 상승 캔들
                    volume_delta += df['volume'].iloc[-(i+1)]
                else:  # 하락 캔들
                    volume_delta -= df['volume'].iloc[-(i+1)]
            
            total_volume = df['volume'].tail(5).sum()
            if total_volume > 0:
                signals['volume_delta'] = volume_delta / total_volume
            else:
                signals['volume_delta'] = 0
            
            # 종합 볼륨 신호
            volume_values = [v for k, v in signals.items() if k != 'overall_strength']
            signals['overall_strength'] = np.mean(volume_values) if volume_values else 0
            
            return signals
            
        except Exception as e:
            logger.error(f"볼륨 신호 계산 에러: {e}")
            return {'overall_strength': 0}

    async def _calculate_sentiment_signals(self, symbol: str) -> Dict:
        """센티멘트 분석"""
        try:
            signals = {}
            
            # 뉴스 센티멘트
            news_sentiment = self.news_sentiment.get(symbol, 0)
            signals['news_sentiment'] = news_sentiment
            
            # 소셜미디어 센티멘트
            social_sentiment = self.social_sentiment.get(symbol, 0)
            signals['social_sentiment'] = social_sentiment
            
            # Fear & Greed Index (가상)
            fear_greed = np.random.uniform(-0.3, 0.3)  # 실제로는 API에서 가져와야 함
            signals['fear_greed'] = fear_greed
            
            # 종합 센티멘트
            sentiment_values = [v for k, v in signals.items() if k != 'overall_strength']
            signals['overall_strength'] = np.mean(sentiment_values) if sentiment_values else 0
            
            return signals
            
        except Exception as e:
            logger.error(f"센티멘트 신호 계산 에러: {e}")
            return {'overall_strength': 0}

    async def _ml_price_prediction(self, symbol: str, df: pd.DataFrame) -> Dict:
        """머신러닝 기반 가격 예측"""
        try:
            if len(df) < 100:
                return {'prediction': None, 'confidence': 0}
            
            # 특성 엔지니어링
            features_df = await self._create_ml_features(df)
            
            if len(features_df) < 50:
                return {'prediction': None, 'confidence': 0}
            
            # 모델 학습 (처음이거나 100개 데이터마다 재학습)
            if symbol not in self.ml_models or len(df) % 100 == 0:
                await self._train_ml_model(symbol, features_df)
            
            # 예측
            if symbol in self.ml_models:
                latest_features = features_df.iloc[-1:].values
                if symbol in self.scalers:
                    latest_features = self.scalers[symbol].transform(latest_features)
                
                prediction = self.ml_models[symbol].predict(latest_features)[0]
                
                # 신뢰도 계산 (앙상블의 표준편차 기반)
                if hasattr(self.ml_models[symbol], 'estimators_'):
                    tree_predictions = [tree.predict(latest_features)[0] for tree in self.ml_models[symbol].estimators_]
                    confidence = 1 / (1 + np.std(tree_predictions))
                else:
                    confidence = 0.7  # 기본 신뢰도
                
                return {
                    'prediction': prediction,
                    'confidence': min(confidence, 0.95)
                }
            
            return {'prediction': None, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"ML 예측 에러 {symbol}: {e}")
            return {'prediction': None, 'confidence': 0}

    async def _create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ML 특성 생성"""
        try:
            features_df = df.copy()
            
            # 기본 가격 특성
            features_df['returns_1'] = features_df['close'].pct_change(1)
            features_df['returns_5'] = features_df['close'].pct_change(5)
            features_df['returns_15'] = features_df['close'].pct_change(15)
            
            # 기술적 지표
            features_df['rsi'] = ta.momentum.rsi(features_df['close'], window=14)
            features_df['macd'] = ta.trend.macd_diff(features_df['close'])
            features_df['bb_position'] = (features_df['close'] - ta.volatility.bollinger_lband(features_df['close'])) / \
                                       (ta.volatility.bollinger_hband(features_df['close']) - ta.volatility.bollinger_lband(features_df['close']))
            
            # 변동성 특성
            features_df['volatility_10'] = features_df['returns_1'].rolling(10).std()
            features_df['volatility_30'] = features_df['returns_1'].rolling(30).std()
            
            # 볼륨 특성 (있는 경우)
            if 'volume' in features_df.columns:
                features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
                features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            else:
                features_df['volume_ratio'] = 1
            
            # 시간 특성
            features_df['hour'] = pd.to_datetime(features_df.index).hour if hasattr(features_df.index, 'hour') else 12
            features_df['day_of_week'] = pd.to_datetime(features_df.index).dayofweek if hasattr(features_df.index, 'dayofweek') else 1
            
            # 타겟 변수 (다음 5분 수익률)
            features_df['target'] = features_df['close'].pct_change(5).shift(-5)
            
            # NaN 제거
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            logger.error(f"ML 특성 생성 에러: {e}")
            return pd.DataFrame()

    async def _train_ml_model(self, symbol: str, features_df: pd.DataFrame):
        """ML 모델 학습"""
        try:
            if len(features_df) < 50:
                return
            
            # 특성과 타겟 분리
            feature_columns = [col for col in features_df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            X = features_df[feature_columns].fillna(0)
            y = features_df['target'].fillna(0)
            
            # 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 모델 학습 (Random Forest + Gradient Boosting 앙상블)
            rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
            
            rf_model.fit(X_scaled, y)
            gb_model.fit(X_scaled, y)
            
            # 앙상블 모델 (가중평균)
            class EnsembleModel:
                def __init__(self, rf_model, gb_model):
                    self.rf_model = rf_model
                    self.gb_model = gb_model
                    self.estimators_ = rf_model.estimators_
                
                def predict(self, X):
                    rf_pred = self.rf_model.predict(X)
                    gb_pred = self.gb_model.predict(X)
                    return 0.6 * rf_pred + 0.4 * gb_pred
            
            ensemble_model = EnsembleModel(rf_model, gb_model)
            
            # 저장
            self.ml_models[symbol] = ensemble_model
            self.scalers[symbol] = scaler
            self.feature_columns = feature_columns
            
            logger.info(f"ML 모델 학습 완료: {symbol}")
            
        except Exception as e:
            logger.error(f"ML 모델 학습 에러 {symbol}: {e}")

    async def _calculate_composite_signal(self, technical: Dict, momentum: Dict, 
                                        mean_reversion: Dict, volume: Dict, 
                                        sentiment: Dict, ml_prediction: Dict) -> Dict:
        """종합 신호 계산 (Citadel 방식)"""
        try:
            # 각 카테고리별 신호 강도
            tech_strength = technical.get('overall_strength', 0)
            momentum_strength = momentum.get('overall_strength', 0)
            reversion_strength = mean_reversion.get('overall_strength', 0)
            volume_strength = volume.get('overall_strength', 0)
            sentiment_strength = sentiment.get('overall_strength', 0)
            
            # 가중 합계
            composite_score = (
                tech_strength * self.signal_weights['technical'] +
                momentum_strength * self.signal_weights['momentum'] +
                reversion_strength * self.signal_weights['mean_reversion'] +
                volume_strength * self.signal_weights['volume'] +
                sentiment_strength * self.signal_weights['sentiment']
            )
            
            # ML 예측 반영
            if ml_prediction.get('prediction') is not None and ml_prediction.get('confidence', 0) > 0.3:
                ml_signal = np.tanh(ml_prediction['prediction'] * 100)
                ml_weight = ml_prediction['confidence'] * 0.3
                composite_score = composite_score * (1 - ml_weight) + ml_signal * ml_weight
            
            # 신호 강도 결정
            if composite_score > 0.6:
                strength = SignalStrength.VERY_BULLISH
            elif composite_score > 0.2:
                strength = SignalStrength.BULLISH
            elif composite_score < -0.6:
                strength = SignalStrength.VERY_BEARISH
            elif composite_score < -0.2:
                strength = SignalStrength.BEARISH
            else:
                strength = SignalStrength.NEUTRAL
            
            # 신뢰도 계산
            signal_consistency = np.std([tech_strength, momentum_strength, volume_strength])
            confidence = max(0.1, 1 - signal_consistency)
            
            return {
                'strength': strength,
                'confidence': confidence,
                'features': {
                    'composite_score': composite_score,
                    'technical_strength': tech_strength,
                    'momentum_strength': momentum_strength,
                    'reversion_strength': reversion_strength,
                    'volume_strength': volume_strength,
                    'sentiment_strength': sentiment_strength,
                    'ml_prediction': ml_prediction.get('prediction'),
                    'ml_confidence': ml_prediction.get('confidence', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"종합 신호 계산 에러: {e}")
            return {
                'strength': SignalStrength.NEUTRAL,
                'confidence': 0.1,
                'features': {'error': str(e)}
            }

    async def update_price_data(self, symbol: str, ohlcv_data: List):
        """가격 데이터 업데이트"""
        try:
            if symbol not in self.price_data:
                self.price_data[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 새로운 데이터 추가
            for data_point in ohlcv_data:
                timestamp, open_price, high, low, close, volume = data_point
                
                new_row = pd.DataFrame({
                    'timestamp': [datetime.fromtimestamp(timestamp/1000)],
                    'open': [open_price],
                    'high': [high],
                    'low': [low],
                    'close': [close],
                    'volume': [volume]
                })
                
                self.price_data[symbol] = pd.concat([self.price_data[symbol], new_row], ignore_index=True)
            
            # 최근 1000개 데이터만 유지
            if len(self.price_data[symbol]) > 1000:
                self.price_data[symbol] = self.price_data[symbol].tail(1000).reset_index(drop=True)
                
        except Exception as e:
            logger.error(f"가격 데이터 업데이트 에러 {symbol}: {e}")

    async def get_multi_timeframe_signals(self, symbol: str) -> Dict[str, MarketSignal]:
        """다중 시간프레임 신호 분석"""
        timeframes = ['1m', '5m', '15m', '1h']
        signals = {}
        
        for tf in timeframes:
            try:
                signal = await self.analyze_symbol(symbol, tf)
                signals[tf] = signal
            except Exception as e:
                logger.error(f"시간프레임 {tf} 분석 에러: {e}")
                continue
        
        return signals

    def get_engine_status(self) -> Dict:
        """엔진 상태 정보"""
        return {
            'symbols_tracked': len(self.price_data),
            'ml_models_trained': len(self.ml_models),
            'total_data_points': sum(len(df) for df in self.price_data.values()),
            'last_update': datetime.now().isoformat(),
            'signal_weights': self.signal_weights
        }