import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Crypto Scalping Predictor",
    page_icon="📈",
    layout="wide"
)

# Título
st.title("🚀 Predictor de Criptomonedas - Análisis Técnico + Monte Carlo")
st.markdown("*Predicción en tiempo real para scalping 1-15 minutos*")

# ============================================================================
# CLASE PREDICTOR SIMPLIFICADA (SIN TENSORFLOW)
# ============================================================================

class CryptoPredictor:
    def __init__(self, symbol, prediction_minutes):
        self.symbol = symbol
        self.pred_minutes = prediction_minutes
        self.lookback = 60
        
        # Exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    
    def fetch_data(self):
        """Obtiene datos del mercado"""
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def fetch_orderflow(self):
        """Order flow en tiempo real"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=20)
            bid_vol = sum([b[1] for b in orderbook['bids'][:10]])
            ask_vol = sum([a[1] for a in orderbook['asks'][:10]])
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
            
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            spread_pct = (best_ask - best_bid) / best_bid * 100
            
            trades = self.exchange.fetch_trades(self.symbol, limit=50)
            buy_vol = sum([t['amount'] for t in trades if t['side'] == 'buy'])
            sell_vol = sum([t['amount'] for t in trades if t['side'] == 'sell'])
            aggressor_imb = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0
            
            return {
                'imbalance': imbalance, 
                'aggressor_imb': aggressor_imb,
                'spread_pct': spread_pct
            }
        except:
            return {'imbalance': 0, 'aggressor_imb': 0, 'spread_pct': 0}
    
    def fetch_funding(self):
        """Funding rate"""
        try:
            funding = self.exchange.fetch_funding_rate(self.symbol)
            return funding['fundingRate'] * 100
        except:
            return 0
    
    def calculate_technical_indicators(self, df):
        """Calcula indicadores técnicos avanzados"""
        # Returns y momentum
        df['returns'] = df['close'].pct_change()
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Volatilidad
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_fast'] = df['returns'].rolling(5).std()
        
        # Volume profile
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        # EMA crossovers
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_cross'] = df['ema_9'] - df['ema_21']
        
        df = df.dropna()
        return df
    
    def predict_advanced(self):
        """
        Predicción usando regresión multi-factor + Monte Carlo
        (Sin necesidad de TensorFlow/LSTM)
        """
        # Obtener datos
        df = self.fetch_data()
        flow = self.fetch_orderflow()
        funding = self.fetch_funding()
        
        current_price = df['close'].iloc[-1]
        
        # Calcular indicadores
        df = self.calculate_technical_indicators(df)
        
        # Características actuales
        volatility = df['volatility'].iloc[-1]
        momentum_10 = df['momentum_10'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        bb_position = df['bb_position'].iloc[-1]
        atr = df['atr'].iloc[-1]
        ema_cross = df['ema_cross'].iloc[-1]
        price_vs_vwap = df['price_vs_vwap'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        # === PREDICCIÓN MULTI-FACTOR ===
        
        # Factor 1: Momentum (40% peso)
        momentum_signal = momentum_10 / current_price * 0.40
        
        # Factor 2: RSI mean reversion (15% peso)
        rsi_signal = 0
        if rsi > 70:  # Sobrecompra
            rsi_signal = -0.002 * 0.15
        elif rsi < 30:  # Sobreventa
            rsi_signal = 0.002 * 0.15
        
        # Factor 3: Bollinger position (15% peso)
        bb_signal = 0
        if bb_position > 0.9:  # Cerca de banda superior
            bb_signal = -0.001 * 0.15
        elif bb_position < 0.1:  # Cerca de banda inferior
            bb_signal = 0.001 * 0.15
        
        # Factor 4: Order Flow (20% peso)
        flow_signal = flow['imbalance'] * 0.003 * 0.20
        
        # Factor 5: EMA cross (10% peso)
        ema_signal = (ema_cross / current_price) * 0.10
        
        # Predicción combinada
        combined_signal = momentum_signal + rsi_signal + bb_signal + flow_signal + ema_signal
        
        # Ajuste por tiempo (cuanto más lejos, más incertidumbre)
        time_decay = 1 - (self.pred_minutes / 60)
        adjusted_signal = combined_signal * time_decay
        
        # Precio predicho
        predicted_price = current_price * (1 + adjusted_signal)
        
        # === MONTE CARLO SIMULATION ===
        
        drift = adjusted_signal
        dt = self.pred_minutes / (24 * 60)
        
        # Ajustar volatilidad por order flow (más flujo = menos volátil en dirección)
        adjusted_vol = volatility * (1 - abs(flow['imbalance']) * 0.3)
        
        # Generar escenarios
        n_sims = 1000
        random_shocks = np.random.normal(0, adjusted_vol * np.sqrt(dt), n_sims)
        mc_prices = current_price * (1 + drift + random_shocks)
        
        # Percentiles
        p10 = np.percentile(mc_prices, 10)
        p25 = np.percentile(mc_prices, 25)
        p50 = np.percentile(mc_prices, 50)
        p75 = np.percentile(mc_prices, 75)
        p90 = np.percentile(mc_prices, 90)
        
        # === SEÑAL DE TRADING ===
        
        expected_move = (p50 - current_price) / current_price * 100
        confidence_range = (p75 - p25) / current_price * 100
        
        # Condiciones para señal
        strong_flow = abs(flow['imbalance']) > 0.15
        low_volatility = volatility < df['volatility'].rolling(50).mean().iloc[-1]
        aligned_momentum = (momentum_10 > 0 and flow['imbalance'] > 0) or (momentum_10 < 0 and flow['imbalance'] < 0)
        
        signal = "⚪ NO OPERAR"
        target = None
        stop = None
        
        if expected_move > 0.08 and strong_flow and aligned_momentum:
            signal = "🟢 LONG"
            target = p75
            stop = p25
        elif expected_move < -0.08 and strong_flow and aligned_momentum:
            signal = "🔴 SHORT"
            target = p25
            stop = p75
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
            'flow': flow,
            'funding': funding,
            'volatility': volatility,
            'rsi': rsi,
            'bb_position': bb_position,
            'momentum': momentum_10,
            'volume_ratio': volume_ratio,
            'signal': signal,
            'target': target,
            'stop': stop,
            'mc_prices': mc_prices,
            'confidence_range': confidence_range,
            'indicators': {
                'momentum_signal': momentum_signal * 100,
                'flow_signal': flow_signal * 100,
                'rsi_signal': rsi_signal * 100,
                'bb_signal': bb_signal * 100,
                'ema_signal': ema_signal * 100
            }
        }

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

# Sidebar
st.sidebar.header("⚙️ Configuración")

symbol = st.sidebar.selectbox(
    "Selecciona Criptomoneda",
    ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT"]
)

pred_minutes = st.sidebar.slider(
    "Minutos de Predicción",
    min_value=1, max_value=15, value=5
)

auto_refresh = st.sidebar.checkbox("Auto-refresh cada 60 segundos", value=False)

# Botón de predicción
if st.sidebar.button("🚀 PREDECIR AHORA", type="primary"):
    
    with st.spinner(f'Analizando {symbol}...'):
        
        predictor = CryptoPredictor(symbol, pred_minutes)
        
        progress_bar = st.progress(0)
        st.info("📊 Obteniendo datos del mercado...")
        progress_bar.progress(33)
        
        st.info("🔍 Calculando indicadores técnicos...")
        progress_bar.progress(66)
        
        st.info("🎲 Ejecutando simulaciones Monte Carlo...")
        result = predictor.predict_advanced()
        progress_bar.progress(100)
        
        st.success("✅ Análisis completado!")
        
        st.session_state['result'] = result
        st.session_state['symbol'] = symbol
        st.session_state['pred_minutes'] = pred_minutes
        st.session_state['timestamp'] = datetime.now()

# Mostrar resultados
if 'result' in st.session_state:
    result = st.session_state['result']
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"📊 Análisis de {st.session_state['symbol']}")
    with col2:
        st.caption(f"⏰ {st.session_state['timestamp'].strftime('%H:%M:%S')}")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precio Actual", f"${result['current_price']:,.2f}")
    
    with col2:
        pred_change = (result['predicted_price'] - result['current_price']) / result['current_price'] * 100
        st.metric(
            f"Predicción ({pred_minutes}min)",
            f"${result['predicted_price']:,.2f}",
            f"{pred_change:+.2f}%"
        )
    
    with col3:
        p50_change = (result['p50'] - result['current_price']) / result['current_price'] * 100
        st.metric("MC Mediana", f"${result['p50']:,.2f}", f"{p50_change:+.2f}%")
    
    with col4:
        st.metric("Señal", result['signal'])
    
    # Gráfico Monte Carlo
    st.subheader("📈 Distribución de Precios Posibles (Monte Carlo)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=result['mc_prices'],
        name="Distribución MC",
        marker_color='rgba(100, 149, 237, 0.5)',
        nbinsx=50
    ))
    
    fig.add_vline(x=result['current_price'], line_dash="dash", line_color="white", annotation_text="Actual")
    fig.add_vline(x=result['p10'], line_dash="dot", line_color="red", annotation_text="P10")
    fig.add_vline(x=result['p50'], line_dash="solid", line_color="blue", annotation_text="P50")
    fig.add_vline(x=result['p90'], line_dash="dot", line_color="green", annotation_text="P90")
    
    fig.update_layout(template="plotly_dark", height=400, showlegend=False, xaxis_title="Precio", yaxis_title="Frecuencia")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de percentiles
    st.subheader("📊 Rangos de Precios Probables")
    
    percentile_data = {
        'Escenario': ['P10 (Pesimista)', 'P25', 'P50 (Mediana)', 'P75', 'P90 (Optimista)'],
        'Precio': [f"${p:,.2f}" for p in [result['p10'], result['p25'], result['p50'], result['p75'], result['p90']]],
        'Cambio %': [f"{(p/result['current_price']-1)*100:+.2f}%" for p in [result['p10'], result['p25'], result['p50'], result['p75'], result['p90']]]
    }
    
    st.dataframe(percentile_data, use_container_width=True, hide_index=True)
    
    # Columnas info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Microestructura")
        flow_status = "🟢 COMPRADOR" if result['flow']['imbalance'] > 0.15 else "🔴 VENDEDOR" if result['flow']['imbalance'] < -0.15 else "⚪ NEUTRAL"
        st.write(f"**Order Flow:** {flow_status}")
        st.write(f"- Imbalance: {result['flow']['imbalance']:+.3f}")
        st.write(f"- Aggressor: {result['flow']['aggressor_imb']:+.3f}")
        st.write(f"- Spread: {result['flow']['spread_pct']:.4f}%")
        st.write(f"**Funding Rate:** {result['funding']:+.4f}%")
    
    with col2:
        st.subheader("📊 Indicadores")
        st.write(f"**RSI:** {result['rsi']:.1f}")
        st.write(f"**Volatilidad:** {result['volatility']*100:.2f}%")
        st.write(f"**BB Position:** {result['bb_position']*100:.1f}%")
        st.write(f"**Momentum:** {result['momentum']:+.2f}")
        st.write(f"**Volume Ratio:** {result['volume_ratio']:.2f}x")
    
    with col3:
        st.subheader("🎯 Setup de Trading")
        if result['target'] and result['stop']:
            st.write(f"**Señal:** {result['signal']}")
            st.write(f"**Entry:** ${result['current_price']:,.2f}")
            st.write(f"**Target:** ${result['target']:,.2f} ({(result['target']/result['current_price']-1)*100:+.2f}%)")
            st.write(f"**Stop:** ${result['stop']:,.2f} ({(result['stop']/result['current_price']-1)*100:+.2f}%)")
            rr = abs((result['target']-result['current_price'])/(result['current_price']-result['stop']))
            st.write(f"**R/R:** {rr:.2f}")
        else:
            st.warning("⚠️ Sin setup claro")
    
    # Contribución de factores
    with st.expander("🔬 Análisis Detallado de Factores"):
        st.write("**Contribución de cada factor a la predicción:**")
        indicators = result['indicators']
        for name, value in indicators.items():
            st.write(f"- {name.replace('_', ' ').title()}: {value:+.3f}%")
    
    st.warning("⚠️ **Disclaimer:** Predicción estadística. NO es consejo financiero. Usa siempre stop loss.")

# Auto-refresh
if auto_refresh and 'result' in st.session_state:
    time.sleep(60)
    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("📈 Crypto Scalping Predictor v2.0")
st.sidebar.caption("Análisis Técnico + Monte Carlo + Order Flow")
