import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
st.title("🚀 Predictor de Criptomonedas - LSTM + Monte Carlo")
st.markdown("*Predicción en tiempo real para scalping 1-15 minutos*")

# ============================================================================
# CLASE PREDICTOR (simplificada para Streamlit)
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
        
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.model = None
    
    def fetch_data(self):
        """Obtiene datos del mercado"""
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def fetch_orderflow(self):
        """Order flow simplificado"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=20)
            bid_vol = sum([b[1] for b in orderbook['bids'][:10]])
            ask_vol = sum([a[1] for a in orderbook['asks'][:10]])
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
            
            trades = self.exchange.fetch_trades(self.symbol, limit=50)
            buy_vol = sum([t['amount'] for t in trades if t['side'] == 'buy'])
            sell_vol = sum([t['amount'] for t in trades if t['side'] == 'sell'])
            aggressor_imb = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0
            
            return {'imbalance': imbalance, 'aggressor_imb': aggressor_imb}
        except:
            return {'imbalance': 0, 'aggressor_imb': 0}
    
    def fetch_funding(self):
        """Funding rate"""
        try:
            funding = self.exchange.fetch_funding_rate(self.symbol)
            return funding['fundingRate'] * 100
        except:
            return 0
    
    def engineer_features(self, df, flow, funding):
        """Crear features"""
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_volume'] = np.log1p(df['volume'])
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['flow_imbalance'] = flow['imbalance']
        df['aggressor_imb'] = flow['aggressor_imb']
        df['funding_rate'] = funding
        df = df.dropna()
        return df
    
    def prepare_sequences(self, df):
        """Secuencias LSTM"""
        feature_cols = ['returns', 'log_volume', 'volatility', 'momentum', 
                       'hl_range', 'flow_imbalance', 'aggressor_imb', 'funding_rate']
        
        prices = df[['close']].values
        features = df[feature_cols].values
        
        self.price_scaler.fit(prices)
        self.feature_scaler.fit(features)
        
        prices_scaled = self.price_scaler.transform(prices)
        features_scaled = self.feature_scaler.transform(features)
        
        X, y = [], []
        for i in range(self.lookback, len(df) - self.pred_minutes):
            X.append(features_scaled[i-self.lookback:i])
            y.append(prices_scaled[i + self.pred_minutes][0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_quick(self):
        """Entrenamiento rápido"""
        df = self.fetch_data()
        flow = self.fetch_orderflow()
        funding = self.fetch_funding()
        df = self.engineer_features(df, flow, funding)
        X, y = self.prepare_sequences(df)
        
        self.model = self.build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0, validation_split=0.2)
    
    def predict(self):
        """Predicción completa"""
        df = self.fetch_data()
        flow = self.fetch_orderflow()
        funding = self.fetch_funding()
        
        current_price = df['close'].iloc[-1]
        df = self.engineer_features(df, flow, funding)
        volatility = df['volatility'].iloc[-1]
        
        # Última secuencia
        feature_cols = ['returns', 'log_volume', 'volatility', 'momentum', 
                       'hl_range', 'flow_imbalance', 'aggressor_imb', 'funding_rate']
        last_seq = df[feature_cols].iloc[-self.lookback:].values
        last_seq_scaled = self.feature_scaler.transform(last_seq)
        X_pred = last_seq_scaled.reshape(1, self.lookback, len(feature_cols))
        
        # LSTM prediction
        lstm_pred_scaled = self.model.predict(X_pred, verbose=0)[0][0]
        lstm_pred = self.price_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
        
        # Monte Carlo
        drift = (lstm_pred - current_price) / current_price
        dt = self.pred_minutes / (24 * 60)
        random_shocks = np.random.normal(0, volatility * np.sqrt(dt), 1000)
        mc_prices = current_price * (1 + drift + random_shocks)
        
        p10 = np.percentile(mc_prices, 10)
        p25 = np.percentile(mc_prices, 25)
        p50 = np.percentile(mc_prices, 50)
        p75 = np.percentile(mc_prices, 75)
        p90 = np.percentile(mc_prices, 90)
        
        # Señal
        expected_move = (p50 - current_price) / current_price * 100
        
        if expected_move > 0.1 and flow['imbalance'] > 0.1:
            signal = "🟢 LONG"
            target = p75
            stop = p25
        elif expected_move < -0.1 and flow['imbalance'] < -0.1:
            signal = "🔴 SHORT"
            target = p25
            stop = p75
        else:
            signal = "⚪ NO OPERAR"
            target = None
            stop = None
        
        return {
            'current_price': current_price,
            'lstm_pred': lstm_pred,
            'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
            'flow': flow, 'funding': funding, 'volatility': volatility,
            'signal': signal, 'target': target, 'stop': stop,
            'mc_prices': mc_prices
        }

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración")

symbol = st.sidebar.selectbox(
    "Selecciona Criptomoneda",
    ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
)

pred_minutes = st.sidebar.slider(
    "Minutos de Predicción",
    min_value=1, max_value=15, value=5
)

auto_refresh = st.sidebar.checkbox("Auto-refresh cada 60 segundos", value=False)

# Botón de predicción
if st.sidebar.button("🚀 PREDECIR AHORA", type="primary"):
    
    with st.spinner(f'Analizando {symbol}...'):
        
        # Inicializar predictor
        predictor = CryptoPredictor(symbol, pred_minutes)
        
        # Entrenar (mostrar progreso)
        progress_bar = st.progress(0)
        st.info("📊 Obteniendo datos del mercado...")
        progress_bar.progress(25)
        
        st.info("🧠 Entrenando modelo LSTM...")
        predictor.train_quick()
        progress_bar.progress(75)
        
        st.info("🎲 Ejecutando simulaciones Monte Carlo...")
        result = predictor.predict()
        progress_bar.progress(100)
        
        st.success("✅ Análisis completado!")
        
        # Guardar en session state
        st.session_state['result'] = result
        st.session_state['symbol'] = symbol
        st.session_state['pred_minutes'] = pred_minutes
        st.session_state['timestamp'] = datetime.now()

# Mostrar resultados si existen
if 'result' in st.session_state:
    result = st.session_state['result']
    
    # Header con timestamp
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"📊 Análisis de {st.session_state['symbol']}")
    with col2:
        st.caption(f"⏰ {st.session_state['timestamp'].strftime('%H:%M:%S')}")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Precio Actual",
            f"${result['current_price']:,.2f}"
        )
    
    with col2:
        lstm_change = (result['lstm_pred'] - result['current_price']) / result['current_price'] * 100
        st.metric(
            f"LSTM ({pred_minutes}min)",
            f"${result['lstm_pred']:,.2f}",
            f"{lstm_change:+.2f}%"
        )
    
    with col3:
        p50_change = (result['p50'] - result['current_price']) / result['current_price'] * 100
        st.metric(
            "MC Mediana (P50)",
            f"${result['p50']:,.2f}",
            f"{p50_change:+.2f}%"
        )
    
    with col4:
        # Señal destacada
        signal_color = "🟢" if "LONG" in result['signal'] else "🔴" if "SHORT" in result['signal'] else "⚪"
        st.metric("Señal", result['signal'])
    
    # Gráfico de distribución Monte Carlo
    st.subheader("📈 Distribución de Precios Posibles (Monte Carlo)")
    
    fig = go.Figure()
    
    # Histograma
    fig.add_trace(go.Histogram(
        x=result['mc_prices'],
        name="Distribución MC",
        marker_color='rgba(100, 149, 237, 0.5)',
        nbinsx=50
    ))
    
    # Líneas de percentiles
    fig.add_vline(x=result['current_price'], line_dash="dash", line_color="white", 
                  annotation_text="Precio Actual")
    fig.add_vline(x=result['p10'], line_dash="dot", line_color="red", 
                  annotation_text="P10")
    fig.add_vline(x=result['p50'], line_dash="solid", line_color="blue", 
                  annotation_text="P50")
    fig.add_vline(x=result['p90'], line_dash="dot", line_color="green", 
                  annotation_text="P90")
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
        xaxis_title="Precio",
        yaxis_title="Frecuencia"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de percentiles
    st.subheader("📊 Rangos de Precios Probables")
    
    percentile_data = {
        'Escenario': ['P10 (Pesimista)', 'P25', 'P50 (Mediana)', 'P75', 'P90 (Optimista)'],
        'Precio': [
            f"${result['p10']:,.2f}",
            f"${result['p25']:,.2f}",
            f"${result['p50']:,.2f}",
            f"${result['p75']:,.2f}",
            f"${result['p90']:,.2f}"
        ],
        'Cambio %': [
            f"{(result['p10']/result['current_price']-1)*100:+.2f}%",
            f"{(result['p25']/result['current_price']-1)*100:+.2f}%",
            f"{(result['p50']/result['current_price']-1)*100:+.2f}%",
            f"{(result['p75']/result['current_price']-1)*100:+.2f}%",
            f"{(result['p90']/result['current_price']-1)*100:+.2f}%"
        ]
    }
    
    st.dataframe(percentile_data, use_container_width=True, hide_index=True)
    
    # Microestructura del mercado
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Microestructura")
        
        flow_status = "🟢 COMPRADOR" if result['flow']['imbalance'] > 0.1 else \
                     "🔴 VENDEDOR" if result['flow']['imbalance'] < -0.1 else "⚪ NEUTRAL"
        
        st.write(f"**Order Flow:** {flow_status}")
        st.write(f"- Imbalance: {result['flow']['imbalance']:+.3f}")
        st.write(f"- Aggressor: {result['flow']['aggressor_imb']:+.3f}")
        st.write(f"**Funding Rate:** {result['funding']:+.4f}%")
        st.write(f"**Volatilidad:** {result['volatility']*100:.2f}%")
    
    with col2:
        st.subheader("🎯 Setup de Trading")
        
        if result['target'] and result['stop']:
            st.write(f"**Señal:** {result['signal']}")
            st.write(f"**Entry:** ${result['current_price']:,.2f}")
            st.write(f"**Target:** ${result['target']:,.2f} ({(result['target']/result['current_price']-1)*100:+.2f}%)")
            st.write(f"**Stop:** ${result['stop']:,.2f} ({(result['stop']/result['current_price']-1)*100:+.2f}%)")
            
            rr = abs((result['target']-result['current_price'])/(result['current_price']-result['stop']))
            st.write(f"**R/R Ratio:** {rr:.2f}")
        else:
            st.warning("⚠️ No hay setup claro. Condiciones de mercado no favorables.")
    
    # Warning
    st.warning("⚠️ **IMPORTANTE:** Esta es una predicción estadística basada en datos históricos. NO es una garantía. Siempre usa stop loss y gestión de riesgo adecuada.")

# Auto-refresh
if auto_refresh and 'result' in st.session_state:
    time.sleep(60)
    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("📈 Crypto Scalping Predictor v1.0")
st.sidebar.caption("LSTM + Monte Carlo + Order Flow")
