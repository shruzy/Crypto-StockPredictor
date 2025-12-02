import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
import traceback
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load models
try:
    crypto_model = joblib.load("best_crypto_model.joblib")
    print("✓ Cryptocurrency model loaded successfully")
except Exception as e:
    print(f"✗ Error loading crypto model: {e}")
    crypto_model = None

try:
    stock_model = joblib.load("best_stock_model.joblib")
    print("✓ Stock model loaded successfully")
except Exception as e:
    print(f"✗ Error loading stock model: {e}")
    stock_model = None

# Sample cryptocurrencies and stocks for dropdown
CRYPTOS = {
    'aave': 'Aave',
    'btc': 'Bitcoin',
    'eth': 'Ethereum',
    'bnb': 'Binance Coin',
    'xrp': 'Ripple',
    'ada': 'Cardano',
    'sol': 'Solana',
    'doge': 'Dogecoin'
}

STOCKS = {
    'AMZN': 'Amazon.com',
    'NVDA': 'NVIDIA',
    'AAPL': 'Apple',
    'GOOGL': 'Alphabet',
    'MSFT': 'Microsoft',
    'TSLA': 'Tesla',
    'META': 'Meta',
    'NFLX': 'Netflix'
}

@app.route('/')
def index():
    """Home page with model selection"""
    return render_template('index.html', cryptos=CRYPTOS, stocks=STOCKS)

@app.route('/predict/crypto', methods=['POST'])
def predict_crypto():
    """Predict cryptocurrency price"""
    try:
        if crypto_model is None:
            return jsonify({'error': 'Cryptocurrency model not loaded'}), 500
        
        # Get data from request
        data = request.json
        crypto_name = data.get('name')
        symbol = data.get('symbol')
        
        # Generate sample data based on crypto selection
        sample_data = generate_crypto_sample_data(crypto_name, symbol)
        
        # Prepare features for prediction
        features = prepare_crypto_features(sample_data)
        
        # Make prediction
        prediction = crypto_model.predict(features)
        
        # Get current price from sample data
        current_price = sample_data.iloc[-1]['price_usd']
        predicted_price = float(prediction[-1])
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        return jsonify({
            'success': True,
            'crypto_name': crypto_name,
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change_percent': round(price_change, 2),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_used': list(features.columns)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict/stock', methods=['POST'])
def predict_stock():
    """Predict stock price"""
    try:
        if stock_model is None:
            return jsonify({'error': 'Stock model not loaded'}), 500
        
        # Get data from request
        data = request.json
        stock_symbol = data.get('symbol')
        stock_name = data.get('name')
        
        # Generate sample data based on stock selection
        sample_data = generate_stock_sample_data(stock_name, stock_symbol)
        
        # Prepare features for prediction
        features = prepare_stock_features(sample_data)
        
        # Make prediction
        prediction = stock_model.predict(features)
        
        # Get current price from sample data
        current_price = abs(sample_data.iloc[-1]['last'])
        predicted_price = abs(float(prediction[-1]))
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        return jsonify({
            'success': True,
            'stock_name': stock_name,
            'symbol': stock_symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change_percent': round(price_change, 2),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_used': list(features.columns)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def generate_crypto_sample_data(name, symbol):
    """Generate sample cryptocurrency data based on selected crypto"""
    # Base values for different cryptocurrencies
    crypto_base_values = {
        'aave': {'base_price': 170, 'vol_multiplier': 300000000},
        'btc': {'base_price': 45000, 'vol_multiplier': 30000000000},
        'eth': {'base_price': 2500, 'vol_multiplier': 15000000000},
        'bnb': {'base_price': 300, 'vol_multiplier': 2000000000},
        'xrp': {'base_price': 0.6, 'vol_multiplier': 2000000000},
        'ada': {'base_price': 0.45, 'vol_multiplier': 1000000000},
        'sol': {'base_price': 100, 'vol_multiplier': 3000000000},
        'doge': {'base_price': 0.08, 'vol_multiplier': 500000000}
    }
    
    base_info = crypto_base_values.get(symbol.lower(), crypto_base_values['aave'])
    
    # Generate 5 recent timestamps
    timestamps = [(datetime.now() - timedelta(minutes=i*30)).strftime('%Y-%m-%d %H:%M:%S') 
                  for i in range(5, 0, -1)]
    
    # Create sample data
    data = []
    base_price = base_info['base_price']
    
    for i, ts in enumerate(timestamps):
        price = base_price * (1 + np.random.uniform(-0.02, 0.02))
        vol_24h = base_info['vol_multiplier'] * np.random.uniform(0.8, 1.2)
        chg_24h = np.random.uniform(-5, 5)
        chg_7d = np.random.uniform(-10, 10)
        market_cap = price * (vol_24h / np.random.uniform(0.1, 0.3))
        
        # Generate technical indicators
        MA7 = price * np.random.uniform(0.98, 1.02)
        MA30 = price * np.random.uniform(0.97, 1.03)
        price_lag1 = price * np.random.uniform(0.99, 1.01) if i > 0 else price * 0.99
        daily_return = ((price - price_lag1) / price_lag1) * 100 if i > 0 else 0
        
        data.append({
            'timestamp': ts,
            'name': name,
            'symbol': symbol,
            'price_usd': round(price, 2),
            'vol_24h': round(vol_24h, 2),
            'total_vol': round(np.random.uniform(0.1, 0.3), 2),
            'chg_24h': round(chg_24h, 2),
            'chg_7d': round(chg_7d, 2),
            'market_cap': round(market_cap, 2),
            'MA7': round(MA7, 2),
            'MA30': round(MA30, 2),
            'price_lag1': round(price_lag1, 2) if i > 0 else price,
            'daily_return': round(daily_return, 4),
            'price_range_proxy': round(vol_24h, 2)
        })
    
    return pd.DataFrame(data)

def generate_stock_sample_data(name, symbol):
    """Generate sample stock data based on selected stock"""
    # Base values for different stocks (using absolute values since your data shows negatives)
    stock_base_values = {
        'AMZN': {'base_price': 175, 'vol_multiplier': 0.2},
        'NVDA': {'base_price': 500, 'vol_multiplier': 0.3},
        'AAPL': {'base_price': 190, 'vol_multiplier': 0.15},
        'GOOGL': {'base_price': 140, 'vol_multiplier': 0.12},
        'MSFT': {'base_price': 380, 'vol_multiplier': 0.18},
        'TSLA': {'base_price': 240, 'vol_multiplier': 0.25},
        'META': {'base_price': 350, 'vol_multiplier': 0.2},
        'NFLX': {'base_price': 600, 'vol_multiplier': 0.1}
    }
    
    base_info = stock_base_values.get(symbol, stock_base_values['AMZN'])
    
    # Generate 5 recent timestamps
    timestamps = [(datetime.now() - timedelta(minutes=i*60)).strftime('%Y-%m-%d %H:%M:%S') 
                  for i in range(5, 0, -1)]
    
    # Create sample data
    data = []
    base_price = -base_info['base_price']  # Negative as per your sample data
    
    for i, ts in enumerate(timestamps):
        last = base_price * (1 + np.random.uniform(-0.01, 0.01))
        high = last * np.random.uniform(0.998, 1.002)
        low = last * np.random.uniform(0.995, 0.999)
        chg_ = np.random.uniform(-3, -1)
        chg_percent = np.random.uniform(-2, -0.5)
        vol_ = np.random.uniform(-0.2, 0.8)
        
        # Generate technical indicators
        MA7 = last * np.random.uniform(0.998, 1.002)
        MA30 = last * np.random.uniform(0.995, 1.005)
        last_lag1 = last * np.random.uniform(0.999, 1.001) if i > 0 else last * 1.001
        last_lag2 = last_lag1 * np.random.uniform(0.999, 1.001) if i > 1 else last_lag1 * 1.001
        high_lag1 = high * np.random.uniform(0.999, 1.001) if i > 0 else high
        low_lag1 = low * np.random.uniform(0.999, 1.001) if i > 0 else low
        returns = np.random.uniform(-0.02, 0.02)
        price_range = np.random.uniform(-0.9, -0.3)
        
        data.append({
            'timestamp': ts,
            'name': name,
            'last': round(last, 6),
            'high': round(high, 6),
            'low': round(low, 6),
            'chg_': round(chg_, 2),
            'chg_%': round(chg_percent, 2),
            'vol_': round(vol_, 6),
            'time': '15:59:59',
            'MA7': round(MA7, 6),
            'MA30': round(MA30, 6),
            'last_lag1': round(last_lag1, 6) if i > 0 else last,
            'last_lag2': round(last_lag2, 6) if i > 1 else last_lag1,
            'high_lag1': round(high_lag1, 6) if i > 0 else high,
            'low_lag1': round(low_lag1, 6) if i > 0 else low,
            'returns': round(returns, 6),
            'range': round(price_range, 6)
        })
    
    return pd.DataFrame(data)

def prepare_crypto_features(df):
    """Prepare features for crypto model prediction"""
    # Select relevant features based on your data sample
    feature_columns = [
        'price_usd', 'vol_24h', 'total_vol', 'chg_24h', 'chg_7d',
        'market_cap', 'MA7', 'MA30', 'price_lag1', 'daily_return',
        'price_range_proxy'
    ]
    
    # Ensure all columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Default value if column missing
    
    return df[feature_columns]

def prepare_stock_features(df):
    """Prepare features for stock model prediction"""
    # Select relevant features based on your data sample
    feature_columns = [
        'last', 'high', 'low', 'chg_', 'chg_%', 'vol_',
        'MA7', 'MA30', 'last_lag1', 'last_lag2', 'high_lag1',
        'low_lag1', 'returns', 'range'
    ]
    
    # Ensure all columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Default value if column missing
    
    return df[feature_columns]

@app.route('/crypto')
def crypto_page():
    """Cryptocurrency prediction page"""
    return render_template('crypto_prediction.html', cryptos=CRYPTOS)

@app.route('/stock')
def stock_page():
    """Stock prediction page"""
    return render_template('stock_prediction.html', stocks=STOCKS)

if __name__ == '__main__':
    app.run(debug=True, port=5000)