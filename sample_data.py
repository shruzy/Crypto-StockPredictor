"""
Sample data generation for cryptocurrency and stock prediction models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_crypto_sample_data(name="Aave", symbol="aave"):
    """Generate sample cryptocurrency data for prediction"""
    
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
    
    df = pd.DataFrame(data)
    
    # Display sample data
    print(f"\n📊 Generated sample data for {name} ({symbol.upper()}):")
    print(f"Shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nLast 2 rows:")
    print(df.tail(2).to_string())
    
    return df

def generate_stock_sample_data(name="Amazon.com", symbol="AMZN"):
    """Generate sample stock data for prediction"""
    
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
    
    df = pd.DataFrame(data)
    
    # Display sample data
    print(f"\n📈 Generated sample data for {name} ({symbol}):")
    print(f"Shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nLast 2 rows:")
    print(df.tail(2).to_string())
    
    return df

def test_sample_generation():
    """Test the sample data generation functions"""
    print("="*60)
    print("TESTING SAMPLE DATA GENERATION")
    print("="*60)
    
    # Test cryptocurrency data
    crypto_data = generate_crypto_sample_data("Aave", "aave")
    
    print("\n" + "="*60)
    
    # Test stock data
    stock_data = generate_stock_sample_data("Amazon.com", "AMZN")
    
    print("\n" + "="*60)
    print("SAMPLE DATA GENERATION COMPLETE!")
    print("="*60)
    
    return crypto_data, stock_data

if __name__ == "__main__":
    # Run test when this file is executed directly
    crypto_df, stock_df = test_sample_generation()
    
    # Save sample data to CSV files for reference
    crypto_df.to_csv("sample_crypto_data.csv", index=False)
    stock_df.to_csv("sample_stock_data.csv", index=False)
    print("\n💾 Sample data saved to 'sample_crypto_data.csv' and 'sample_stock_data.csv'")