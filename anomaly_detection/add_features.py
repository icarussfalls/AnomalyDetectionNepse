import pandas as pd
import numpy as np

def get_dimensionless_features(df, brokers):
    """
    Extract dimensionless broker features for trading data that generalize across stocks
    All features are normalized/relative to make them stock-agnostic
    """

    df = df.copy()
    
    # Validate single stock
    if df['Symbol'].nunique() > 1:
        print(f"Warning: Multiple stocks detected: {df['Symbol'].unique()}")
        print("Consider filtering to single stock for better analysis")
    
    stock_symbol = df['Symbol'].iloc[0]
    broker_features = []
    
    # Ensure Rate and Quantity are numeric, raise error if not
    for col in ['Rate', 'Quantity']:
        # Remove commas (thousands separators)
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        if not pd.api.types.is_numeric_dtype(df[col]):
            bad_mask = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
            if bad_mask.any():
                bad_vals = df.loc[bad_mask, col].unique()
                raise ValueError(f"Non-numeric values found in '{col}': {bad_vals}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Pre-calculate stock-specific reference metrics (for normalization only)
    stock_vwap = (df['Rate'] * df['Quantity']).sum() / df['Quantity'].sum()
    stock_mean_rate = df['Rate'].mean()
    stock_rate_std = df['Rate'].std()
    stock_min_rate = df['Rate'].min()
    stock_max_rate = df['Rate'].max()
    stock_rate_range = stock_max_rate - stock_min_rate
    stock_mean_quantity = df['Quantity'].mean()
    stock_quantity_std = df['Quantity'].std()
    total_stock_volume = df['Quantity'].sum()
    total_stock_transactions = len(df)
    
    # Calculate stock-specific percentiles for normalization
    stock_rate_quartiles = df['Rate'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    stock_quantity_quartiles = df['Quantity'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    
    for broker in brokers:
        buys = df[df['Buyer'] == broker].copy()
        sells = df[df['Seller'] == broker].copy()
        all_trades = pd.concat([buys, sells]).sort_values('Transaction_No')
        
        # === DIMENSIONLESS PARTICIPATION METRICS ===
        total_buy_quantity = buys['Quantity'].sum()
        total_sell_quantity = sells['Quantity'].sum()
        total_broker_volume = total_buy_quantity + total_sell_quantity
        
        # Market share (already dimensionless)
        volume_market_share = total_broker_volume / total_stock_volume if total_stock_volume > 0 else 0
        transaction_market_share = len(all_trades) / total_stock_transactions if total_stock_transactions > 0 else 0
        
        # Buy/sell balance (dimensionless ratio)
        if total_broker_volume > 0:
            buy_sell_ratio = total_buy_quantity / total_sell_quantity if total_sell_quantity > 0 else float('inf')
            buy_volume_fraction = total_buy_quantity / total_broker_volume
            sell_volume_fraction = total_sell_quantity / total_broker_volume
            net_volume_fraction = (total_buy_quantity - total_sell_quantity) / total_broker_volume
        else:
            buy_sell_ratio = 0
            buy_volume_fraction = 0
            sell_volume_fraction = 0
            net_volume_fraction = 0
        
        # === DIMENSIONLESS PRICE POSITIONING ===
        
        # Where broker trades relative to stock's price distribution (percentile-based)
        if not buys.empty:
            buy_price_percentiles = []
            for rate in buys['Rate']:
                percentile = (df['Rate'] <= rate).mean()
                buy_price_percentiles.append(percentile)
            avg_buy_price_percentile = np.mean(buy_price_percentiles)
            
            # Price level concentration
            buy_low_percentile = (np.array(buy_price_percentiles) <= 0.25).mean()
            buy_high_percentile = (np.array(buy_price_percentiles) >= 0.75).mean()
            buy_price_spread = np.std(buy_price_percentiles)  # How spread out are the buy prices
        else:
            avg_buy_price_percentile = 0.5
            buy_low_percentile = 0
            buy_high_percentile = 0
            buy_price_spread = 0
            
        if not sells.empty:
            sell_price_percentiles = []
            for rate in sells['Rate']:
                percentile = (df['Rate'] <= rate).mean()
                sell_price_percentiles.append(percentile)
            avg_sell_price_percentile = np.mean(sell_price_percentiles)
            
            sell_low_percentile = (np.array(sell_price_percentiles) <= 0.25).mean()
            sell_high_percentile = (np.array(sell_price_percentiles) >= 0.75).mean()
            sell_price_spread = np.std(sell_price_percentiles)
        else:
            avg_sell_price_percentile = 0.5
            sell_low_percentile = 0
            sell_high_percentile = 0
            sell_price_spread = 0
        
        # === DIMENSIONLESS TRADE SIZE PATTERNS ===
        
        if not buys.empty:
            # Normalize trade sizes by stock's distribution
            buy_size_percentiles = []
            for qty in buys['Quantity']:
                percentile = (df['Quantity'] <= qty).mean()
                buy_size_percentiles.append(percentile)
            avg_buy_size_percentile = np.mean(buy_size_percentiles)
            buy_large_trades_fraction = (np.array(buy_size_percentiles) >= 0.75).mean()
            buy_small_trades_fraction = (np.array(buy_size_percentiles) <= 0.25).mean()
            buy_size_consistency = 1 - np.std(buy_size_percentiles)  # Higher = more consistent sizing
        else:
            avg_buy_size_percentile = 0.5
            buy_large_trades_fraction = 0
            buy_small_trades_fraction = 0
            buy_size_consistency = 0
            
        if not sells.empty:
            sell_size_percentiles = []
            for qty in sells['Quantity']:
                percentile = (df['Quantity'] <= qty).mean()
                sell_size_percentiles.append(percentile)
            avg_sell_size_percentile = np.mean(sell_size_percentiles)
            sell_large_trades_fraction = (np.array(sell_size_percentiles) >= 0.75).mean()
            sell_small_trades_fraction = (np.array(sell_size_percentiles) <= 0.25).mean()
            sell_size_consistency = 1 - np.std(sell_size_percentiles)
        else:
            avg_sell_size_percentile = 0.5
            sell_large_trades_fraction = 0
            sell_small_trades_fraction = 0
            sell_size_consistency = 0
        
        # === DIMENSIONLESS TIMING PATTERNS ===
        
        # Transaction timing distribution (relative to session)
        if not all_trades.empty:
            transaction_numbers = all_trades['Transaction_No'].astype(int)
            session_percentiles = []
            for txn in transaction_numbers:
                percentile = (df['Transaction_No'].astype(int) <= txn).mean()
                session_percentiles.append(percentile)
            
            avg_session_timing = np.mean(session_percentiles)
            early_session_fraction = (np.array(session_percentiles) <= 0.33).mean()
            late_session_fraction = (np.array(session_percentiles) >= 0.67).mean()
            session_timing_spread = np.std(session_percentiles)
        else:
            avg_session_timing = 0.5
            early_session_fraction = 0
            late_session_fraction = 0
            session_timing_spread = 0
        
        # === DIMENSIONLESS MOMENTUM AND TREND ALIGNMENT ===
        
        # Safe correlation calculation
        def safe_correlation(x, y):
            if len(x) < 2 or len(y) < 2:
                return 0
            if np.std(x) == 0 or np.std(y) == 0:
                return 0
            try:
                corr = np.corrcoef(x, y)[0,1]
                return corr if not np.isnan(corr) else 0
            except:
                return 0
        
        # Time-price momentum alignment (already dimensionless correlation)
        if not buys.empty:
            buy_time_momentum = safe_correlation(
                buys['Transaction_No'].astype(int).values, 
                buys['Rate'].values
            )
        else:
            buy_time_momentum = 0
            
        if not sells.empty:
            sell_time_momentum = safe_correlation(
                sells['Transaction_No'].astype(int).values, 
                sells['Rate'].values
            )
        else:
            sell_time_momentum = 0
        
        # === DIMENSIONLESS EXECUTION QUALITY ===
        
        # Price improvement relative to VWAP (normalized by stock's volatility)
        if stock_rate_std > 0:
            if not buys.empty:
                buy_vwap_deviation = (stock_vwap - buys['Rate'].mean()) / stock_rate_std
            else:
                buy_vwap_deviation = 0
                
            if not sells.empty:
                sell_vwap_deviation = (sells['Rate'].mean() - stock_vwap) / stock_rate_std
            else:
                sell_vwap_deviation = 0
        else:
            buy_vwap_deviation = 0
            sell_vwap_deviation = 0
        
        # Combined execution quality score
        execution_quality_score = (buy_vwap_deviation + sell_vwap_deviation) / 2
        
        # === DIMENSIONLESS MARKET MAKING INDICATORS ===
        
        # Two-sided trading (binary)
        two_sided_trading = int(not buys.empty and not sells.empty)
        
        # Inventory turnover (already dimensionless ratio)
        if total_broker_volume > 0:
            inventory_turnover_ratio = min(total_buy_quantity, total_sell_quantity) * 2 / total_broker_volume
        else:
            inventory_turnover_ratio = 0
        
        # Spread capture (normalized by stock's range)
        if not buys.empty and not sells.empty and stock_rate_range > 0:
            spread_capture = (sells['Rate'].mean() - buys['Rate'].mean()) / stock_rate_range
        else:
            spread_capture = 0
        
        # === DIMENSIONLESS VOLATILITY INTERACTION ===
        
        # Price volatility of broker's trades relative to market
        if stock_rate_std > 0:
            if not buys.empty and len(buys) > 1:
                buy_price_volatility_ratio = buys['Rate'].std() / stock_rate_std
            else:
                buy_price_volatility_ratio = 0
                
            if not sells.empty and len(sells) > 1:
                sell_price_volatility_ratio = sells['Rate'].std() / stock_rate_std
            else:
                sell_price_volatility_ratio = 0
        else:
            buy_price_volatility_ratio = 0
            sell_price_volatility_ratio = 0
        
        # === DIMENSIONLESS BEHAVIORAL PATTERNS ===
        
        # Trade frequency consistency (coefficient of variation of inter-trade intervals)
        if len(all_trades) > 2:
            trade_intervals = np.diff(all_trades['Transaction_No'].astype(int))
            if len(trade_intervals) > 0 and np.mean(trade_intervals) > 0:
                trade_frequency_consistency = 1 - (np.std(trade_intervals) / np.mean(trade_intervals))
            else:
                trade_frequency_consistency = 0
        else:
            trade_frequency_consistency = 0
        
        # Self-trading indicators
        self_trades = df[(df['Buyer'] == broker) & (df['Seller'] == broker)]
        self_trading_ratio = len(self_trades) / len(all_trades) if len(all_trades) > 0 else 0
        self_volume_ratio = self_trades['Quantity'].sum() / total_broker_volume if total_broker_volume > 0 else 0
        
        # === DIMENSIONLESS CONCENTRATION METRICS ===
        
        # Gini coefficient for trade size distribution (measures inequality)
        def gini_coefficient(values):
            if len(values) == 0:
                return 0
            values = np.array(values)
            values = np.sort(values)
            n = len(values)
            cumsum = np.cumsum(values)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        if not all_trades.empty:
            trade_size_gini = gini_coefficient(all_trades['Quantity'])
        else:
            trade_size_gini = 0
        
        # Compile all dimensionless features
        broker_feature = {
            'Broker': broker,
            'Stock': stock_symbol,
            
            # Market participation (dimensionless ratios)
            'volume_market_share': volume_market_share,
            'transaction_market_share': transaction_market_share,
            'buy_volume_fraction': buy_volume_fraction,
            'sell_volume_fraction': sell_volume_fraction,
            'net_volume_fraction': net_volume_fraction,
            'buy_sell_ratio': min(buy_sell_ratio, 10),  # Cap extreme ratios
            
            # Price positioning (percentile-based)
            'avg_buy_price_percentile': avg_buy_price_percentile,
            'avg_sell_price_percentile': avg_sell_price_percentile,
            'buy_low_percentile_fraction': buy_low_percentile,
            'buy_high_percentile_fraction': buy_high_percentile,
            'sell_low_percentile_fraction': sell_low_percentile,
            'sell_high_percentile_fraction': sell_high_percentile,
            'buy_price_spread': buy_price_spread,
            'sell_price_spread': sell_price_spread,
            
            # Trade sizing (percentile-based)
            'avg_buy_size_percentile': avg_buy_size_percentile,
            'avg_sell_size_percentile': avg_sell_size_percentile,
            'buy_large_trades_fraction': buy_large_trades_fraction,
            'sell_large_trades_fraction': sell_large_trades_fraction,
            'buy_small_trades_fraction': buy_small_trades_fraction,
            'sell_small_trades_fraction': sell_small_trades_fraction,
            'buy_size_consistency': buy_size_consistency,
            'sell_size_consistency': sell_size_consistency,
            'trade_size_gini': trade_size_gini,
            
            # Timing patterns (percentile-based)
            'avg_session_timing': avg_session_timing,
            'early_session_fraction': early_session_fraction,
            'late_session_fraction': late_session_fraction,
            'session_timing_spread': session_timing_spread,
            'trade_frequency_consistency': trade_frequency_consistency,
            
            # Momentum and trends (correlation coefficients)
            'buy_time_momentum': buy_time_momentum,
            'sell_time_momentum': sell_time_momentum,
            
            # Execution quality (standardized deviations)
            'buy_vwap_deviation': buy_vwap_deviation,
            'sell_vwap_deviation': sell_vwap_deviation,
            'execution_quality_score': execution_quality_score,
            
            # Market making (ratios and spreads)
            'two_sided_trading': two_sided_trading,
            'inventory_turnover_ratio': inventory_turnover_ratio,
            'spread_capture': spread_capture,
            
            # Volatility interaction (relative volatilities)
            'buy_price_volatility_ratio': buy_price_volatility_ratio,
            'sell_price_volatility_ratio': sell_price_volatility_ratio,
            
            # Behavioral indicators (ratios)
            'self_trading_ratio': self_trading_ratio,
            'self_volume_ratio': self_volume_ratio,
        }
        
        broker_features.append(broker_feature)
    
    return pd.DataFrame(broker_features)


def attention_enrich_features(F: np.ndarray) -> np.ndarray:
    """
    Apply non-trainable self-attention to enrich broker feature matrix.
    
    Parameters:
    - F: np.ndarray of shape [n_brokers, d_features]
    
    Returns:
    - np.ndarray of same shape [n_brokers, d_features] with enriched features
    """
    if F.shape[0] < 2:
        return F  # no enrichment needed for a single broker

    Q = F
    K = F
    V = F
    dk = Q.shape[1]

    # --- Compute raw attention scores ---
    scores = Q @ K.T / np.sqrt(dk)  # shape [n_brokers, n_brokers]

    # --- Apply row-wise softmax with numerical stability ---
    scores -= np.max(scores, axis=1, keepdims=True)  # stability trick
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)

    # --- Enrich features via weighted combination ---
    enriched = weights @ V  # shape [n_brokers, d_features]

    # --- Residual connection ---
    return F + enriched