#!/usr/bin/env python3
"""
SPY 대비 글로벌 ETF 밸류에이션 — 데이터 업데이트
GitHub Actions에서 매일 실행
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

ZSCORE_WINDOW = 252
HOLD_DAYS = 126
START_DATE = "2000-01-01"
DATA_DIR = "data"

COUNTRY_ETFS = {
    "EPI":  ["인도",     "🇮🇳", "아시아"],
    "EWA":  ["호주",     "🇦🇺", "아시아"],
    "EWC":  ["캐나다",   "🇨🇦", "아메리카"],
    "EWD":  ["스웨덴",   "🇸🇪", "유럽"],
    "EWG":  ["독일",     "🇩🇪", "유럽"],
    "EWH":  ["홍콩",     "🇭🇰", "아시아"],
    "EWI":  ["이탈리아", "🇮🇹", "유럽"],
    "EWJ":  ["일본",     "🇯🇵", "아시아"],
    "EWL":  ["스위스",   "🇨🇭", "유럽"],
    "EWN":  ["네덜란드", "🇳🇱", "유럽"],
    "EWP":  ["스페인",   "🇪🇸", "유럽"],
    "EWQ":  ["프랑스",   "🇫🇷", "유럽"],
    "EWS":  ["싱가포르", "🇸🇬", "아시아"],
    "EWT":  ["대만",     "🇹🇼", "아시아"],
    "EWU":  ["영국",     "🇬🇧", "유럽"],
    "EWW":  ["멕시코",   "🇲🇽", "아메리카"],
    "EWY":  ["한국",     "🇰🇷", "아시아"],
    "EWZ":  ["브라질",   "🇧🇷", "아메리카"],
    "EZA":  ["남아공",   "🇿🇦", "기타"],
    "FXI":  ["중국",     "🇨🇳", "아시아"],
    "THD":  ["태국",     "🇹🇭", "아시아"],
}

def download_all():
    """Download SPY + all ETFs"""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_tickers = ["SPY"] + list(COUNTRY_ETFS.keys())
    
    print(f"Downloading {len(all_tickers)} tickers...")
    
    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df['Close'][ticker] if ticker in df['Close'].columns else df['Close'].iloc[:, 0]
                else:
                    close = df['Close']
                close = close.dropna()
                close.name = ticker
                prices[ticker] = close
                
                # Save raw CSV
                save_df = pd.DataFrame({'Date': close.index.strftime('%Y-%m-%d'), 'Close': close.values})
                save_df.to_csv(f"{DATA_DIR}/{ticker}.csv", index=False)
                print(f"  ✅ {ticker}: {len(close)} days")
            else:
                print(f"  ❌ {ticker}: no data")
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")
    
    return prices

def compute_vs_spy(etf_prices, spy_prices):
    """Compute ETF/SPY ratio and Z-Score"""
    combined = pd.DataFrame({'etf': etf_prices, 'spy': spy_prices}).dropna()
    if len(combined) < ZSCORE_WINDOW + 50:
        return None
    
    combined['ratio'] = combined['etf'] / combined['spy']
    combined['ratio_mean'] = combined['ratio'].rolling(ZSCORE_WINDOW).mean()
    combined['ratio_std'] = combined['ratio'].rolling(ZSCORE_WINDOW).std()
    combined['zscore'] = (combined['ratio'] - combined['ratio_mean']) / combined['ratio_std']
    
    return combined[['etf', 'spy', 'ratio', 'zscore']].dropna()

def backtest(df):
    """Simple backtest: buy when Z <= -2, hold for HOLD_DAYS"""
    trades = []
    i = 0
    while i < len(df) - HOLD_DAYS:
        z = df['zscore'].iloc[i]
        if z <= -2:
            exit_i = i + HOLD_DAYS
            etf_ret = (df['etf'].iloc[exit_i] / df['etf'].iloc[i] - 1) * 100
            spy_ret = (df['spy'].iloc[exit_i] / df['spy'].iloc[i] - 1) * 100
            trades.append({'alpha': etf_ret - spy_ret})
            i += 60
        else:
            i += 1
    
    if trades:
        return {
            'count': len(trades),
            'avg_alpha': round(np.mean([t['alpha'] for t in trades]), 1),
            'win_rate': round(np.mean([1 if t['alpha'] > 0 else 0 for t in trades]) * 100, 0),
        }
    return None

def main():
    print("=" * 60)
    print("  SPY vs Global ETF Valuation Update")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    prices = download_all()
    
    if 'SPY' not in prices:
        print("❌ SPY data not available")
        return
    
    spy = prices['SPY']
    rankings = []
    
    for ticker, info in COUNTRY_ETFS.items():
        if ticker not in prices:
            continue
        
        result = compute_vs_spy(prices[ticker], spy)
        if result is None:
            continue
        
        # Save _vs_SPY CSV
        save_df = result.copy()
        save_df.index = save_df.index.strftime('%Y-%m-%d')
        save_df.index.name = 'Date'
        save_df.to_csv(f"{DATA_DIR}/{ticker}_vs_SPY.csv")
        
        # Backtest
        bt = backtest(result)
        
        current_z = round(float(result['zscore'].iloc[-1]), 2)
        current_ratio = round(float(result['ratio'].iloc[-1]), 4)
        last_date = result.index[-1].strftime('%Y-%m-%d')
        
        rankings.append({
            'ticker': ticker,
            'name': info[0],
            'flag': info[1],
            'region': info[2],
            'zscore': current_z,
            'ratio': current_ratio,
            'date': last_date,
            'alpha': bt['avg_alpha'] if bt else '',
            'winrate': bt['win_rate'] if bt else '',
        })
        
        print(f"  📊 {ticker} ({info[0]}): Z={current_z:+.2f}σ")
    
    # Sort by Z-Score and save rankings
    rankings.sort(key=lambda x: x['zscore'])
    pd.DataFrame(rankings).to_csv(f"{DATA_DIR}/current_rankings.csv", index=False)
    
    print(f"\n✅ Updated {len(rankings)} countries")
    print(f"   Most undervalued: {rankings[0]['flag']} {rankings[0]['name']} (Z={rankings[0]['zscore']:+.2f}σ)")
    print(f"   Most overvalued:  {rankings[-1]['flag']} {rankings[-1]['name']} (Z={rankings[-1]['zscore']:+.2f}σ)")

if __name__ == "__main__":
    main()
