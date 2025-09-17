# -*- coding: utf-8 -*-
import csv
import os
import yfinance as yf
from datetime import datetime
from stock import TaiwanStockHistoricalScanner  # 換成你的檔名

CSV_FILE = "stock_results.csv"

def get_current_price(ticker_code: str) -> float:
    """取得當下價格（用 yfinance 最新一筆資料）"""
    ticker = yf.Ticker(ticker_code)
    hist = ticker.history(period="1d", interval="1m")  # 取當日分鐘資料
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    return None

def save_to_csv(codes):
    """將股票代碼與當下價格寫進 CSV（追加模式）"""
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 第一次執行加上 header
        if not file_exists:
            writer.writerow(["股票代碼", "當下價格", "記錄時間"])
        
        for code in codes:
            # 把台股代碼轉成 yfinance 格式
            if code.startswith("6") or code.startswith("1"):  # 上市
                yf_code = f"{code}.TW"
            else:  # 上櫃
                yf_code = f"{code}.TWO"

            price = get_current_price(yf_code)
            if price is not None:
                writer.writerow([code, price, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            else:
                writer.writerow([code, "無法取得", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def main():
    scanner = TaiwanStockHistoricalScanner(target_date=None)
    picks = scanner.scan_stocks()
    if picks:
        save_to_csv(picks)
        print(f"已寫入 {len(picks)} 檔股票到 {CSV_FILE}")
    else:
        print("無符合條件股票")

if __name__ == "__main__":
    main()