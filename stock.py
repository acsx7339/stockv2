import logging
import numpy as np
import pandas as pd
import yfinance as yf
import twstock
from typing import List, Optional
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class GetStockList:

    def __init__(self, target_date: str = None):
        """
        初始化選股器
        Args:
            target_date: 目標日期，格式 'YYYY-MM-DD'，如 '2024-01-15'
                        如果為 None，則使用當前日期
        """
        self.twse_list = set(twstock.twse.keys())
        self.tpex_list = set(twstock.tpex.keys())
        
        # 設定目標日期，確保是 pandas Timestamp
        if target_date:
            self.target_date = pd.Timestamp(target_date)
        else:
            self.target_date = pd.Timestamp.now().normalize()  # 去除時間部分
            
        logging.info(f"🎯 設定目標分析日期: {self.target_date.strftime('%Y-%m-%d')}")
        
        # 資料快取
        self.data_cache = {}
        
    def get_stock_codes(self) -> List[str]:
        """取得所有4位數股票代碼（排除0開頭）"""
        all_codes = self.twse_list | self.tpex_list
        return sorted([code for code in all_codes 
                    if len(code) == 4 and not code.startswith('0')])

    def _get_ticker(self, code: str) -> str:
        """轉換為yfinance ticker格式"""
        if code in self.twse_list:
            return f"{code}.TW"
        elif code in self.tpex_list:
            return f"{code}.TWO"
        else:
            raise ValueError(f"股票代碼 {code} 不在上市/上櫃清單中")
        
    def scan_stocks(self, limit: int = None) -> List[str]:
        """執行完整選股流程 - 歷史回測版"""
        codes = self.get_stock_codes()
        if limit:
            codes = codes[:limit]
        logging.info(f"🔍 開始在 {self.target_date.strftime('%Y-%m-%d')} 篩選 {len(codes)} 檔股票...")
        return codes
    
    
class getStockData():

    def _download_historical_data(self, code: str, years_back: int = 3) -> Optional[pd.DataFrame]:
        """下載歷史資料，從目標日期往前推算指定年數"""
        
        # 檢查快取
        cache_key = f"{code}_{years_back}y"
        if cache_key in self.data_cache:
            return self._slice_until_date(self.data_cache[cache_key], self.target_date)
            
        try:
            ticker = self._get_ticker(code)
            
            # 計算下載範圍
            end_date = self.target_date + timedelta(days=1)  # +1天確保包含目標日期
            start_date = self.target_date - timedelta(days=years_back * 365)
            
            df = yf.download(
                ticker, 
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d', 
                auto_adjust=False, 
                progress=False,
                timeout=30  # 增加超時設定
            )
            
            if df.empty:
                logging.warning(f"{code} 無資料")
                return None
                
            # 修正MultiIndex columns問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 快取完整資料
            self.data_cache[cache_key] = df.copy()
            
            # 返回切片後的資料
            return self._slice_until_date(df, self.target_date)
            
        except Exception as e:
            logging.warning(f"下載 {code} 歷史資料失敗: {e}")
            return None


class monthly_filter():

    def history_data(self, code):
        df = getStockData._download_historical_data(code, years_back=3)
        if df is None:
            return False
        try:
            # 取月線資料（以目標日期為基準）
            close_s = df['Close'].resample('ME').last()
            high_s = df['High'].resample('ME').max()
            low_s = df['Low'].resample('ME').min()

            monthly = pd.concat([close_s, high_s, low_s], axis=1)
            monthly.columns = ['Close', 'High', 'Low']
            monthly.dropna(inplace=True)

            if len(monthly) < 25:  # 至少需要25個月資料
                return False
                
            # 計算月均線
            monthly['MA5'] = monthly['Close'].rolling(5).mean()
            monthly['MA20'] = monthly['Close'].rolling(20).mean()
            monthly.dropna(inplace=True)
            
            if len(monthly) < 2:
                return False
                
            # 找到目標日期當月或最接近的月份
            target_month = self.target_date.to_period('M')
            
            # 找到最接近目標日期的月線資料
            monthly_periods = monthly.index.to_period('M')
            if target_month not in monthly_periods:
                # 如果目標月份不存在，找最接近的前一個月
                available_months = monthly_periods[monthly_periods <= target_month]
                if len(available_months) == 0:
                    return False
                target_month = available_months[-1]
            
            # 獲取目標月份的索引
            target_idx = monthly_periods.get_loc(target_month)
            if target_idx == 0:  # 如果是第一個月，無法比較前一月
                return False
                
            current = monthly.iloc[target_idx]
            prev = monthly.iloc[target_idx - 1]
        except Exception as e:
            logging.warning(f"{code} 月線篩選失敗: {e}")
            return False

    
    
def main():
    target_date = "2025-06-20"  # 可以修改為任何歷史日期 
    # target_date: Optional[str] = None    
    # 建立掃描器
    scanner = GetStockList(target_date=target_date)
    # 執行單日選股
    candidates = scanner.scan_stocks()
    # show all stock list
    logging.info(f"{candidates}")




if __name__ == "__main__":
    main()