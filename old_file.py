import logging
import numpy as np
import pandas as pd
import yfinance as yf
import twstock
from typing import List, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class TaiwanStockHistoricalScanner:
    """台股技術面選股器 - 支援歷史回測"""
    
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
        
        # 突破拉回策略參數
        self.params = {
            'breakout_volume_mult': 2.0,    # 突破量倍數
            'breakout_window': 20,          # 突破高點回看天數
            'pullback_days': 13,            # 拉回觀察天數
            'ma_support': 20,               # 均線支撐
            'atr_limit': 0.04,              # ATR限制（4%）
        }

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

    def _slice_until_date(self, df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        """切片資料到目標日期"""
        if df is None or df.empty:
            return df
            
        # 確保 target_date 是 pandas Timestamp
        if not isinstance(target_date, pd.Timestamp):
            target_date = pd.Timestamp(target_date)
            
        # 處理時區問題
        if target_date.tzinfo is not None:
            target_date = target_date.tz_convert(None)
        
        # 只保留目標日期之前的資料（包含目標日期）
        return df[df.index <= target_date]

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

    def monthly_filter(self, code: str) -> bool:
        """月線趨勢篩選 - 基於目標日期"""
        df = self._download_historical_data(code, years_back=3)
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
            
            # 月線篩選條件
            conditions = [
                current['MA5'] > prev['MA5'],           # 月MA5上升
                current['MA20'] > prev['MA20'],         # 月MA20上升  
                # current['MA5'] > current['MA20'],       # MA5在MA20之上
                1 < current['MA5'] / current['MA20'] <= 1.04,  # 均線糾結
            ]
            
            ratio = current['MA5'] / current['MA20']
            
            # 股價位置（近20個月）
            recent_data = monthly.iloc[max(0, target_idx-19):target_idx+1]
            recent_high = recent_data['High'].max()
            recent_low = recent_data['Low'].min()
            price_position = (current['Close'] - recent_low) / (recent_high - recent_low)
            conditions.append(price_position < 0.7)  # 不在高檔
            
            result = all(conditions)
            if result:
                logging.info(
                    f"{code} ✓ 月線通過 [{current.name.strftime('%Y-%m')}]: "
                    f"MA5={current['MA5']:.2f}, MA20={current['MA20']:.2f}, "
                    f"ratio={ratio:.4f}, price_pos={price_position:.2f}"
                )
            
            return result
            
        except Exception as e:
            logging.warning(f"{code} 月線篩選失敗: {e}")
            return False

    def breakout_pullback_filter(self, code: str) -> bool:
        """突破拉回型態篩選 - 基於目標日期"""
        df = self._download_historical_data(code, years_back=1)
        if df is None or len(df) < 60:
            return False
            
        try:
            # 只使用目標日期之前的資料
            df = df[df.index <= self.target_date]
            
            if len(df) < 60:
                return False
            
            # 計算技術指標
            df['MA20'] = df['Close'].rolling(20).mean()
            df['Volume_MA'] = df['Volume'].rolling(self.params['breakout_window']).mean()
            df['High_Max'] = df['High'].shift(1).rolling(self.params['breakout_window']).max()
            
            # ATR計算
            df['TR'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            df['ATR'] = df['TR'].rolling(14).mean()
            
            # 移除缺失值
            df.dropna(subset=['High', 'High_Max', 'Volume', 'Volume_MA'], inplace=True)            
            if len(df) < 30:
                return False
            
            # 尋找突破點
            breakout_mask = (
                (df['High'] > df['High_Max']) & 
                (df['Volume'] > self.params['breakout_volume_mult'] * df['Volume_MA'])
            )
            
            if not breakout_mask.any():
                return False
                
            # 最近突破日期
            breakout_indices = df[breakout_mask].index
            last_breakout = breakout_indices[-1]
            
            # 確保分析的是目標日期當天的狀況
            if self.target_date not in df.index:
                # 如果目標日期不在交易日中，找最接近的前一個交易日
                available_dates = df.index[df.index <= self.target_date]
                if len(available_dates) == 0:
                    return False
                target_trading_date = available_dates[-1]
            else:
                target_trading_date = self.target_date
            
            current = df.loc[target_trading_date]
            breakout_data = df.loc[last_breakout]
            
            # 拉回條件檢查
            days_since_breakout = len(df.loc[last_breakout:target_trading_date]) - 1
            
            pullback_conditions = [
                34 <= days_since_breakout,  # 至少拉回六週起跳
                current['Close'] < breakout_data['High_Max'],  # 收盤回落
                current['Low'] >= current['MA20'],  # 守住MA20
                (current['ATR'] / current['Close']) <= self.params['atr_limit'],  # 波動可控
            ]
            
            result = all(pullback_conditions)
            if result:
                logging.info(
                    f"{code} ✓ 突破拉回通過 [{target_trading_date.strftime('%Y-%m-%d')}]: "
                    f"突破距今{days_since_breakout}天, 突破日期={last_breakout.strftime('%Y-%m-%d')}"
                )
            
            return result
            
        except Exception as e:
            logging.warning(f"{code} 突破拉回篩選失敗: {e}")
            return False

    def scan_stocks(self, limit: int = None) -> List[str]:
        """執行完整選股流程 - 歷史回測版"""
        codes = self.get_stock_codes()
        if limit:
            codes = codes[:limit]
            
        logging.info(f"🔍 開始在 {self.target_date.strftime('%Y-%m-%d')} 篩選 {len(codes)} 檔股票...")
        
        # 第一階段：月線篩選
        logging.info("執行月線篩選...")
        monthly_passed = []
        failed_downloads = []
        
        for i, code in enumerate(codes, 1):
            if i % 100 == 0:
                logging.info(f"月線篩選進度: {i}/{len(codes)}")
            
            try:
                if self.monthly_filter(code):
                    monthly_passed.append(code)
            except Exception as e:
                failed_downloads.append(code)
                logging.warning(f"{code} 月線篩選失敗: {e}")
                continue
                
        logging.info(f"月線篩選通過: {len(monthly_passed)} 檔")
        if failed_downloads:
            logging.info(f"下載失敗: {len(failed_downloads)} 檔 - {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")
        
        # 第二階段：突破拉回篩選
        logging.info("執行突破拉回篩選...")
        final_candidates = []
        
        for code in monthly_passed:
            try:
                if self.breakout_pullback_filter(code):
                    final_candidates.append(code)
            except Exception as e:
                logging.warning(f"{code} 突破拉回篩選失敗: {e}")
                continue
                
        return final_candidates

    def backtest_multiple_dates(self, date_list: List[str], limit: int = None) -> dict:
        """多日期回測"""
        results = {}
        
        for date_str in date_list:
            logging.info(f"\n{'='*50}")
            logging.info(f"🔄 回測日期: {date_str}")
            logging.info(f"{'='*50}")
            
            # 重設目標日期
            self.target_date = pd.to_datetime(date_str)
            
            # 執行選股
            candidates = self.scan_stocks(limit)
            results[date_str] = candidates
            
            logging.info(f"📊 {date_str} 選股結果: {len(candidates)} 檔")
            
        return results


def main():
    """主程式"""
    
    # 設定要回測的日期
    # target_date = "2025-06-20"  # 可以修改為任何歷史日期 
    target_date: Optional[str] = None    
    # 建立掃描器
    scanner = TaiwanStockHistoricalScanner(target_date=target_date)
    
    # 執行單日選股
    candidates = scanner.scan_stocks()
    
    if candidates:
        logging.info(f"🎯 {target_date} 最終選股結果 ({len(candidates)} 檔): {candidates}")
        
        # 顯示股票名稱
        for code in candidates:
            try:
                stock_name = twstock.codes[code].name if code in twstock.codes else "未知"
                logging.info(f"  {code} - {stock_name}")
            except:
                logging.info(f"  {code} - 無法取得名稱")
    else:
        logging.info(f"❌ {target_date} 無符合條件股票")
    
    print(f"\n📈 歷史選股結果: {candidates}")
    
    # 示範多日期回測
    # date_list = ["2024-01-15", "2024-03-15", "2024-06-15"]
    # results = scanner.backtest_multiple_dates(date_list, limit=20)
    # 
    # print("\n📊 多日期回測結果:")
    # for date, stocks in results.items():
    #     print(f"{date}: {len(stocks)} 檔 - {stocks}")


if __name__ == "__main__":
    main()