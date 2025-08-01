import logging
import numpy as np
import pandas as pd
import yfinance as yf
import twstock
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class TaiwanStockScanner:
    """台股技術面選股器"""
    
    def __init__(self):
        self.twse_list = set(twstock.twse.keys())
        self.tpex_list = set(twstock.tpex.keys())
        
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

    def _download_data(self, code: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """下載股價資料並處理MultiIndex問題"""
        try:
            ticker = self._get_ticker(code)
            df = yf.download(ticker, period=period, interval='1d', auto_adjust=False, progress=False)
            
            if df.empty:
                return None
                
            # 修正MultiIndex columns問題
            if isinstance(df.columns, pd.MultiIndex):
                # 如果是MultiIndex，取第一層級作為列名
                df.columns = df.columns.get_level_values(0)
            
            return df
            
        except Exception as e:
            logging.warning(f"下載 {code} 資料失敗: {e}")
            return None

    def monthly_filter(self, code: str) -> bool:
        """月線趨勢篩選"""
        df = self._download_data(code, period="2y")
        if df is None:
            return False
            
        try:
           # ─── 取月線資料（修正版） ───
            close_s = df['Close'].resample('ME').last()
            high_s  = df['High'].resample('ME').max()
            low_s   = df['Low'].resample('ME').min()

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
                
            current = monthly.iloc[-1]
            prev = monthly.iloc[-2]
            
            # 月線篩選條件
            conditions = [
                current['MA5'] > prev['MA5'],           # 月MA5上升
                current['MA20'] > prev['MA20'],         # 月MA20上升  
                current['MA5'] > current['MA20'],       # MA5在MA20之上
                1 < current['MA5'] / current['MA20'] <= 1.04,  # 均線糾結
            ]
            ratio = current['MA5'] / current['MA20']
            # **加這段 logging**，把數值和結果都印出來
            logging.info(
                f"{code} 月線條件檢查："
                f"MA5_prev={prev['MA5']:.2f}, MA5_curr={current['MA5']:.2f}; "
                f"MA20_prev={prev['MA20']:.2f}, MA20_curr={current['MA20']:.2f}; "
                f"MA5>MA20=({current['MA5']} > {current['MA20']}); "
                f"ratio={ratio:.4f}"
            )

            # 股價位置（近100天）
            recent_high = monthly['High'].tail(20).max()
            recent_low = monthly['Low'].tail(20).min()
            price_position = (current['Close'] - recent_low) / (recent_high - recent_low)
            conditions.append(price_position < 0.7)  # 不在高檔
            
            return all(conditions)
            
        except Exception as e:
            logging.warning(f"{code} 月線篩選失敗: {e}")
            return False

    def breakout_pullback_filter(self, code: str) -> bool:
        """突破拉回型態篩選"""
        df = self._download_data(code, period="6mo")
        if df is None or len(df) < 60:
            return False
            
        try:            
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
            
            # 檢查拉回條件（最近幾天）
            recent_days = min(self.params['pullback_days'], len(df) - df.index.get_loc(last_breakout) - 1)
            if recent_days <= 0:
                return False
                
            current = df.iloc[-1]
            breakout_data = df.loc[last_breakout]
            
            # 拉回條件檢查
            days_since_breakout = len(df) - df.index.get_loc(last_breakout) - 1
            
            pullback_conditions = [
                34 <= days_since_breakout,  # 至少拉回六週起跳
                current['Close'] < breakout_data['High_Max'],                  # 收盤回落
                current['Close'] < breakout_data['High_Max'],              # 拉回中
                # current['Volume'] < current['Volume_MA'],                   # 量縮
                current['Low'] >= current['MA20'],# 守住MA20
                (current['ATR'] / current['Close']) <= self.params['atr_limit'],  # 波動可控
            ]
            
            logging.info(f"{code} 突破拉回條件檢查: {pullback_conditions}")
            return all(pullback_conditions)
            
        except Exception as e:
            logging.warning(f"{code} 突破拉回篩選失敗: {e}")
            return False

    def scan_stocks(self, limit: int = None) -> List[str]:
        """執行完整選股流程"""
        codes = self.get_stock_codes()
        if limit:
            codes = codes[:limit]
            
        logging.info(f"開始篩選 {len(codes)} 檔股票...")
        
        # 第一階段：月線篩選
        logging.info("執行月線篩選...")
        monthly_passed = []
        for i, code in enumerate(codes, 1):
            if i % 100 == 0:
                logging.info(f"月線篩選進度: {i}/{len(codes)}")
                
            if self.monthly_filter(code):
                monthly_passed.append(code)
                
        logging.info(f"月線篩選通過: {len(monthly_passed)} 檔 - {monthly_passed}")
        # monthly_passed = ['1264', '1321', '1560', '1612', '1618', '2102', '2247', '2365', '2379', '2382', '2402', '2412', '2427', '2493', '2546', '2630', '2836', '3005', '3036', '3059', '3086', '3131', '3149', '3376', '3625', '4129', '4153', '5205', '5292', '5434', '5490', '5523', '6156', '6187', '6201', '6213', '6515', '6561', '8464', '8931', '9902']
        
        # 第二階段：突破拉回篩選
        logging.info("執行突破拉回篩選...")
        final_candidates = []
        for code in monthly_passed:
            if self.breakout_pullback_filter(code):
                final_candidates.append(code)
                logging.info(f"{code} 通過突破拉回篩選")
                
        return final_candidates


def main():
    """主程式"""
    scanner = TaiwanStockScanner()
    
    # 執行選股（可設定limit測試）
    candidates = scanner.scan_stocks()  # 測試時限制50檔
    print(f"item: {candidates}")
    if candidates:
        logging.info(f"🎯 最終選股結果 ({len(candidates)} 檔): {candidates}")
        
        # 顯示股票名稱
        for code in candidates:
            try:
                stock_name = twstock.codes[code].name if code in twstock.codes else "未知"
                logging.info(f"  {code} - {stock_name}")
            except:
                logging.info(f"  {code} - 無法取得名稱")
    else:
        logging.info("本次篩選無符合條件股票")


if __name__ == "__main__":
    main()