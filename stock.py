# -*- coding: utf-8 -*-
import logging
import time
import random
from typing import List, Optional, Dict
from functools import wraps
import concurrent.futures
from threading import Lock
import numpy as np
import pandas as pd
import twstock
import yfinance as yf
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class TaiwanStockHistoricalScanner:
    """台股技術面選股器（歷史回測）— 防止請求過多版"""

    def __init__(self, target_date: Optional[str] = None, years_back_default: int = 3):
        # 上市/上櫃清單與代碼集
        self.twse_list = set(twstock.twse.keys())
        self.tpex_list = set(twstock.tpex.keys())
        self.all_codes = sorted([c for c in (self.twse_list | self.tpex_list) 
                               if len(c) == 4 and not c.startswith("0")])

        # 目標日期
        self.target_date = pd.Timestamp(target_date).normalize() if target_date else pd.Timestamp.now().normalize()
        logging.info(f"目標日期: {self.target_date.date()}")

        # 快取與預設下載年限
        self.cache: Dict[str, pd.DataFrame] = {}
        self.years_back_default = years_back_default
        
        # 請求控制
        self.request_count = 0
        self.last_request_time = 0
        self.cache_lock = Lock()  # 執行緒安全的快取
        
        # 請求限制參數
        self.max_requests_per_minute = 60  # 每分鐘最大請求數
        self.min_request_interval = 1.0    # 最小請求間隔（秒）
        self.max_retries = 3              # 最大重試次數
        
        # 統一參數集中管理
        self.params = {
            # 月線
            "ma_ratio_low": 1.00,
            "ma_ratio_high": 1.04,
            "monthly_lowpos_thresh": 0.7,
            "monthly_min_points": 15,
            # 突破-拉回
            "breakout_volume_mult": 2.0,
            "breakout_window": 20,
            "pullback_min_days": 34,
            "ma_support": 20,
            "atr_limit": 0.04,
        }

    def _rate_limit_check(self):
        """檢查並執行請求頻率限制"""
        current_time = time.time()
        
        # 如果距離上次請求不足最小間隔，則等待
        if self.last_request_time > 0:
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                logging.debug(f"請求頻率限制：等待 {sleep_time:.2f} 秒")
                time.sleep(sleep_time)
        
        # 更新請求計數和時間
        self.request_count += 1
        self.last_request_time = time.time()
        
        # 每 60 個請求後稍作休息
        if self.request_count % self.max_requests_per_minute == 0:
            logging.info(f"已處理 {self.request_count} 個請求，休息 10 秒...")
            time.sleep(10)

    def _to_ticker(self, code: str) -> str:
        if code in self.twse_list:
            return f"{code}.TW"
        if code in self.tpex_list:
            return f"{code}.TWO"
        raise ValueError(f"未知代碼: {code}")

    def _slice_until(self, df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.tz_convert(None)
        return df.loc[df.index <= date]

    def _download_with_retry(self, code: str, years_back: Optional[int] = None) -> Optional[pd.DataFrame]:
        """帶重試機制的下載函式"""
        years_back = years_back or self.years_back_default
        key = f"{code}:{years_back}"
        
        # 執行緒安全的快取檢查
        with self.cache_lock:
            if key in self.cache:
                return self._slice_until(self.cache[key], self.target_date)

        for attempt in range(self.max_retries):
            try:
                # 執行請求頻率限制
                self._rate_limit_check()
                
                start = (self.target_date - pd.Timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
                end = (self.target_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                
                logging.debug(f"下載 {code} (第 {attempt + 1} 次嘗試)")
                
                df = yf.download(
                    self._to_ticker(code), 
                    start=start, 
                    end=end, 
                    interval="1d",
                    auto_adjust=False, 
                    progress=False, 
                    timeout=30
                )
                
                if df.empty:
                    logging.debug(f"{code} 無資料")
                    return None
                    
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # 執行緒安全的快取更新
                with self.cache_lock:
                    self.cache[key] = df.copy()
                
                return self._slice_until(df, self.target_date)
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)  # 指數退避 + 隨機
                    logging.warning(f"下載失敗 {code} (第 {attempt + 1} 次): {e}，{wait_time:.1f}秒後重試")
                    time.sleep(wait_time)
                else:
                    logging.error(f"下載失敗 {code} (已達最大重試次數): {e}")
                    return None

    def _download(self, code: str, years_back: Optional[int] = None) -> Optional[pd.DataFrame]:
        """原有的下載函式，現在調用帶重試的版本"""
        return self._download_with_retry(code, years_back)

    def _last_trading_on_or_before(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """回傳 df 中 <= date 的最後一個交易日索引"""
        if df is None or df.empty:
            return None
        ix = df.index[df.index <= date]
        return ix[-1] if len(ix) else None

    # ---------- 技術工具 ----------

    @staticmethod
    def _resample_monthly_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """以月末為基準取 Close/High/Low（用於月線條件），回傳含 Close/High/Low/MA5/MA20"""
        close = df["Close"].resample("ME").last()
        high = df["High"].resample("ME").max()
        low = df["Low"].resample("ME").min()
        m = pd.concat([close, high, low], axis=1)
        m.columns = ["Close", "High", "Low"]
        m.dropna(inplace=True)
        if len(m) >= 5:
            m["MA5"] = m["Close"].rolling(5).mean()
        if len(m) >= 20:
            m["MA20"] = m["Close"].rolling(20).mean()
        return m.dropna()

    @staticmethod
    def _price_position(close: float, low: float, high: float) -> float:
        rng = max(high - low, 1e-9)
        return (close - low) / rng

    # ---------- 條件 ----------

    def monthly_filter(self, code: str) -> bool:
        try:
            df = self._download(code, years_back=3)
            if df is None:
                return False

            m = self._resample_monthly_ohlc(df)
            if len(m) < self.params["monthly_min_points"]:
                return False

            # 對齊到目標月份
            target_month = self.target_date.to_period("M")
            m_idx = m.index.to_period("M")

            if target_month not in m_idx:
                pos = np.where(m_idx <= target_month)[0]
                if len(pos) == 0:
                    return False
                t_i = pos[-1]
            else:
                t_i = m_idx.get_loc(target_month)
            if t_i == 0:
                return False

            cur, prev = m.iloc[t_i], m.iloc[t_i - 1]
            ratio = float(cur["MA5"] / cur["MA20"])
            
            # 近 20 個月的區間位置
            left = max(0, t_i - 19)
            recent = m.iloc[left:t_i + 1]
            price_pos = self._price_position(
                cur["Close"], recent["Low"].min(), recent["High"].max()
            )

            ok = (
                (cur["MA5"] > prev["MA5"]) and
                (cur["MA20"] > prev["MA20"]) and
                (self.params["ma_ratio_low"] < ratio <= self.params["ma_ratio_high"]) and
                (price_pos < self.params["monthly_lowpos_thresh"])
            )

            if ok:
                logging.info(
                    f"{code} ✓ 月線: "
                    f"MA5={cur['MA5']:.2f} MA20={cur['MA20']:.2f} "
                    f"ratio={ratio:.4f} pos={price_pos:.2f}"
                )
            return ok

        except Exception as e:
            logging.warning(f"{code} 月線篩選失敗: {e}")
            return False

    def breakout_pullback_filter(self, code: str) -> bool:
        df = self._download(code, years_back=1)
        if df is None or len(df) < 60:
            return False

        # 確保使用目標日前最後一個交易日
        tdate = self._last_trading_on_or_before(df, self.target_date)
        if tdate is None:
            return False
        df = df.loc[:tdate]

        # 指標計算
        w = self.params["breakout_window"]
        df["MA20"] = df["Close"].rolling(20).mean()
        df["VolMA"] = df["Volume"].rolling(w).mean()
        df["HighMax"] = df["High"].shift(1).rolling(w).max()

        tr = np.maximum(df["High"] - df["Low"],
                        np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                                   (df["Low"] - df["Close"].shift(1)).abs()))
        df["ATR"] = tr.rolling(14).mean()

        df = df.dropna(subset=["High", "HighMax", "Volume", "VolMA", "MA20", "ATR"])
        if len(df) < 30:
            return False

        # 突破點：創新高 + 量能放大
        brk_mask = (df["High"] > df["HighMax"]) & (df["Volume"] > self.params["breakout_volume_mult"] * df["VolMA"])
        if not brk_mask.any():
            return False

        last_breakout = df.index[brk_mask][-1]
        days_since = (df.index.get_loc(tdate) - df.index.get_loc(last_breakout))

        cur = df.loc[tdate]
        pre_highmax = df.loc[last_breakout, "HighMax"]

        ok = (
            (days_since >= self.params["pullback_min_days"]) and
            (cur["Close"] < pre_highmax) and
            (cur["Low"] >= cur["MA20"]) and
            (float(cur["ATR"] / cur["Close"]) <= self.params["atr_limit"])
        )
        
        if ok:
            logging.info(f"{code} 拉回: 突破{days_since}天前({last_breakout.date()}) 守MA20, ATR%={cur['ATR']/cur['Close']:.3%}")
        return ok

    # ---------- 批次處理 ----------

    def batch_download(self, codes: List[str], batch_size: int = 50) -> Dict[str, pd.DataFrame]:
        """批次下載股票資料"""
        results = {}
        total_batches = (len(codes) + batch_size - 1) // batch_size
        
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logging.info(f"處理第 {batch_num}/{total_batches} 批次 ({len(batch)} 檔股票)")
            
            for code in batch:
                df = self._download(code)
                if df is not None:
                    results[code] = df
                    
            # 批次之間休息
            if batch_num < total_batches:
                logging.info(f"批次 {batch_num} 完成，休息 5 秒...")
                time.sleep(5)
                
        return results

    # ---------- 流程 ----------

    def get_stock_codes(self) -> List[str]:
        return self.all_codes

    def scan_stocks(self, limit: Optional[int] = None) -> List[str]:
        codes = self.all_codes[:limit] if limit else self.all_codes
        logging.info(f"{self.target_date.date()} start to scanning {len(codes)} files...")

        # 重置請求計數
        self.request_count = 0
        
        passed_monthly = []
        for i, code in enumerate(codes, 1):
            if i % 100 == 0:  # 每100檔顯示進度
                logging.info(f"進度: {i}/{len(codes)} ({i/len(codes)*100:.1f}%)")
                
            if self.monthly_filter(code):
                passed_monthly.append(code)
                
        logging.info(f"月線通過: {len(passed_monthly)} 檔")
        logging.info(f"月線通過個股: {passed_monthly}")

        final_list = []
        for code in passed_monthly:
            if self.breakout_pullback_filter(code):
                final_list.append(code)
                
        return final_list

    def backtest_multiple_dates(self, date_list: List[str], limit: Optional[int] = None) -> Dict[str, List[str]]:
        results = {}
        for d in date_list:
            logging.info("\n" + "=" * 48 + f"\n 回測: {d}\n" + "=" * 48)
            self.target_date = pd.to_datetime(d).normalize()
            results[d] = self.scan_stocks(limit)
            logging.info(f"{d} 結果: {len(results[d])} 檔")
            
            # 回測之間休息
            time.sleep(2)
            
        return results

def main():
    target_date: Optional[str] = None  # None = 今天
    scanner = TaiwanStockHistoricalScanner(target_date=target_date)

    # 可以先測試少量股票
    picks = scanner.scan_stocks(limit=100)  # 限制前100檔測試
    
    if picks:
        logging.info(f"最終({len(picks)}): {picks}")
        for c in picks:
            name = getattr(twstock.codes.get(c, None), "name", "未知")
            logging.info(f"  {c} - {name}")
    else:
        logging.info("無符合條件股票")

    print(f"\n選股結果: {picks}")

if __name__ == "__main__":
    main()
