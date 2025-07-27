import twstock
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Stock:
    def __init__(self):
        # 合併上市（TWSE）與上櫃（TPEx）代碼清單
        self.twse_list = set(twstock.twse.keys())
        self.tpex_list = set(twstock.tpex.keys())

    def get_stock_list(self) -> list:
        """
        取得所有 4 碼且首碼非 '0' 的上市/上櫃股票代碼，並排序
        """
        all_codes = self.twse_list | self.tpex_list
        filtered = [c for c in all_codes if len(c) == 4 and c[0] != '0']
        return sorted(filtered)

    def _get_ticker(self, symbol: str) -> str:
        """
        根據 twstock 分類，自動加上 yfinance 後綴
        上市 .TW / 上櫃 .TWO
        """
        if symbol in self.twse_list:
            return f"{symbol}.TW"
        elif symbol in self.tpex_list:
            return f"{symbol}.TWO"
        else:
            raise ValueError(f"{symbol} 不在上市或上櫃清單中。")

    def _is_breakout_and_pullback(self, series: pd.Series, lookback: int) -> bool:
        # 只取最後 lookback 筆
        recent = series.iloc[-lookback:]
        arr = recent.values         # numpy array of floats
        trough_pos = arr.argmin()   # 回傳最低點的相對位置 (0-based)
        # 在 trough 之後的子陣列中找最高點
        peak_rel = arr[trough_pos:].argmax()
        peak_pos = trough_pos + peak_rel
        last_pos = len(arr) - 1
        # 檢查  trough_pos < peak_pos < last_pos
        return (0 <= trough_pos < peak_pos < last_pos)

    def get_monthly_selection(self,
                              symbol: str
                              ) -> pd.Series:
        """
        月線篩選：
        • 計算月末收盤的 MA5／MA20
        • 檢查 MA5 與 MA20 的多頭趨勢（MA5↑、MA20↑ 且 MA5 > MA20）
        回傳最後一個月的 (Close, MA_5, MA_20)，或空 Series
        """
        ticker = self._get_ticker(symbol)
        df = yf.download(ticker, period='2y', interval='1d',
                         auto_adjust=False, progress=False)
        if df.empty:
            return pd.Series()

        # 月末收盤
        monthly = df[['Close']].resample('ME').last().dropna()
        monthly['MA_5']  = monthly['Close'].rolling(5).mean()
        monthly['MA_20'] = monthly['Close'].rolling(20).mean()
        monthly.dropna(inplace=True)
        if len(monthly) < 2:
            return pd.Series()

        prev, last = monthly.iloc[-2], monthly.iloc[-1]
        cond1 = last['MA_5']  > prev['MA_5']   # MA5 上升
        cond2 = last['MA_20'] > prev['MA_20']  # MA20 上升
        cond3 = last['MA_5']  > last['MA_20']  # MA5 穿越 MA20
        cond4 = (last['MA_5'] / last['MA_20']) <= 1.04

        logging.info(f"{ticker} 月線篩選: MA5↑ {cond1}, MA20↑ {cond2}, MA5>MA20 {cond3}")

        if (cond1 & cond2 & cond3 & cond4).all():
            return last[['Close','MA_5','MA_20']]
        return pd.Series()

    def get_weekly_selection(self,
                             symbol: str,
                             lookback_weeks: int = 20,
                             pct_threshold: float = 0.05
                            ) -> pd.Series:
        """
        週線篩選：
        • 計算每週（週五收盤）的 MA5／MA20／MA60
        • 檢查最近 lookback_weeks 週內收盤價位於最高最低區間內
        • 檢查「先最低→再最高→再當前」的突破拉回模式
        • 檢查均線糾纏（差距小於 pct_threshold）
        回傳最後一週的 (Close, MA_5, MA_20, MA_60)，或空 Series
        """
        try:
            ticker = self._get_ticker(symbol)
            df = yf.download(ticker, period='2y', interval='1d',
                             auto_adjust=False, progress=False)
            if df.empty:
                logging.warning(f"{ticker} 無法取得數據")
                return pd.Series()

            # 2) 計算週收盤 / 週低 / 週高
            weekly_close = df['Close'].resample('W-FRI').last()
            weekly_low   = df['Low'].resample('W-FRI').min()
            weekly_high  = df['High'].resample('W-FRI').max()

            # 3) 合併成一個 DataFrame
            weekly = pd.concat(
                [weekly_close, weekly_low, weekly_high],
                axis=1, keys=['Close','Low','High']
            ).dropna()

            # 4) 計算移動平均線
            weekly['MA_5']  = weekly['Close'].rolling(window=5).mean()
            weekly['MA_20'] = weekly['Close'].rolling(window=20).mean()
            weekly['MA_60'] = weekly['Close'].rolling(window=60).mean()
            weekly.dropna(inplace=True)

            # 檢查是否有足夠的數據
            if len(weekly) < 20:
                logging.warning(f"{ticker} 週線數據不足，只有 {len(weekly)} 週")
                return pd.Series()

            # 4) 取「前第 5~20 週」（不含最近 4 週：本週-前3週）
            if len(weekly) < 20:
                logging.warning(f"{ticker} 數據不足20週")
                return pd.Series()
            
            prior = weekly.iloc[-20:-4]

            # 5) 分別計算這段週期的最高盤中價、最低盤中價
            max_c = round(prior['High'].max().item(), 2)
            min_c = round(prior['Low'].min().item(), 2)

            # 6) 取得本週收盤價和移動平均線
            curr_row = weekly.iloc[-1]
            curr_c = round(curr_row['Close'].item(), 2)

            logging.info(f"{ticker} 5~20 週最高：{max_c}，最低：{min_c}，本週收盤：{curr_c}")
            
            # 5) 判斷本週收盤是否在區間內
            cond_a = (min_c <= curr_c) & (curr_c <= max_c)

            # 6) 突破拉回模式檢查
            arr_low  = prior['Low'].values
            arr_high = prior['High'].values
            
            if len(arr_low) < 2:
                logging.warning(f"{ticker} prior 數據不足")
                return pd.Series()
                
            trough = arr_low.argmin()
            peak = trough + arr_high[trough:].argmax()
            cond_b = (0 <= trough < peak < len(arr_low)-1)

            # 7) 均線糾纏檢查
            mas = curr_row[['MA_5','MA_20','MA_60']]
            cond_c = (mas.max() - mas.min()) / mas.mean() < pct_threshold

            logging.info(f"{symbol} 條件檢查: 區間內={cond_a}, 突破拉回={cond_b}, 均線糾纏={cond_c}")

            if (cond_a & cond_b & cond_c).all():
                return curr_row[['Close','MA_5','MA_20','MA_60']]
            return pd.Series()
            
        except Exception as e:
            logging.error(f"{symbol} 處理時發生錯誤: {e}")
            return pd.Series()
        
if __name__ == '__main__':
    stock = Stock()
    # codes = stock.get_stock_list()
    # logging.info(f"總共 {len(codes)} 檔股票將進行篩選")

    # # 1. 月線初篩
    # monthly_pass = []
    # for c in codes:
    #     res = stock.get_monthly_selection(c)
    #     if not res.empty:
    #         monthly_pass.append(c)
    # logging.info(f"月線通過({len(monthly_pass)})：{monthly_pass}")
    monthly_pass = ['1216', '1612', '2010', '2247', '2364', '2379', '2412', '2423', '2439', '2451', '2493', '2607', '2801', '2820', '2834', '2836', '2886', '2890', '2892', '3028', '3030', '3052', '3231', '3535', '4114', '4129', '4198', '4760', '5274', '5490', '5601', '6177', '6669', '8038', '8358', '8422', '8424', '9902', '9911']
    # 2. 週線複篩
    final_pass = []
    for c in monthly_pass:
        res = stock.get_weekly_selection(c)
        if not res.empty:
            final_pass.append(c)
    logging.info(f"最終符合({len(final_pass)})：{final_pass}")