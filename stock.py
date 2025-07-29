import twstock
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Stock:
    def __init__(self):
        self.twse_list = set(twstock.twse.keys())
        self.tpex_list = set(twstock.tpex.keys())

    def get_stock_list(self) -> list:
        all_codes = self.twse_list | self.tpex_list
        filtered = [c for c in all_codes if len(c) == 4 and c[0] != '0']
        return sorted(filtered)

    def _get_ticker(self, symbol: str) -> str:
        if symbol in self.twse_list:
            return f"{symbol}.TW"
        elif symbol in self.tpex_list:
            return f"{symbol}.TWO"
        else:
            raise ValueError(f"{symbol} 不在上市或上櫃清單中。")

    def get_monthly_selection(self, symbol: str) -> pd.Series:
        """
        月線篩選：
        • 計算月末收盤、最高、最低
        • MA5/MA20多頭排列
        • 近100月收盤相對位置低於70%
        """
        ticker = self._get_ticker(symbol)
        df = yf.download(ticker, period='10y', interval='1d', auto_adjust=False, progress=False)
        if df.empty:
            return pd.Series()

        # 取每月數據
        monthly_close = df['Close'].resample('ME').last()
        monthly_high  = df['High'].resample('ME').max()
        monthly_low   = df['Low'].resample('ME').min()
        monthly = pd.concat([monthly_close, monthly_high, monthly_low],
                            axis=1, keys=['Close', 'High', 'Low']).dropna()

        # 計算MA
        monthly['MA_5']  = monthly['Close'].rolling(5).mean()
        monthly['MA_20'] = monthly['Close'].rolling(20).mean()
        monthly.dropna(inplace=True)
        if len(monthly) < 2:
            return pd.Series()

        prev, last = monthly.iloc[-2], monthly.iloc[-1]
        cond1 = (last['MA_5'].item() > prev['MA_5'].item())
        cond2 = (last['MA_20'].item() > prev['MA_20'].item())
        cond3 = (last['MA_5'].item() > last['MA_20'].item())
        print(f"last 5 {last['MA_5'].item()}, last 20 {last['MA_20'].item()}")
        ratio = (last['MA_5'] / last['MA_20']).item()
        cond4 = 1 < ratio <= 1.04
        # 近100月相對位置
        lookback_months = min(100, len(monthly))
        recent = monthly.iloc[-lookback_months:]
        max_price = recent['High'].max().item()
        min_price = recent['Low'].min().item()
        current_close = last['Close'].item()
        # 避免除零錯誤
        if max_price != min_price:
            relative_position = (current_close - min_price) / (max_price - min_price)
            cond5 = relative_position < 0.7
            relative_pct = relative_position * 100
        else:
            cond5 = False
            relative_pct = 0

        logging.info(f"{ticker} 月線篩選: MA5↑{cond1}, MA20↑{cond2}, MA5>MA20{cond3}, MA距離是否接近{cond4}, 相對位置{ratio:.2f}%，PL<70%:{cond5}")

        if cond1 and cond2 and cond3 and cond4 and cond5:
            return last[['Close', 'MA_5', 'MA_20']]
        return pd.Series()

    def get_weekly_selection(self,
                             symbol: str,
                             lookback_weeks: int = 20,
                             pct_threshold: float = 0.03
                            ) -> pd.Series:
        """
        週線篩選：
        • MA5/MA20/MA60計算
        • 5~20週先低點後高點，現收盤於區間內
        • MA5/MA20糾纏，MA60向上
        """
        try:
            ticker = self._get_ticker(symbol)
            df = yf.download(ticker, period='2y', interval='1d',
                             auto_adjust=False, progress=False)
            if df.empty:
                return pd.Series()

            weekly_close = df['Close'].resample('W-FRI').last()
            weekly_low   = df['Low'].resample('W-FRI').min()
            weekly_high  = df['High'].resample('W-FRI').max()
            weekly = pd.concat(
                [weekly_close, weekly_low, weekly_high],
                axis=1, keys=['Close','Low','High']
            ).dropna()

            weekly['MA_5']  = weekly['Close'].rolling(window=5).mean()
            weekly['MA_20'] = weekly['Close'].rolling(window=20).mean()
            weekly['MA_60'] = weekly['Close'].rolling(window=60).mean()
            weekly.dropna(inplace=True)

            if len(weekly) < 21:
                return pd.Series()

            prior    = weekly.iloc[-21:-1]
            curr_row = weekly.iloc[-1]

            arr_low = prior['Low'].values
            arr_high = prior['High'].values
            trough_pos = arr_low.argmin()
            peak_pos   = arr_high.argmax()
            min_c = float(prior['Low'].min())
            max_c = float(prior['High'].max())
            curr_c = float(curr_row['Close'])

            cond_a = trough_pos < peak_pos
            cond_b = (min_c <= curr_c <= max_c)

            ma5 = weekly['MA_5'].iat[-1]
            ma20 = weekly['MA_20'].iat[-1]
            ratio = abs(ma5 - ma20) / ma20
            ma5_ma20_entangle = ratio < pct_threshold

            prev_ma60 = weekly['MA_60'].iat[-2]
            curr_ma60 = weekly['MA_60'].iat[-1]
            ma60_up = curr_ma60 > prev_ma60

            passed = cond_a and cond_b and ma5_ma20_entangle and ma60_up

            # 統一乾淨log
            logging.info(f"{ticker} 週線: 低點前:{cond_a}, 收盤區:{cond_b}, MA5/MA20糾纏:{ma5_ma20_entangle}, MA60上:{ma60_up}, 通過:{passed}")

            if passed:
                return curr_row[['Close', 'MA_5', 'MA_20', 'MA_60']]
            return pd.Series()

        except Exception as e:
            logging.error(f"{symbol} 處理時發生錯誤: {e}")
            return pd.Series()

if __name__ == '__main__':
    stock = Stock()
    codes = stock.get_stock_list()
    logging.info(f"總共 {len(codes)} 檔股票將進行篩選")

    # 1. 月線初篩
    monthly_pass = []
    for c in codes:
        res = stock.get_monthly_selection(c)
        if not res.empty:
            monthly_pass.append(c)
    logging.info(f"月線通過({len(monthly_pass)})：{monthly_pass}")

    # 2. 週線複篩
    final_pass = []
    for c in monthly_pass:
        res = stock.get_weekly_selection(c)
        if not res.empty:
            final_pass.append(c)
            logging.info(f"{c} 通過週線篩選: {res.to_dict()}")
        else:
            logging.info(f"{c} 未通過週線篩選")

    logging.info(f"最終符合({len(final_pass)})：{final_pass}")