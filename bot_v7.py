import os
import re
import json
import yaml
import ccxt
import time
import asyncio
import datetime
import pandas as pd
import pandas_ta as ta
import streamlit as st
from loguru import logger
from hashlib import sha256
from tardis_dev import datasets
from traceback import format_exc

USE_STREAMLIT = False

logger.add('logs/backtester.log', rotation='50mb', level='DEBUG')


def older_than(now, ts, minutes=0, seconds=0):
    return (now - ts).total_seconds() > seconds + minutes * 60


class Backtester:
    def __init__(self, exchange, base_symbols, quote_symbol, from_date, to_date, order_fee, max_number_of_orders_per_symbol,
                 buy_order_price_offset, sell_order_price_offset, buy_trade_size, sell_trade_size, max_number_of_open_trades_for_symbol, max_open_order_age,
                 initial_balance_base, initial_balance_quote, timeout=0, direction='both', timeframes=[],
                 enable_psar=True, psar_timeframe=None, max_number_of_trades_for_period=None, max_number_of_trades_period=None, trade_channels=[]):
        
        self.exchange = exchange
        self.base_symbols = base_symbols
        self.quote_symbol = quote_symbol
        self.from_date = from_date
        self.to_date = to_date
        self.trade_channels = trade_channels
        
        self.symbols = [s + self.quote_symbol for s in self.base_symbols]
        self.trades = []
        self.trade_id = 0
        self.closed_trades = []
        
        self.max_number_of_orders_per_symbol = max_number_of_orders_per_symbol
        self.order_fee = order_fee / 100
        
        self.timeout = timeout
        
        self.buy_order_offset = buy_order_price_offset / 100
        self.sell_order_offset = sell_order_price_offset / 100
        self.buy_order_size_quote = buy_trade_size
        self.sell_order_size_quote = sell_trade_size
        self.max_number_of_open_trades_for_symbol = max_number_of_open_trades_for_symbol
        self.max_open_order_age = max_open_order_age  # In minutes
        
        self.max_number_of_trades_for_period = max_number_of_trades_for_period
        self.max_number_of_trades_period = max_number_of_trades_period
        
        self.initial_balance_base = initial_balance_base
        self.balance_base = initial_balance_base
        self.initial_balance_quote = initial_balance_quote
        self.balance_quote = initial_balance_quote
        
        self.direction = direction  # long, short, both
        
        self.enable_psar = enable_psar
        self.psar_timeframe = psar_timeframe
        
        self.timeframes = timeframes
        self.candles = {}
        
        # Used for debugging
        if self.enable_psar:
            self.psar_signals_for_report = []
        
    def get_data(self):
        file_name = f'trades_{self.exchange}_{self.from_date}_{self.to_date}_{self.symbols}.csv.gz'
        file_path = f'./datasets/{file_name}'
        
        def default_file_name(exchange: str, data_type: str, date: datetime, symbol: str, format: str):
            return file_name
        
        if not os.path.isfile(file_path):
            datasets.download(
                exchange=self.exchange,
                data_types=["trades"],
                from_date=self.from_date,
                to_date=self.to_date,
                symbols=self.symbols,
                api_key="TD.KjejGCDzRoa6bi-z.9YpyVWGLT6CbNMq.OcQxyUz-BRsm1A6.R20Jgmp-RmiJbay.-J1TB2RqJnXnqLf.VyXS",
                get_filename=default_file_name
            )
        self.data = pd.read_csv(file_path, compression='gzip')
        self.data['local_timestamp'] = pd.to_datetime(self.data['local_timestamp'], unit='us')
        self.data.set_index('local_timestamp', drop=False, inplace=True)
        
        for timeframe in self.timeframes:
            timeframe_df = self.data[['price', 'amount']].resample(timeframe).agg({'price': 'ohlc', 'amount': 'sum'})
            self.candles[timeframe] = timeframe_df['price'].assign(volume=timeframe_df['amount'])
            
    def run(self):
        self.get_data()
        starting_timestamp = time.time()
        timeout = self.timeout * 60  # in minutes
        for i, r in self.data.iterrows():
            if timeout and (time.time() - starting_timestamp) > timeout:
                break
            
            candles = {}
            for timeframe in self.timeframes:
                candles[timeframe] = self.candles[timeframe][self.candles[timeframe].index <= i]
            self.process_message(dict(r), candles=candles)

        self.process_trades(r['local_timestamp'], r['price'], close_all=True)
        # for trade in self.trades:
        #     if self.direction == 'both' and trade['buy_order_status'] == 'open' and trade['sell_order_status'] == 'open':
        #         self.balance_quote += trade['buy_size_quote']
        #         self.balance_base += trade['sell_size_base']
        #     elif self.direction == 'long' and trade['buy_order_status'] == 'open':
        #         self.balance_quote += trade['buy_size_quote']
        #     elif self.direction == 'short' and trade['sell_order_status'] == 'open':
        #         self.balance_base += trade['sell_size_base']
        
        self.trades_df = pd.DataFrame(self.closed_trades)

    def get_psar_indicator(self, candles):
        psar = ta.psar(
            high=candles[self.psar_timeframe]['high'],
            low=candles[self.psar_timeframe]['low'],
            close=candles[self.psar_timeframe]['close']
        )
        if not pd.isna(psar[psar.columns[0]].iloc[-1]):
            return 'long'
        if not pd.isna(psar[psar.columns[1]].iloc[-1]):
            return 'short'
        
    def process_message(self, message, candles):
        timestamp = message['local_timestamp']
        price = message['price']
        self.last_price = price
        symbol = message['symbol'].lower()

        if self.enable_psar:
            psar_signal = self.get_psar_indicator(candles)
            psar_buy_signal = psar_signal == 'long'
            psar_sell_signal = psar_signal == 'short'
            if psar_signal:
                self.psar_signals_for_report.append({'timestamp': timestamp, 'symbol': symbol, 'price': price, 'psar_signal': psar_signal, 'psar_timeframe': self.psar_timeframe})
        else:
            psar_buy_signal = True
            psar_sell_signal = True

        buy_order_size_base = self.buy_order_size_quote / price
        buy_order_size_quote = self.buy_order_size_quote

        sell_order_size_quote = buy_order_size_quote
        sell_order_size_base = buy_order_size_base

        number_of_trades_below_maximum = len(self.trades) < self.max_number_of_open_trades_for_symbol
        buy_order_price = price * (1 - self.buy_order_offset)
        sell_order_price = price * (1 + self.sell_order_offset)
        
        # Max number of trades per period check
        if self.max_number_of_trades_for_period:
            period_starting_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(self.max_number_of_trades_period)
            number_of_trades_during_period = len([t for t in self.trades if t['timestamp_created'] >= period_starting_timestamp if t['status'] != 'canceled'])
            number_of_trades_for_period_below_maximum = number_of_trades_during_period < self.max_number_of_trades_for_period
        
        # Max number of trades per channel
        number_of_trades_per_channel_below_maximum = True
        if self.trade_channels:
            for channel in self.trade_channels:
                channel['number_of_trades'] = 0
            for trade in [t for t in self.trades if t['status'] == 'open']:
                if not number_of_trades_per_channel_below_maximum:
                    break
                for channel in self.trade_channels:
                    if self.direction == 'both':
                        if (trade['buy_order_price'] >= channel['from'] and trade['buy_order_price'] < channel['to']) or (trade['sell_order_price'] >= channel['from'] and trade['sell_order_price'] < channel['to']):
                            channel['number_of_trades'] += 1
                    elif self.direction == 'long':
                        if trade['buy_order_price'] >= channel['from'] and trade['buy_order_price'] < channel['to']:
                            channel['number_of_trades'] += 1
                    elif self.direction == 'short':
                        if trade['sell_order_price'] >= channel['from'] and trade['sell_order_price'] < channel['to']:
                            channel['number_of_trades'] += 1
                    if channel['number_of_trades'] >= channel['max']:
                        logger.warning(f'Max number of trades per channel reached for channel {channel}')
                        number_of_trades_per_channel_below_maximum = False
                        break
                    
        place_trade = False
        if number_of_trades_per_channel_below_maximum and number_of_trades_below_maximum and number_of_trades_for_period_below_maximum:
            if self.direction == 'both':
                buy_order_size_base = self.buy_order_size_quote / price
                buy_order_size_quote = self.buy_order_size_quote
                sell_order_size_quote = buy_order_size_quote
                sell_order_size_base = buy_order_size_base
                sufficient_balance_for_new_sell_orders = self.balance_base >= sell_order_size_base
                sufficient_balance_for_new_buy_orders = self.balance_quote >= self.buy_order_size_quote
                if sufficient_balance_for_new_buy_orders and sufficient_balance_for_new_sell_orders:
                    self.balance_base -= sell_order_size_base
                    self.balance_quote -= buy_order_size_quote
                    place_trade = True
            elif self.direction == 'long':
                buy_order_size_base = self.buy_order_size_quote / price
                buy_order_size_quote = self.buy_order_size_quote
                sell_order_size_quote = buy_order_size_quote
                sell_order_size_base = buy_order_size_base
                sufficient_balance_for_new_buy_orders = self.balance_quote >= self.buy_order_size_quote
                if sufficient_balance_for_new_buy_orders and psar_buy_signal:
                    self.balance_quote -= buy_order_size_quote
                    place_trade = True
            elif self.direction == 'short':
                sell_order_size_quote = self.sell_order_size_quote
                sell_order_size_base = sell_order_size_quote / price
                buy_order_size_base = sell_order_size_base
                buy_order_size_quote = sell_order_size_quote
                sufficient_balance_for_new_sell_orders = self.balance_base >= sell_order_size_base
                if sufficient_balance_for_new_sell_orders and psar_sell_signal:
                    self.balance_base -= sell_order_size_base
                    place_trade = True
            
        if place_trade:
            self.trades.append(dict(
                id=self.trade_id,
                symbol=symbol,
                status='open',
                timestamp_created=timestamp,
                buy_order_price=buy_order_price,
                buy_order_status='open',
                sell_order_price=sell_order_price,
                sell_order_status='open',
                buy_size_quote=buy_order_size_quote,
                buy_size_base=buy_order_size_base,
                sell_size_quote=sell_order_size_quote,
                sell_size_base=sell_order_size_base,
                psar_timeframe=self.psar_timeframe,
                max_number_of_trades_for_period=self.max_number_of_trades_for_period,
                max_number_of_trades_period=self.max_number_of_trades_period
            ))
            self.trade_id += 1
        self.process_trades(timestamp, price)
        # for trade in self.trades:
        #     buy_order_filled = False
        #     sell_order_filled = False
            
        #     if self.direction == 'both':
        #         if trade['buy_order_status'] == 'open' and trade['sell_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
        #             self.balance_quote += trade['buy_size_quote']
        #             self.balance_base += trade['sell_size_base']
        #             trade['status'] = 'canceled'
        #             continue
        #         else:
        #             buy_order_filled = trade['buy_order_status'] == 'open' and price <= trade['buy_order_price']
        #             sell_order_filled = trade['sell_order_status'] == 'open' and price >= trade['sell_order_price']
        #     elif self.direction == 'long':
        #         if trade['buy_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
        #             self.balance_quote += trade['buy_size_quote']
        #             trade['status'] = 'canceled'
        #             continue
        #         else:
        #             buy_order_filled = trade['buy_order_status'] == 'open' and price <= trade['buy_order_price']
        #             sell_order_filled = trade['buy_order_status'] == 'filled' and trade['sell_order_status'] == 'open' and price >= trade['sell_order_price']
        #             if buy_order_filled:
        #                 self.balance_base -= trade['sell_size_base']
        #     elif self.direction == 'short':
        #         if trade['sell_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
        #             self.balance_base += trade['sell_size_base']
        #             trade['status'] = 'canceled'
        #             continue
        #         else:
        #             sell_order_filled = trade['sell_order_status'] == 'open' and price >= trade['sell_order_price']
        #             buy_order_filled = trade['sell_order_status'] == 'filled' and trade['buy_order_status'] == 'open' and price <= trade['buy_order_price']
        #             if sell_order_filled:
        #                 self.balance_quote -= trade['buy_size_quote']
                        
        #     if buy_order_filled:
        #         trade['buy_size_base'] = trade['buy_size_quote'] / trade['buy_order_price']
        #         trade['buy_order_status'] = 'filled'
        #         trade['buy_order_fill_timestamp'] = timestamp
        #         trade['buy_order_fee'] = trade['buy_size_quote'] * self.order_fee
        #         self.balance_base += trade['buy_size_base']
        #     elif sell_order_filled:
        #         trade['sell_size_quote'] = trade['sell_size_base'] * trade['sell_order_price']
        #         trade['sell_order_status'] = 'filled'
        #         trade['sell_order_fill_timestamp'] = timestamp
        #         trade['sell_order_fee'] = trade['sell_size_quote'] * self.order_fee
        #         self.balance_quote += trade['sell_size_quote']
             
        #     if trade['buy_order_status'] == 'filled' and trade['sell_order_status'] == 'filled':
        #         if trade['buy_order_fill_timestamp'] < trade['sell_order_fill_timestamp']:
        #             trade['side'] = 'long'
        #             size_closed = trade['sell_size_base'] * price
        #             trade['gross_pnl'] = size_closed - trade['buy_size_quote']
        #         else:
        #             trade['side'] = 'short'
        #             size_closed = trade['buy_size_base'] * price
        #             trade['gross_pnl'] = trade['sell_size_quote'] - size_closed
                    
        #         trade['net_pnl'] = trade['gross_pnl'] - (trade['buy_order_fee'] + trade['sell_order_fee'])
        #         trade['status'] = 'closed'
        #         self.closed_trades.append(trade)

        # self.trades = [t for t in self.trades if t['status'] == 'open']

    def process_trades(self, timestamp, price, close_all=False):
        for trade in self.trades:
            buy_order_filled = False
            sell_order_filled = False
            
            if self.direction == 'both':
                if trade['buy_order_status'] == 'open' and trade['sell_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
                    self.balance_quote += trade['buy_size_quote']
                    self.balance_base += trade['sell_size_base']
                    trade['status'] = 'canceled'
                    continue
                else:
                    buy_order_filled = trade['buy_order_status'] == 'open' and (price <= trade['buy_order_price'] or close_all)
                    sell_order_filled = trade['sell_order_status'] == 'open' and (price >= trade['sell_order_price'] or close_all)
            elif self.direction == 'long':
                if trade['buy_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
                    self.balance_quote += trade['buy_size_quote']
                    trade['status'] = 'canceled'
                    continue
                else:
                    buy_order_filled = trade['buy_order_status'] == 'open' and (price <= trade['buy_order_price'] or close_all)
                    sell_order_filled = trade['buy_order_status'] == 'filled' and trade['sell_order_status'] == 'open' and (price >= trade['sell_order_price'] or close_all)
                    if buy_order_filled:
                        self.balance_base -= trade['sell_size_base']
            elif self.direction == 'short':
                if trade['sell_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
                    self.balance_base += trade['sell_size_base']
                    trade['status'] = 'canceled'
                    continue
                else:
                    sell_order_filled = trade['sell_order_status'] == 'open' and (price >= trade['sell_order_price'] or close_all)
                    buy_order_filled = trade['sell_order_status'] == 'filled' and trade['buy_order_status'] == 'open' and (price <= trade['buy_order_price'] or close_all)
                    if sell_order_filled:
                        self.balance_quote -= trade['buy_size_quote']
                        
            if buy_order_filled:
                trade['buy_size_base'] = trade['buy_size_quote'] / trade['buy_order_price']
                trade['buy_order_status'] = 'filled'
                trade['buy_order_fill_timestamp'] = timestamp
                trade['buy_order_fee'] = trade['buy_size_quote'] * self.order_fee
                self.balance_base += trade['buy_size_base']
            elif sell_order_filled:
                trade['sell_size_quote'] = trade['sell_size_base'] * trade['sell_order_price']
                trade['sell_order_status'] = 'filled'
                trade['sell_order_fill_timestamp'] = timestamp
                trade['sell_order_fee'] = trade['sell_size_quote'] * self.order_fee
                self.balance_quote += trade['sell_size_quote']
             
            if trade['buy_order_status'] == 'filled' and trade['sell_order_status'] == 'filled':
                if trade['buy_order_fill_timestamp'] < trade['sell_order_fill_timestamp']:
                    trade['side'] = 'long'
                    size_closed = trade['sell_size_base'] * trade['sell_order_price']
                    trade['gross_pnl'] = size_closed - trade['buy_size_quote']
                else:
                    trade['side'] = 'short'
                    size_closed = trade['buy_size_base'] * trade['buy_order_price']
                    trade['gross_pnl'] = trade['sell_size_quote'] - size_closed
                    
                trade['net_pnl'] = trade['gross_pnl'] - (trade['buy_order_fee'] + trade['sell_order_fee'])
                trade['status'] = 'closed'
                self.closed_trades.append(trade)

        self.trades = [t for t in self.trades if t['status'] == 'open']


class Trader:
    def __init__(self, exchange, api_key, api_secret, api_password, base_symbols, quote_symbol, order_fee, max_number_of_orders_per_symbol, 
                 buy_order_price_offset, sell_order_price_offset, buy_trade_size, sell_trade_size, max_number_of_open_trades_for_symbol, 
                 max_open_order_age, initial_balance_base, initial_balance_quote, timeout=0, direction='both', timeframes=[],
                 enable_psar=True, psar_timeframe=None, max_number_of_trades_for_period=None, max_number_of_trades_period=None, trade_channels=[]):
        
        assert isinstance(base_symbols, list)
        
        self.exchange = exchange
        self.base_symbols = base_symbols
        self.quote_symbol = quote_symbol
        self.from_date = from_date
        self.to_date = to_date
        self.direction = direction
        self.trade_channels = trade_channels
        
        # TODO: Switch to spot
        # self.client = ccxt.binanceusdm({'apiKey': api_key, 'secret': api_secret})
        # self.client = ccxt.binance({'apiKey': api_key, 'secret': api_secret})
        
        # TODO: Switch to api secrets
        self.client = getattr(ccxt, exchange)({'apiKey': api_key, 'secret': api_secret, 'password': api_password})
        
        self.symbols = [s + self.quote_symbol for s in self.base_symbols]
        self.standard_symbols = {s: self.get_standard_symbol(s) for s in self.symbols}
        self.trades = {}
        self.closed_trades = []
        self.trades = []
        self.trade_id = 0
        
        self.max_number_of_orders_per_symbol = max_number_of_orders_per_symbol
        self.order_fee = order_fee / 100
        
        self.timeout = timeout
        
        self.buy_order_offset = buy_order_price_offset / 100
        self.sell_order_offset = sell_order_price_offset / 100
        self.buy_order_size_quote = buy_trade_size
        self.sell_order_size_quote = sell_trade_size
        self.max_number_of_open_trades_for_symbol = max_number_of_open_trades_for_symbol
        self.max_open_order_age = max_open_order_age  # In minutes
        
        self.max_number_of_trades_for_period = max_number_of_trades_for_period
        self.max_number_of_trades_period = max_number_of_trades_period
        
        self.initial_balance_base = initial_balance_base
        self.balance_base = initial_balance_base
        self.initial_balance_quote = initial_balance_quote
        self.balance_quote = initial_balance_quote
        
        self.timeframes = timeframes
        self.candles = {}
        
        self.enable_psar = enable_psar
        self.psar_timeframe = psar_timeframe
        
        # Used for debugging
        if self.enable_psar:
            self.psar_signals_for_report = []

    def get_standard_symbol(self, symbol):
        symbol = symbol.upper()
        markets = self.client.fetch_markets()
        for m in markets:
            if m['id'] == symbol:
                return m['symbol']
    
    def get_price(self, symbol):
        ticker = self.client.fetch_ticker(symbol)
        return {
            'local_timestamp': datetime.datetime.utcnow(),
            'price': ticker['last'],
            'symbol': symbol.replace('/', '').lower()
        }
    
    def get_candles(self, symbol, timeframe):
        timeframe = timeframe.rstrip('in')  # Convert min to m
        candles = self.client.fetch_ohlcv(self.get_standard_symbol(symbol), timeframe=timeframe)
        return pd.DataFrame(candles, columns=('timestamp', 'open', 'high', 'low', 'close', 'volume'))
    
    def run(self):
        starting_timestamp = time.time()
        timeout = self.timeout * 60  # in minutes
        while True:
            try:
                if timeout and (time.time() - starting_timestamp) > timeout:
                    break
                for symbol in self.symbols:
                    data = self.get_price(self.standard_symbols[symbol])
                    candles = {}
                    for timeframe in self.timeframes:
                        candles[timeframe] = self.get_candles(symbol=symbol, timeframe=timeframe)
                    self.process_message(data, candles)
                    self.trades_df = pd.DataFrame(self.closed_trades)
                    self.trades_df.to_csv('trades.csv', index=False)
            except KeyboardInterrupt:
                self.client.cancel_all_orders(self.standard_symbols[symbol])
            except:
                logger.error(format_exc())
                time.sleep(5)
            finally:
                balance = self.client.fetch_balance()
                base_symbol, quote_symbol = self.standard_symbols[symbol].split('/')
                self.balance_base = balance[base_symbol]['total']
                self.balance_quote = balance[quote_symbol]['total']

    def get_psar_indicator(self, candles):
        psar = ta.psar(
            high=candles[self.psar_timeframe]['high'],
            low=candles[self.psar_timeframe]['low'],
            close=candles[self.psar_timeframe]['close']
        )
        if not pd.isna(psar[psar.columns[0]].iloc[-1]):
            return 'long'
        if not pd.isna(psar[psar.columns[1]].iloc[-1]):
            return 'short'
    
    def process_message(self, message, candles):
        timestamp = message['local_timestamp']
        price = message['price']
        self.last_price = price
        symbol = message['symbol'].lower()
        
        if self.enable_psar:
            psar_signal = self.get_psar_indicator(candles)
            psar_buy_signal = psar_signal == 'long'
            psar_sell_signal = psar_signal == 'short'
            if psar_signal:
                self.psar_signals_for_report.append({'timestamp': timestamp, 'symbol': symbol, 'price': price, 'psar_signal': psar_signal, 'psar_timeframe': self.psar_timeframe})
        else:
            psar_buy_signal = True
            psar_sell_signal = True
            
        number_of_trades_below_maximum = len(self.trades) < self.max_number_of_open_trades_for_symbol
        buy_order_price = price * (1 - self.buy_order_offset)
        buy_order_price = float(self.client.price_to_precision(symbol=self.standard_symbols[symbol], price=buy_order_price))
        sell_order_price = price * (1 + self.sell_order_offset)
        sell_order_price = float(self.client.price_to_precision(symbol=self.standard_symbols[symbol], price=sell_order_price))
        
        balance = self.client.fetch_balance()
        base_symbol, quote_symbol = self.standard_symbols[symbol].split('/')
        free_balance_base = balance[base_symbol]['free']
        free_balance_quote = balance[quote_symbol]['free']
        
        self.balance_base = free_balance_base
        self.balance_quote = free_balance_quote
        
        place_trade = False
        buy_order_id = None
        sell_order_id = None
        
        # Max number of trades per period check
        if self.max_number_of_trades_for_period:
            period_starting_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(self.max_number_of_trades_period)
            number_of_trades_during_period = len([t for t in self.trades if t['timestamp_created'] >= period_starting_timestamp if t['status'] != 'canceled'])
            number_of_trades_for_period_below_maximum = number_of_trades_during_period < self.max_number_of_trades_for_period
        
        # Max number of trades per channel
        number_of_trades_per_channel_below_maximum = True
        if self.trade_channels:
            for channel in self.trade_channels:
                channel['number_of_trades'] = 0
            for trade in [t for t in self.trades if t['status'] == 'open']:
                if not number_of_trades_per_channel_below_maximum:
                    break
                for channel in self.trade_channels:
                    if self.direction == 'both':
                        if (trade['buy_order_price'] >= channel['from'] and trade['buy_order_price'] < channel['to']) or (trade['sell_order_price'] >= channel['from'] and trade['sell_order_price'] < channel['to']):
                            channel['number_of_trades'] += 1
                    elif self.direction == 'long':
                        if trade['buy_order_price'] >= channel['from'] and trade['buy_order_price'] < channel['to']:
                            channel['number_of_trades'] += 1
                    elif self.direction == 'short':
                        if trade['sell_order_price'] >= channel['from'] and trade['sell_order_price'] < channel['to']:
                            channel['number_of_trades'] += 1
                    if channel['number_of_trades'] >= channel['max']:
                        logger.warning(f'Max number of trades per channel reached for channel {channel}')
                        number_of_trades_per_channel_below_maximum = False
                        break
            
        if number_of_trades_per_channel_below_maximum and number_of_trades_below_maximum and number_of_trades_for_period_below_maximum:
            if self.direction == 'both':
                buy_order_size_base = self.buy_order_size_quote / price
                buy_order_size_base = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=buy_order_size_base))
                buy_order_size_quote = self.buy_order_size_quote
                sell_order_size_quote = buy_order_size_quote
                sell_order_size_base = buy_order_size_base
                sell_order_size_base = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=sell_order_size_base))
                sufficient_balance_for_new_sell_orders = self.balance_base >= sell_order_size_base
                sufficient_balance_for_new_buy_orders = self.balance_quote >= self.buy_order_size_quote
                if sufficient_balance_for_new_buy_orders and sufficient_balance_for_new_sell_orders:
                    buy_order = self.client.create_order(symbol=self.standard_symbols[symbol], side='buy', amount=buy_order_size_base, type='limit', price=buy_order_price)
                    sell_order = self.client.create_order(symbol=self.standard_symbols[symbol], side='sell', amount=sell_order_size_base, type='limit', price=sell_order_price)
                    buy_order_id = buy_order['id']
                    sell_order_id = sell_order['id']
                    place_trade = True
            elif self.direction == 'long':
                buy_order_size_base = self.buy_order_size_quote / price
                buy_order_size_base = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=buy_order_size_base))
                buy_order_size_quote = self.buy_order_size_quote
                sell_order_size_quote = buy_order_size_quote
                sell_order_size_base = buy_order_size_base
                sell_order_size_base = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=sell_order_size_base))
                sufficient_balance_for_new_buy_orders = self.balance_quote >= self.buy_order_size_quote
                if sufficient_balance_for_new_buy_orders and psar_buy_signal:
                    buy_order = self.client.create_order(symbol=self.standard_symbols[symbol], side='buy', amount=buy_order_size_base, type='limit', price=buy_order_price)
                    buy_order_id = buy_order['id']
                    place_trade = True
            elif self.direction == 'short':
                sell_order_size_quote = self.sell_order_size_quote
                sell_order_size_base = sell_order_size_quote / price
                sell_order_size_base = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=sell_order_size_base))
                buy_order_size_base = sell_order_size_base
                buy_order_size_base = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=buy_order_size_base))

                buy_order_size_quote = sell_order_size_quote
                sufficient_balance_for_new_sell_orders = self.balance_base >= sell_order_size_base
                if sufficient_balance_for_new_sell_orders and psar_sell_signal:
                    sell_order = self.client.create_order(symbol=self.standard_symbols[symbol], side='sell', amount=sell_order_size_base, type='limit', price=sell_order_price)
                    sell_order_id = sell_order['id']
                    place_trade = True
            
        if place_trade:
            self.trades.append(dict(
                id=self.trade_id,
                symbol=symbol,
                status='open',
                timestamp_created=timestamp,
                buy_order_id=buy_order_id,
                buy_order_price=buy_order_price,
                buy_order_status='open',
                sell_order_id=sell_order_id,
                sell_order_price=sell_order_price,
                sell_order_status='open',
                buy_size_quote=buy_order_size_quote,
                buy_size_base=buy_order_size_base,
                sell_size_quote=sell_order_size_quote,
                sell_size_base=sell_order_size_base,
                psar_timeframe=self.psar_timeframe,
                max_number_of_trades_for_period=self.max_number_of_trades_for_period,
                max_number_of_trades_period=self.max_number_of_trades_period
            ))
            self.trade_id += 1

        for trade in self.trades:
            logger.info(f'Checking trade: {trade}')
            buy_order_filled = False
            sell_order_filled = False
            if trade['buy_order_status'] != 'filled' and trade['buy_order_id']:
                buy_order = self.client.fetch_order(trade['buy_order_id'], self.standard_symbols[symbol])
                if buy_order['status'] == 'closed':
                    logger.info('Buy order filled')
                    trade['buy_order_status'] = 'filled'
                    buy_order_filled = True
            if trade['sell_order_status'] != 'filled' and trade['sell_order_id']:
                sell_order = self.client.fetch_order(trade['sell_order_id'], self.standard_symbols[symbol])
                if sell_order['status'] == 'closed':
                    logger.info('Sell order filled')
                    trade['sell_order_status'] = 'filled'
                    sell_order_filled = True
            
            if self.direction == 'both':
                if trade['buy_order_status'] == 'open' and trade['sell_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
                    try:
                        logger.info('Cancelling buy and sell orders')
                        self.client.cancel_order(id=trade['buy_order_id'], symbol=self.standard_symbols[symbol])
                        self.client.cancel_order(id=trade['sell_order_id'], symbol=self.standard_symbols[symbol])
                        trade['status'] = 'canceled'
                    except:
                        logger.error(format_exc())
                    continue
            elif self.direction == 'long':
                if trade['buy_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
                    try:
                        logger.info('Cancelling buy order')
                        self.client.cancel_order(id=trade['buy_order_id'], symbol=self.standard_symbols[symbol])
                        trade['status'] = 'canceled'
                    except:
                        logger.error(format_exc())
                    continue
                elif trade['buy_order_status'] == 'filled' and not trade['sell_order_id']:
                    logger.info('Buy order filled, creating sell order')
                    order_amount = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=trade['sell_size_base']))
                    sell_order = self.client.create_order(symbol=self.standard_symbols[symbol], side='sell', amount=order_amount, type='limit', price=trade['sell_order_price'])
                    trade['sell_order_id'] = sell_order['id']
            elif self.direction == 'short':
                if trade['sell_order_status'] == 'open' and older_than(timestamp, trade['timestamp_created'], self.max_open_order_age):
                    try:
                        logger.info('Cancelling sell order')
                        self.client.cancel_order(id=trade['sell_order_id'], symbol=self.standard_symbols[symbol])
                        trade['status'] = 'canceled'
                    except:
                        logger.error(format_exc())
                    continue
                elif trade['sell_order_status'] == 'filled' and not trade['buy_order_id']:
                    logger.info('Sell order filled, creating buy order')
                    order_amount = float(self.client.amount_to_precision(symbol=self.standard_symbols[symbol], amount=trade['buy_size_base']))
                    buy_order = self.client.create_order(symbol=self.standard_symbols[symbol], side='buy', amount=order_amount, type='limit', price=trade['buy_order_price'])
                    trade['buy_order_id'] = buy_order['id']
                        
            if buy_order_filled:
                logger.info('Buy order filled')
                trade['buy_size_base'] = trade['buy_size_quote'] / trade['buy_order_price']
                trade['buy_order_status'] = 'filled'
                trade['buy_order_fill_timestamp'] = timestamp
                trade['buy_order_fee'] = trade['buy_size_quote'] * self.order_fee
            elif sell_order_filled:
                logger.info('Sell order filled')
                trade['sell_size_quote'] = trade['sell_size_base'] * trade['sell_order_price']
                trade['sell_order_status'] = 'filled'
                trade['sell_order_fill_timestamp'] = timestamp
                trade['sell_order_fee'] = trade['sell_size_quote'] * self.order_fee
             
            if trade['buy_order_status'] == 'filled' and trade['sell_order_status'] == 'filled':
                if trade['buy_order_fill_timestamp'] < trade['sell_order_fill_timestamp']:
                    trade['side'] = 'long'
                    size_closed = trade['sell_size_base'] * price
                    trade['gross_pnl'] = size_closed - trade['buy_size_quote']
                else:
                    trade['side'] = 'short'
                    size_closed = trade['buy_size_base'] * price
                    trade['gross_pnl'] = trade['sell_size_quote'] - size_closed
                    
                trade['net_pnl'] = trade['gross_pnl'] - (trade['buy_order_fee'] + trade['sell_order_fee'])
                trade['status'] = 'closed'
                logger.info(f'Trade closed: {trade}')
                self.closed_trades.append(trade)

        self.trades = [t for t in self.trades if t['status'] == 'open']
    

def parameter_list(p, return_first=False):
    if isinstance(p, list):
        if return_first:
            return [p[0]]
        else:
            return p
    else:
        return [p]


if __name__ == '__main__':

    if USE_STREAMLIT:
        header = st.text('BACKTESTING MODE')
        live_mode = st.checkbox('Live mode', value=False)
        exchange = st.selectbox('Exchange', ('binance',))
        if live_mode:
            api_key = st.text_input('API key')
            api_secret = st.text_input('API secret')
            api_password = st.text_input('API password')
            header.text('LIVE MODE')
        else:
            api_key = api_secret = None
        base_symbol = st.text_input('Base symbol, e.g. btc,eth', 'btc').lower()
        quote_symbol = st.text_input('Quote symbol, e.g. usdt', 'usdt').lower()
        ask_trade_size = st.number_input('Ask trade size', value=10)
        bid_trade_size = st.number_input('Bid trade size', value=10)
        max_number_of_orders_per_symbol = st.number_input('Maximum allowed orders per symbol', value=5)
        order_fee = st.number_input('Order fee', value=0.04)
        from_date = st.date_input('From Date', datetime.date(2022, 4, 1))
        to_date = st.date_input('To Date', datetime.date(2022, 4, 1)) + datetime.timedelta(days=1)
        timeout = st.number_input('Timeout in minutes (set to 0 to disable timeout)', value=0)
        buy_order_price_offset = st.number_input('Buy order price offset percent', value=0.5)
        sell_order_price_offset = st.number_input('Sell order price offset percent', value=0.5)
        max_number_of_open_trades_for_symbol = st.number_input('Maximum number of open trades per symbol', value=10)
        max_open_order_age = st.number_input('Maximum age of open orders (in minutes)', value=5)
        initial_balance_base = st.number_input('Initial balance in base currency', value=100.0)
        initial_balance_quote = st.number_input('Initial balance in quote currency', value=1.0)
        direction = st.selectbox('Direction', ('long', 'short', 'both'))
    
    else:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        exchange = config['exchange'] # binance
        base_symbol = config['base_symbol']# storj
        quote_symbol = config['quote_symbol']# usdt
        ask_trade_size = config['ask_trade_size'] # 10 In quote currency.
        bid_trade_size = config['bid_trade_size'] # 10 
        max_number_of_orders_per_symbol = config['max_number_of_orders_per_symbol'] #10
        order_fee = config['order_fee']#0.1
        buy_order_price_offset = config['buy_order_price_offset']#0.1  # in percents
        sell_order_price_offset = config['sell_order_price_offset']#0.1  # in percents
        max_number_of_open_trades_for_symbol = config['max_number_of_open_trades_for_symbol']#10
        max_open_order_age = config['max_open_order_age']  # in minutes
        from_date = config['from_date'] #datetime.date(2022, 4, 2)
        to_date = config['to_date'] #datetime.date(2022, 4, 2) + datetime.timedelta(days=1)
        timeout = config['timeout'] #2
        live_mode = config['live_mode'] #True
        api_key = config['api_key']
        api_secret = config['api_secret']
        api_password = config['api_password']
        initial_balance_base = config['initial_balance_base']
        initial_balance_quote = config['initial_balance_quote']
        direction = config['direction']
        enable_psar = config['enable_psar']
        psar_timeframe = config['psar_timeframe']
        max_number_of_trades_for_period = config['max_number_of_trades_for_period']
        max_number_of_trades_period = config['max_number_of_trades_period']
        trade_channels = config['trade_channels']
    
    timeframes = []
    if config['enable_psar']:
        timeframes.append(config['psar_timeframe'])
    
    if live_mode:
        mode_name = 'Live'
        bot = Trader(
            exchange=exchange,
            api_key=api_key,
            api_secret=api_secret,
            api_password=api_password,
            base_symbols=base_symbol.split(','),
            quote_symbol=quote_symbol,
            order_fee=order_fee,
            max_number_of_orders_per_symbol=max_number_of_orders_per_symbol,
            buy_order_price_offset=buy_order_price_offset,
            sell_order_price_offset=sell_order_price_offset,
            buy_trade_size=bid_trade_size,
            sell_trade_size=ask_trade_size,
            max_number_of_open_trades_for_symbol=max_number_of_open_trades_for_symbol,
            max_open_order_age=max_open_order_age,
            timeout=timeout,
            initial_balance_base=initial_balance_base,
            initial_balance_quote=initial_balance_quote,
            direction=direction,
            timeframes=timeframes,
            enable_psar=enable_psar,
            psar_timeframe=psar_timeframe,
            max_number_of_trades_for_period=max_number_of_trades_for_period,
            max_number_of_trades_period=max_number_of_trades_period,
            trade_channels=trade_channels
        )
    else:
        backtests_above_minimum_pnl = []
        for enable_psar in parameter_list(config['enable_psar']):
            for psar_timeframe in parameter_list(config['psar_timeframe'], return_first=not enable_psar):
                timeframes = []
                if enable_psar:
                    timeframes.append(psar_timeframe)
                for ask_trade_size in parameter_list(config['ask_trade_size']):
                    for bid_trade_size in parameter_list(config['bid_trade_size']):
                        for max_number_of_orders_per_symbol in parameter_list(config['max_number_of_orders_per_symbol']):
                            for buy_order_price_offset in parameter_list(config['buy_order_price_offset']):
                                for sell_order_price_offset in parameter_list(config['sell_order_price_offset']):
                                    for max_number_of_open_trades_for_symbol in parameter_list(config['max_number_of_open_trades_for_symbol']):
                                        for max_open_order_age in parameter_list(config['max_open_order_age']):
                                            for direction in parameter_list(config['direction']):
                                                
                                                mode_name = 'Backtest'
                                                parameters = dict(
                                                    exchange=exchange,
                                                    base_symbols=base_symbol.split(','),
                                                    quote_symbol=quote_symbol,
                                                    from_date=str(from_date),
                                                    to_date=str(to_date),
                                                    order_fee=order_fee,
                                                    max_number_of_orders_per_symbol=max_number_of_orders_per_symbol,
                                                    buy_order_price_offset=buy_order_price_offset,
                                                    sell_order_price_offset=sell_order_price_offset,
                                                    buy_trade_size=bid_trade_size,
                                                    sell_trade_size=ask_trade_size,
                                                    max_number_of_open_trades_for_symbol=max_number_of_open_trades_for_symbol,
                                                    max_open_order_age=max_open_order_age,
                                                    timeout=timeout,
                                                    initial_balance_base=initial_balance_base,
                                                    initial_balance_quote=initial_balance_quote,
                                                    direction=direction,
                                                    timeframes=timeframes,
                                                    enable_psar=enable_psar,
                                                    psar_timeframe=psar_timeframe,
                                                    max_number_of_trades_for_period=max_number_of_trades_for_period,
                                                    max_number_of_trades_period=max_number_of_trades_period,
                                                    trade_channels=trade_channels
                                                )
                                                bot = Backtester(**parameters)
                                                backtest_hash = sha256(str(parameters).encode('utf-8')).hexdigest()
                                                logger.info(f'Running backtest with id: {backtest_hash} using parameters: \n{parameters}')
                                                
                                                bot.run()
                                                csv_filename = f'{mode_name}_{backtest_hash}_results.csv'
                                                bot.trades_df.to_csv(f'backtest_results/{csv_filename}', index=False)
                                                if bot.enable_psar:
                                                    pd.DataFrame(bot.psar_signals_for_report).to_csv(f'{mode_name}_{backtest_hash}_psar_signals.csv')
                                                base_pnl = ((bot.balance_base - bot.initial_balance_base) / bot.initial_balance_base) * 100
                                                quote_pnl = ((bot.balance_quote - bot.initial_balance_quote) / bot.initial_balance_quote) * 100
                                                total_pnl = base_pnl + quote_pnl
                                                backtest_report_str = f'''
                                                Initial balance base: {bot.initial_balance_base}
                                                Final balance base: {bot.balance_base}
                                                Base PNL: {base_pnl}%
                                                Initial balance quote: {bot.initial_balance_quote}
                                                Final balance quote: {bot.balance_quote}
                                                Quote PNL: {quote_pnl}%
                                                '''
                                                logger.info(f'Backtest results:\n{backtest_report_str}')
                                                
                                                if total_pnl >= config['minimum_pnl']:
                                                    backtests_above_minimum_pnl.append({
                                                        'hash': backtest_hash,
                                                        'csv_filename': csv_filename,
                                                        'backtest_report_str': re.sub('\s+', ' ', backtest_report_str),
                                                        'parameters': parameters,
                                                        'total_pnl': total_pnl
                                                    })
        
        if backtests_above_minimum_pnl:
            backtests_above_minimum_pnl_str = json.dumps(backtests_above_minimum_pnl, indent=2, default=str)
            best_backtest = sorted(backtests_above_minimum_pnl, key=lambda x: x.get("total_pnl"))[-1]
            best_backtest_str = json.dumps(best_backtest, indent=2, default=str)
        else:
            backtests_above_minimum_pnl_str = 'None'
            best_backtest_str = 'None'

        logger.info(f'Backests above minimum pnl:\n{backtests_above_minimum_pnl_str}')
        logger.info(f'Best backtest: {best_backtest_str}')
        
    if USE_STREAMLIT:
        if st.button('START'):
            data_load_state = st.text('Running...')
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            bot.run()
            bot.trades_df.to_csv(f'{mode_name}_results.csv', index=False)
            data_load_state.text(f"{mode_name} finished. CSV file: {mode_name}_results.csv")
            for symbol in bot.symbols:
                st.text(f'{symbol} trades')
                st.dataframe(bot.trades_df[bot.trades_df['symbol'] == symbol])
    else:
        pass
        bot.run()
        bot.trades_df.to_csv(f'{mode_name}_results.csv', index=False)
        if bot.enable_psar:
            pd.DataFrame(bot.psar_signals_for_report).to_csv(f'{mode_name}_psar_signals.csv')
        base_pnl = ((bot.balance_base - bot.initial_balance_base) / bot.initial_balance_base) * 100
        quote_pnl = ((bot.balance_quote - bot.initial_balance_quote) / bot.initial_balance_quote) * 100
        logger.info(f'''
        Initial balance base: {bot.initial_balance_base}
        Final balance base: {bot.balance_base}
        Base PNL: {base_pnl}%
        Initial balance quote: {bot.initial_balance_quote}
        Final balance quote: {bot.balance_quote}
        Quote PNL: {quote_pnl}%
        ''')
