import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, os, tweepy, time, logging, itertools, pytz, sys
from itertools import cycle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from.models import StockData
import httpx
import asyncio
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.abspath(os.path.join(BASE_DIR, '../../../..'))
if config_path not in sys.path:
    sys.path.append(config_path)
import big_baller_moves

debug_level = 0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketEnvironment:
    def __init__(self):
        self.us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        self.closest_time_index = None

    def adjust_to_trading_hours(self, dt):
        # Ensures datetime timestamp is correct and timestmap is in the NYSE trading hours
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = pytz.timezone('America/New_York').localize(dt)
        if dt.weekday() >= 5:  
            dt += self.us_bd
        if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
            dt = dt.replace(hour=9, minute=30)
        elif dt.hour >= 16:
            dt = (dt + self.us_bd).replace(hour=9, minute=30)
        return dt
    
    def save_stock_data(self, ticker, data, tweet_text):
        try:
            # Saves stock data if not already in our database
            existing_dates = set(StockData.objects.filter(ticker=ticker).values_list('date', flat=True))
            bulk_list = []
            for index, row in data.iterrows():
                if isinstance(index, str):
                    index = datetime.fromisoformat(index)
                if not index.tzinfo:
                    index = pytz.UTC.localize(index)
                if index not in existing_dates:
                    bulk_list.append(
                        StockData(
                            ticker=ticker,
                            date=index,
                            close=row['Close'],
                            tweet_text=tweet_text
                        )
                    )
            if bulk_list:
                StockData.objects.bulk_create(bulk_list, ignore_conflicts=True)
            elif debug_level > 0:
                logging.info(f"{ticker} on {index} has already been processed, skipping save.")
            if debug_level > 0:
                logging.info(f"Successfully saved stock data for {ticker}")
        except Exception as e:
            logging.error(f"Failed to save stock data for {ticker}: {str(e)}")

    def fetch_market_data(self, ticker, signal_date, tweet_text):
        # Splits up data and attempts to download from YFINANCE 
        signal_date = self.adjust_to_trading_hours(signal_date)
        start_date_utc = (signal_date - timedelta(days=1)).astimezone(pytz.utc)
        end_date_utc = (signal_date + timedelta(days=6)).astimezone(pytz.utc)
        attempts, wait = 7, 0.1
        for _ in range(attempts):
            try:
                data = yf.download(ticker, start=start_date_utc.strftime('%Y-%m-%d'), end=end_date_utc.strftime('%Y-%m-%d'), interval='1m', progress=False)
                if data.empty:
                    continue
                data.index = data.index.tz_convert(pytz.timezone('America/New_York'))
                data = data[(data.index <= end_date_utc)]
                # Saves stock data, splits data into backtest and ATR calculation (6:1 ratio)
                self.save_stock_data(ticker, data, tweet_text)
                self.closest_time_index = data.index.get_loc(signal_date, method='nearest')
                callout_price = data.at[data.index[self.closest_time_index], 'Close']
                data_for_atr = data.loc[:data.index[self.closest_time_index + 1]]
                data_for_backtest = data.loc[data.index[self.closest_time_index]:]
                return data_for_atr, data_for_backtest, callout_price
            except Exception as e:
                time.sleep(wait)
                wait *= 2
                if wait > 5:
                    wait = 0.01
        logging.error(f"Failed to download data for {ticker} after {str(attempts)} attempts.")
        return None, None, None

    def calculate_atr(self, data, period):
        # Gets ATR as an indicator for exit conditions
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return np.max(ranges, axis=1).rolling(window=period).mean()


class Backtester:
    def __init__(self):
        self.market_env = MarketEnvironment()
        self.first_bought_at = None
        self.last_sold_at = None
        self.portfolio = pd.DataFrame()

    def run_backtest(self, data_for_atr, data_for_backtest, callout_price, atr_multiplier, atr_period):
        # Gets continuous P/L list to evaluate callout based on consequent price data
        atr = self.market_env.calculate_atr(data_for_atr, atr_period).iloc[-1] * atr_multiplier
        profit_losses = ((data_for_backtest['Close'] - callout_price) / callout_price).tolist()
        dates = data_for_backtest.index.tolist()
        return self.evaluate_trades(profit_losses, atr, dates)

    def evaluate_trades(self, profit_losses, atr, dates):
        # Backtests profit loss for an individual trade
        max_profit_loss, max_drawdown, minutes_taken = 0, 0, 0
        sold_at_date = None
        for i, (profit_loss, date) in enumerate(zip(profit_losses, dates)):
            if profit_loss > max_profit_loss:
                max_profit_loss = profit_loss
            if profit_loss < max_drawdown:
                max_drawdown = profit_loss
            if profit_loss < max_profit_loss - (atr) or i == len(profit_losses) - 1:
                total_return = profit_loss * 100
                minutes_taken = i
                sold_at_date = date
                break
            
        # Calculate variance, max drawdown and sharpe ratio for future analyses
        portfolio_variance = np.var(pd.Series(profit_losses)) if profit_losses else 0
        sharpe_ratio = total_return / np.sqrt(portfolio_variance) if portfolio_variance else 0
        return {
            'Total Return' : total_return,
            'Portfolio Variance': portfolio_variance,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Minutes Taken': minutes_taken,
            'Sold At Date': sold_at_date
        }
    
    def set_first_bought_at(self, date):
        self.first_bought_at = date

    def set_last_sold_at(self, date):
        self.last_sold_at = date
            
    def fetch_market_data(self, ticker, start_date, end_date):
        # Gets the market data from yahoo finance wihthin our trading dates at a 1m interval
        start_date_utc = start_date.astimezone(pytz.utc).strftime('%Y-%m-%d')
        end_date += timedelta(days=1)
        end_date_utc = end_date.astimezone(pytz.utc).strftime('%Y-%m-%d')
        try:
            data = yf.download(ticker, start=start_date_utc, end=end_date_utc, interval='1m', progress=False)
            data.index = data.index.tz_convert('America/New_York')
            return data['Close']
        except Exception as e:
            logging.error(f"Failed to download data for {ticker}: {str(e)}")
            return None

    def initialize_portfolio(self):
        # Creates a dataframe with the index being market minutes from the first callout to last sale
        if self.first_bought_at and self.last_sold_at:
            self.first_bought_at = self.first_bought_at
            self.last_sold_at = self.last_sold_at
            
            data_range = pd.date_range(start=self.first_bought_at, end=self.last_sold_at, freq='T', tz='America/New_York')
            data_range = data_range[data_range.indexer_between_time('09:30', '15:59')]
            data_range = data_range[data_range.dayofweek < 5]

            self.portfolio = pd.DataFrame(index=data_range) 
            
            if debug_level > 1:
                logging.info(f"Portfolio initialized with start: {self.portfolio.index[0]} and end: {self.portfolio.index[-1]}")
                logging.info(f"Initializing portfolio from {self.first_bought_at} to {self.last_sold_at}")
                logging.info(f"Data range for portfolio: {data_range}")

    def evaluate_all_trades(self, trades, data):
        trade_counter = 0
        for trade in trades:
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            
            # Adjust dates to within trading hours - skips ahead if not
            entry_date = self.market_env.adjust_to_trading_hours(entry_date)
            exit_date = self.market_env.adjust_to_trading_hours(exit_date)
            
            # Checks if entry and exit dates are present in the data
            if entry_date not in data.index:
                logging.info('Adjusting entry dt index that was out of range.')
                entry_date = data.index[0]
            if exit_date not in data.index:
                logging.info('Adjusting exit dt index that was out of range.')
                exit_date = data.index[-1]

            entry_price = data.at[entry_date]
            exit_price = data.at[exit_date]

            trade_dates = pd.date_range(start=entry_date, end=exit_date, freq='T', tz='America/New_York')
            trade_dates = trade_dates.intersection(self.portfolio.index)
            
            if debug_level > 1:
                logging.info(f"Processing trade from {trade['entry_date']} to {trade['exit_date']}")
                logging.info(f"Adjusted entry_date: {entry_date}, exit_date: {exit_date}")
                logging.info(f"Entry price: {entry_price}, Exit price: {exit_price}")
                logging.info(f"Trade dates: {trade_dates}")

            # Calculates the current profit loss into the dataframe
            if not trade_dates.empty:
                temp_series = pd.Series(1, index=self.portfolio.index)
                for date in trade_dates:
                    if date in data.index:
                        current_price = data.at[date]
                        profit_loss_ratio = current_price / entry_price
                        temp_series.loc[date:] *= profit_loss_ratio
                        entry_price = current_price
                unique_column_name = f"Trade_{entry_date}_{trade_counter}"
                self.portfolio[unique_column_name] = temp_series
                self.portfolio.fillna(method='ffill', inplace=True)
                trade_counter += 1 

            else:
                logging.error(f"No intersection between trade dates and portfolio index for trade from {trade['entry_date']} to {trade['exit_date']}.")
    
    def finalize_portfolio(self):
        if not self.portfolio.empty:
            # Calculates the average value across all trades for each timestamp
            self.portfolio['Average'] = self.portfolio.mean(axis=1)
            self.portfolio = self.portfolio['Average']  
        self.portfolio.ffill(inplace=True)

def optimize_strategy(ticker, created_at, data_for_atr, data_for_backtest, callout_price, param_ranges, backtester):
    # Loops through around 150 parameter combinations to find an arbitrary "good" exit point
    results = []
    for atr_mult, atr_period in itertools.product(param_ranges['atr_multiplier'], param_ranges['atr_periods']):
        result = backtester.run_backtest(data_for_atr, data_for_backtest, callout_price, atr_mult, atr_period)
        if result:
            tr = result['Total Return']
            md = result['Maximum Drawdown']
            mt = result['Minutes Taken']
            sr =  result['Sharpe Ratio']
            results.append({
                'ticker': ticker,
                'created_at': created_at,
                'atr_multiplier': atr_mult,
                'atr_period': atr_period,
                'total_return': tr,
                'portfolio_variance': result['Portfolio Variance'],
                'sharpe_ratio': sr,
                'maximum_drawdown': md,
                'minutes_taken': mt,
                'sold_at_date' : result['Sold At Date'],
                'score': (tr + md + sr) / (mt + 0.0001) * 6000
            })
    return results

def process_row(row,backtester,param_ranges):
    # Fetches market data for a ticker starting from a callout date, optimizes strategy
    data_for_atr, data_for_backtest, callout_price = backtester.market_env.fetch_market_data(row['ticker'], pd.to_datetime(row['created_at']), row['text'])
    if data_for_atr is not None and data_for_backtest is not None:
        return optimize_strategy(row['ticker'], row['created_at'], data_for_atr, data_for_backtest, callout_price, param_ranges, backtester)
    else:
        return []
    
def parallel_optimize_strategy(df, param_ranges):
    # Optimizes a strategy using ATR
    backtester = Backtester()
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_row, row, backtester, param_ranges): row for _, row in df.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.extend(result)
    return pd.DataFrame(results)


class GPTTwitter:
    def __init__(self, username):
        self.username = username
        self.client = tweepy.Client(bearer_token=big_baller_moves.bossman_tingz["twitter_api_key"])
        self.auth = tweepy.OAuth1UserHandler(
            big_baller_moves.bossman_tingz["consumer_key"], 
            big_baller_moves.bossman_tingz["consumer_secret"], 
            big_baller_moves.bossman_tingz["access_token"], 
            big_baller_moves.bossman_tingz["access_token_secret"]
        )
        self.api = tweepy.API(self.auth)
        openai.api_key = big_baller_moves.bossman_tingz["openai_api_key"]
        self.user_id = self.get_user_id()
        self.heisenberg_tweets = pd.DataFrame()
        self.drivers = []

    def get_user_id(self):
        # Needed for tweepy :)
        try:
            user = self.client.get_user(username=self.username)
            user_id = user.data.id
            logging.info("User ID for " + self.username + ": " + str(user_id))
            return user_id
        except Exception as e:
            logging.error("Error:" + str(e))
            return None

    async def get_response_image(self, text):
        # Asynchronously gets the stock purchase information out of an image from OpenAI 4o
        if not text:
            return "No image available"
        if not isinstance(text, str):
            logging.error(f"Unexpected text format: {text}")
            text = text[0]
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {big_baller_moves.bossman_tingz['openai_api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What kind of stock purchase is this image describing? If it is an option play, please specify if it is ultimately bullish (long) or bearish (short)."},
                        {"type": "image_url", "image_url": {"url": text,"detail":"auto"}},
                    ]
                }
            ]
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                logging.error(f"Failed to fetch data: {response.text}, Status Code: {str(response.status_code)}")
                return None

    async def dynamic_prompting(self, text, sys_prompt, user_prompt):
        # Asynchronously gets [Open/Close] [Ticker] [Long/Short] from tweet + image data
        url = "https://api.openai.com/v1/chat/completions"  
        headers = {
            "Authorization": f"Bearer {big_baller_moves.bossman_tingz['openai_api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt + text
                }
            ]
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content'].strip()
                else:
                    logging.error(f"Failed to fetch data: {response.text}")
                    return None
            except httpx.RequestTimeout:
                logging.error("Request timed out")

    async def fetch_image_responses(self,image_urls):
        tasks = [self.get_response_image(url) for url in image_urls if url]
        return await asyncio.gather(*tasks)
    
    def clean_response(self, response):
        return str(response).replace('"', '').replace("'", '').replace('$', '\$').replace('*', '').replace(',', '') if response else ''
    
    def get_tweets(self):
        # Uses tweepy for fetching tweets with relevant fields
        tweets_with_media = []
        media_dict = {}
        tweet_objects = self.client.get_users_tweets(id=self.user_id, max_results=25,
                            tweet_fields=['id', 'text', 'created_at', 'attachments'],
                            media_fields=['url'], expansions=['attachments.media_keys'])
        if tweet_objects.includes and 'media' in tweet_objects.includes:
            media_dict = {media.media_key: media.url for media in tweet_objects.includes['media']}
        for tweet in tweet_objects.data:
            # Ensuring 'attachments' exists and has 'media_keys'
            if 'attachments' in tweet.data and 'media_keys' in tweet.data['attachments']:
                tweet_media_urls = [media_dict[key] for key in tweet.data['attachments']['media_keys']]
                tweets_with_media.append((tweet, tweet_media_urls))
            else:
                tweets_with_media.append((tweet, []))
        return tweets_with_media

    def process_tweets(self):
        self.df = pd.DataFrame([{
            'id': tweet_data[0].id,
            'text': tweet_data[0].text,
            'created_at': tweet_data[0].created_at,
            'image_urls': tweet_data[1]
        } for tweet_data in self.get_tweets()])

        if not self.df.empty:
            self.heisenberg_tweets = self.df[['id', 'text', 'created_at', 'image_urls']].copy()
            self.heisenberg_tweets['image_response'] = None
            self.heisenberg_tweets['created_at'] = pd.to_datetime(self.heisenberg_tweets['created_at']).dt.tz_convert('America/New_York')
            self.heisenberg_tweets['image_urls'] = self.heisenberg_tweets['image_urls'].apply(lambda x: x if x else None)

            # Filtering non-null URLs for processing
            non_null_urls = [url for sublist in self.heisenberg_tweets['image_urls'].dropna().tolist() for url in sublist]
            
            if non_null_urls:
                # Runs the OpenAI calls to turn an image -> transcribed image data
                image_responses = asyncio.run(self.fetch_image_responses(non_null_urls))
                response_iterator = iter(image_responses)
                for idx, urls in self.heisenberg_tweets[self.heisenberg_tweets['image_urls'].notnull()].iterrows():
                    for url in urls['image_urls']:
                        content = next(response_iterator, None)
                        if content:
                            self.heisenberg_tweets.at[idx, 'image_response'] = (self.heisenberg_tweets.at[idx, 'image_response'] or '') + self.clean_response(content) + ' '
            
            # Combine text and transcribed image data into a full response
            self.heisenberg_tweets['full_response'] = self.heisenberg_tweets.apply(
                lambda row: f"{self.clean_response(row['text'])} TRANSCRIBED IMAGE DATA: {row.get('image_response', '')}", axis=1)

            if debug_level > 1:
                for i, row in self.heisenberg_tweets.iterrows():
                    logging.info('response: ' + str(row['full_response']) + '  image_urls: ' + str(row['image_urls']) + '\n==============')
        else:
            logging.error("No data to process. DataFrame is empty.")
    def dynamic_prompt_and_save(self, sys_prompt, user_prompt):
        # Async runs openAI calls tweet + transcribed image data -> [open/close] [Ticker] [Long/Close]
        if self.heisenberg_tweets is not None and not self.heisenberg_tweets.empty:
            async def fetch_and_process_all():
                tasks = [self.dynamic_prompting(row['full_response'], sys_prompt, user_prompt) for _, row in self.heisenberg_tweets.iterrows()]
                responses = await asyncio.gather(*tasks)
                return responses

            responses = asyncio.run(fetch_and_process_all())

            # Check if 'result' column can be added or if it already exists
            if 'result' in self.heisenberg_tweets.columns:
                self.heisenberg_tweets['result'] = responses
            else:
                self.heisenberg_tweets = self.heisenberg_tweets.assign(result=responses)
                
            self.heisenberg_tweets['created_at'] = pd.to_datetime(self.heisenberg_tweets['created_at']).dt.tz_localize(None)
            self.heisenberg_tweets = self.heisenberg_tweets.applymap(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
        else:
            logging.error("heisenberg_tweets DataFrame is empty or not initialized.")