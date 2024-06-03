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
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException
from django.db import transaction
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.abspath(os.path.join(BASE_DIR, '../../../..'))
if config_path not in sys.path:
    sys.path.append(config_path)
import big_baller_moves

debug_mode = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketEnvironment:
    def __init__(self):
        self.us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        self.closest_time_index = None

    def adjust_to_trading_hours(self, dt):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = pytz.timezone('America/New_York').localize(dt)
        if dt.weekday() >= 5:  
            dt += self.us_bd
        if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
            dt = dt.replace(hour=9, minute=30)
        elif dt.hour >= 16:
            dt = (dt + self.us_bd).replace(hour=9, minute=30)
        return dt

    def save_stock_data(self, ticker, data):
        try:
            with transaction.atomic():
                for index, row in data.iterrows():
                    if isinstance(index, str):
                        index = datetime.fromisoformat(index)
                    if not index.tzinfo:
                        index = pytz.UTC.localize(index)
                    StockData.objects.update_or_create(
                        ticker=ticker,
                        date=index,
                        defaults={
                            'close': row['Close'],
                        }
                    )
            if debug_mode:
                logging.info(f"Successfully saved stock data for {ticker}")
        except Exception as e:
            logging.error(f"Failed to save stock data for {ticker}: {str(e)}")
            

    def fetch_market_data(self, ticker, signal_date):
        if ticker in ['U','YINN']:
            logging.error('Ticker is known to not be on RH so it wont work now.')
            return None,None,None

        if signal_date.tzinfo is None or signal_date.tzinfo.utcoffset(signal_date) is None:
            signal_date = pytz.timezone('America/New_York').localize(signal_date)
        signal_date = self.adjust_to_trading_hours(signal_date)
        
        start_date_utc = (signal_date - timedelta(days=1)).astimezone(pytz.utc)
        end_date_utc = (signal_date + timedelta(days=6)).astimezone(pytz.utc)
        attempts, wait = 7, 0.1
        for attempt in range(attempts):
            try:
                data = yf.download(ticker, start=start_date_utc.strftime('%Y-%m-%d'), end=end_date_utc.strftime('%Y-%m-%d'), interval='1m', progress=False)
                if data.empty:
                    continue
                data.index = data.index.tz_convert(pytz.timezone('America/New_York'))
                data = data[(data.index <= end_date_utc)]
                self.save_stock_data(ticker, data)
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
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return np.max(ranges, axis=1).rolling(window=period).mean()

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.market_env = MarketEnvironment()

    def run_backtest(self, data_for_atr, data_for_backtest, callout_price, atr_multiplier, trailing_stop_multiplier, atr_period):
        atr = self.market_env.calculate_atr(data_for_atr, atr_period).iloc[-1] * atr_multiplier
        profit_losses = ((data_for_backtest['Close'] - callout_price) / callout_price * 100).tolist()
        dates = data_for_backtest.index.tolist()
        return self.evaluate_trades(profit_losses, atr, trailing_stop_multiplier, dates)

    def evaluate_trades(self, profit_losses, atr, trailing_stop_multiplier,dates):
        portfolio = {
            'Capital': self.initial_capital,
            'Cash': self.initial_capital,
            'Equity': 0,
            'Returns': [],
            'Drawdowns': [],
            'Successful Trades': 0
        }
        initial_investment = portfolio['Cash']
        portfolio['Cash'] -= initial_investment
        portfolio['Equity'] = initial_investment
        max_profit_loss = 0
        minutes_taken = 0
        sold_at_date = None
        
        for i, (profit_loss, date) in enumerate(zip(profit_losses,dates)):
            if profit_loss > max_profit_loss:
                max_profit_loss = profit_loss
            if profit_loss < max_profit_loss - (atr * trailing_stop_multiplier) or i == len(profit_losses) - 1:
                sell_amount = portfolio['Equity']
                drawdown = ((1 + max_profit_loss / 100) / (1 + profit_loss / 100)) - 1
                portfolio['Drawdowns'].append(drawdown)
                portfolio['Cash'] += sell_amount * (1 + profit_loss / 100)
                portfolio['Equity'] -= sell_amount
                portfolio['Returns'].append(profit_loss / 100)
                minutes_taken = i
                sold_at_date = date
                if profit_loss > 1 and drawdown < 0.005:
                    portfolio['Successful Trades'] += 1
                break
        total_return = (portfolio['Cash'] - self.initial_capital) / self.initial_capital
        portfolio_variance = np.var(portfolio['Returns']) if portfolio['Returns'] else 0
        sharpe_ratio = total_return / np.sqrt(portfolio_variance) if portfolio_variance else 0
        max_drawdown = max(portfolio['Drawdowns']) if portfolio['Drawdowns'] else 0
        avg_trade_gain = np.mean(portfolio['Returns']) if portfolio['Returns'] else 0
        return {
            'Total Return': total_return*100,
            'Portfolio Variance': portfolio_variance,
            'Sharpe Ratio': sharpe_ratio,
            'Final Equity': portfolio['Cash'],
            'Maximum Drawdown': max_drawdown,
            'Average Trade Gain': avg_trade_gain,
            'Successful Trades': portfolio['Successful Trades'],
            'Minutes Taken': minutes_taken,
            'Sold At Date':sold_at_date
        }

def optimize_strategy(ticker, created_at, data_for_atr, data_for_backtest, callout_price, param_ranges, backtester):
    results = []
    for atr_mult, stop_mult, atr_period in itertools.product(param_ranges['atr_multiplier'], param_ranges['trailing_stop_loss_multiplier'], param_ranges['atr_periods']):
        result = backtester.run_backtest(data_for_atr, data_for_backtest, callout_price, atr_mult, stop_mult, atr_period)
        if result:
            results.append({
                'ticker': ticker,
                'created_at': created_at,
                'atr_multiplier': atr_mult,
                'trailing_stop_multiplier': stop_mult,
                'atr_period': atr_period,
                'total_return': result['Total Return'],
                'portfolio_variance': result['Portfolio Variance'],
                'sharpe_ratio': result['Sharpe Ratio'],
                'final_equity': result['Final Equity'],
                'maximum_drawdown': result['Maximum Drawdown'],
                'successful_trades': result['Successful Trades'],
                'minutes_taken': result['Minutes Taken'],
                'sold_at_date' : result['Sold At Date'],
                'score': (result['Total Return'] - result['Maximum Drawdown']) / (result['Minutes Taken'] + 0.0001) * 6000
            })

    return results

def process_row(row,backtester,param_ranges):
    data_for_atr, data_for_backtest, callout_price = backtester.market_env.fetch_market_data(row['ticker'], pd.to_datetime(row['created_at']))
    if data_for_atr is not None and data_for_backtest is not None:
        return optimize_strategy(row['ticker'], row['created_at'], data_for_atr, data_for_backtest, callout_price, param_ranges, backtester)
    else:
        return []
    
def parallel_optimize_strategy(df, param_ranges):
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
    ht_dynamic = []
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
        self.node_urls = [
            "http://localhost:5555/wd/hub",  # url for first chrome node
            "http://localhost:5556/wd/hub"   # url for second chrome node
        ]
        self.user_id = self.get_user_id()
        self.tweets = self.get_tweets()
        self.df = pd.DataFrame(self.tweets.data)
        self.df["text"] = [(str(i).replace(",", "").replace('$', '').replace('\\', '').replace('*','')) for i in self.df["text"]]
        self.heisenberg_tweets = pd.DataFrame()
        self.heisenberg_tweets = pd.DataFrame()
        self.drivers = []
        
    def get_user_id(self):
        try:
            user = self.client.get_user(username=self.username)
            user_id = user.data.id
            logging.info("User ID for " + self.username + ": " + str(user_id))
            return user_id
        except Exception as e:
            logging.error("Error:" + str(e))
            return None

    def get_tweets(self):
        return self.client.get_users_tweets(
            id = self.user_id,
            max_results = 25,  # Number of tweets to retrieve (can adjust as needed)
            tweet_fields = ['id', 'text', 'created_at', 'entities', 'attachments'],  # Fields we  want to retrieve for each tweet
            media_fields = ['preview_image_url', 'url'],
            exclude = ['retweets', 'replies'],
            expansions = ['attachments.media_keys', 'author_id']
        )

    def get_display_url(self, entities):
        if isinstance(entities, dict) and "urls" in entities:
            urls = entities["urls"]
            display_urls = [url.get("display_url") for url in urls]
            return display_urls
        else:
            return 0

    def initialize_webdriver(self):
        options = webdriver.ChromeOptions()  # Example for Chrome, change as needed
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-dev-tools")
        options.add_argument('blink-settings=imagesEnabled=false')

        # Specify capabilities if needed, or just rely on default
        capabilities = options.to_capabilities()

        driver = webdriver.Remote(
            command_executor="http://localhost:4444/wd/hub",  # Replace with your hub URL
            desired_capabilities=capabilities
        )
        return driver
    
    def close_drivers(self):
        for driver in self.drivers:
            driver.quit()
        self.drivers = []
    
    async def get_response_image(self, text):
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
                        {"type": "image_url", "image_url": {"url": text}},  # Ensure this matches the expected structure
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
    
    def get_jpg_url(self, driver, link):
        self.ht_dynamic = []
        max_attempts, wait_time = 2, 0.5
        for attempt in range(max_attempts):
            try:
                if isinstance(link, list):  # Extract first element if URL is a list
                    for u in link:
                        if "twitter" in u:
                            link = u
                            break
                if not link:
                    return None
                url = f'https://{link}'
                driver.get(url)
                # Wait for potential redirects and the page to stabilize
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
                time.sleep(wait_time)  # Extended sleep to ensure the page is loaded
                images = driver.find_elements(By.TAG_NAME, "img")
                for img in images:
                    img_src = img.get_attribute('src')
                    if 'media' in img_src and 'twimg' in img_src:
                        return img_src
                logging.info("No media images found on the page")
            except (StaleElementReferenceException, WebDriverException) as e:
                time.sleep(attempt + 0.5)  # Wait before retrying
                logging.error(f"Attempt {str(attempt + 1)}: Error with {url} - {str(e)}")
            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(f"General error processing URL {url}: {str(e)}")
        return None
    
    def initialize_webdrivers(self):
        # Clear existing drivers if any
        self.close_drivers()
        # Initialize a WebDriver for each node URL
        for node_url in self.node_urls:
            self.drivers.append(self.initialize_webdriver())

    def fetch_images_concurrently(self, links):
        results = [None] * len(links)  # Pre-fill results with None for each link
        # Map each link to a driver, ensuring no more drivers are used than available
        link_driver_pairs = zip(links, cycle(self.drivers))  # cycle to reuse drivers if more links than drivers

        # Use ThreadPoolExecutor to manage concurrent WebDriver usage
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_link = {executor.submit(self.get_jpg_url, driver, link): idx for idx, (link, driver) in enumerate(link_driver_pairs)}
            for future in as_completed(future_to_link):
                idx = future_to_link[future]
                try:
                    result = future.result()
                    results[idx] = result  # Place result in the corresponding position
                    if debug_mode:
                        if result:
                            logging.info(f"Image found for link index {str(idx)}: {result}")
                        else:
                            logging.info(f"No image found for link index {str(idx)}.")
                except Exception as e:
                    logging.info(f"Error processing link at index {idx}: {str(e)}")
                    results[idx] = None  # Ensure failed results are marked as None
        return results
    
    def process_tweets(self):
        self.ht_dynamic = []
        if not self.df.empty:
            self.initialize_webdrivers()
            self.df["image_url"] = self.df["entities"].apply(self.get_display_url)
            selected_columns = ['id', 'text', 'created_at', 'image_url']
            self.heisenberg_tweets = self.df[selected_columns].copy()
            self.heisenberg_tweets['image_response'] = None  # Initialize the column to avoid key errors
            self.heisenberg_tweets['created_at'] = pd.to_datetime(self.heisenberg_tweets['created_at']).dt.tz_convert('America/New_York')
            
            urls_to_fetch = [url for url in self.heisenberg_tweets['image_url'] if url is not None]
            if urls_to_fetch:
                logging.info(f"Fetching images for {str(len(urls_to_fetch))} URLs.")
                image_urls = self.fetch_images_concurrently(urls_to_fetch)
                for idx, img_url in enumerate(image_urls):
                    if img_url:
                        self.heisenberg_tweets.at[idx, 'jpg_url'] = img_url

            if self.heisenberg_tweets['jpg_url'].any():
                non_null_urls = self.heisenberg_tweets['jpg_url'].dropna().tolist()
                image_responses = asyncio.run(self.fetch_image_responses(non_null_urls))
                non_null_indices = self.heisenberg_tweets.index[self.heisenberg_tweets['jpg_url'].notnull()].tolist()
                for idx, content in zip(non_null_indices, image_responses):
                    if content:
                        self.heisenberg_tweets.at[idx, 'image_response'] = self.clean_response(content)

            self.heisenberg_tweets['full_response'] = self.heisenberg_tweets.apply(
                lambda row: f"{self.clean_response(row['text'])} TRANSCRIBED IMAGE DATA: {row.get('image_response', '')}", axis=1)
            
            if debug_mode:
                for i, row in self.heisenberg_tweets.iterrows():
                    logging.info('response: ' + str(row['full_response']) + '  jpg_url: ' + str(row['image_url']) + '\n==============')
            self.heisenberg_tweets = self.heisenberg_tweets.copy()
        else:
            logging.error("No data to process. DataFrame is empty.")
                    
    def dynamic_prompt_and_save(self, sys_prompt, user_prompt):
        if self.heisenberg_tweets is not None and not self.heisenberg_tweets.empty:
            async def fetch_and_process_all():
                tasks = [self.dynamic_prompting(row['full_response'], sys_prompt, user_prompt) for _, row in self.heisenberg_tweets.iterrows()]
                responses = await asyncio.gather(*tasks)
                return responses

            # Run the asynchronous tasks and fetch responses
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
