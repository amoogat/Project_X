import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, random, os, tweepy, time, logging, itertools, pytz, sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.abspath(os.path.join(BASE_DIR, '../../../..'))
if config_path not in sys.path:
    sys.path.append(config_path)
import big_baller_moves

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
        elif dt.hour > 16:
            dt = (dt + self.us_bd).replace(hour=9, minute=30)
        return dt

    def fetch_market_data(self, ticker, signal_date):
        if signal_date.tzinfo is None or signal_date.tzinfo.utcoffset(signal_date) is None:
            signal_date = self.adjust_to_trading_hours(pytz.timezone('America/New_York').localize(signal_date))
        else:
            signal_date = self.adjust_to_trading_hours(signal_date.tz_convert(pytz.timezone('America/New_York')))
        
        start_date_utc = (signal_date - timedelta(days=1)).astimezone(pytz.utc)
        end_date_utc = (signal_date + timedelta(days=6)).astimezone(pytz.utc)
        attempts, wait = 20, 0.001
        for attempt in range(attempts):
            try:
                data = yf.download(ticker, start=start_date_utc.strftime('%Y-%m-%d'), end=end_date_utc.strftime('%Y-%m-%d'), interval='1m', progress=False)
                if data.empty:
                    continue
                data.index = data.index.tz_convert(pytz.timezone('America/New_York'))
                self.closest_time_index = data.index.get_loc(signal_date, method='nearest')
                callout_price = data.at[data.index[self.closest_time_index], 'Close']
                data_for_atr = data.loc[:data.index[self.closest_time_index + 1]]
                data_for_backtest = data.loc[data.index[self.closest_time_index]:]
                return data_for_atr, data_for_backtest, callout_price
            except Exception as e:
                time.sleep(wait)
                wait *= 2
        logging.error(f"Failed to download data for {ticker} after {attempts} attempts.")
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
        return self.evaluate_trades(profit_losses, atr, trailing_stop_multiplier)

    def evaluate_trades(self, profit_losses, atr, trailing_stop_multiplier):
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
        for i, profit_loss in enumerate(profit_losses):
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
                if profit_loss > 1 and drawdown < 0.005:
                    portfolio['Successful Trades'] += 1
                break
        total_return = (portfolio['Cash'] - self.initial_capital) / self.initial_capital
        portfolio_variance = np.var(portfolio['Returns']) if portfolio['Returns'] else 0
        sharpe_ratio = total_return / np.sqrt(portfolio_variance) if portfolio_variance else 0
        max_drawdown = max(portfolio['Drawdowns']) if portfolio['Drawdowns'] else 0
        avg_trade_gain = np.mean(portfolio['Returns']) if portfolio['Returns'] else 0
        return {
            'Total Return': total_return,
            'Portfolio Variance': portfolio_variance,
            'Sharpe Ratio': sharpe_ratio,
            'Final Equity': portfolio['Cash'],
            'Maximum Drawdown': max_drawdown,
            'Average Trade Gain': avg_trade_gain,
            'Successful Trades': portfolio['Successful Trades'],
            'Minutes Taken': minutes_taken
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
                'score': (result['Total Return'] - result['Maximum Drawdown']) / (result['Minutes Taken'] + 0.0001) * 6000
            })

    return results

def process_row(row,backtester,param_ranges):
    data_for_atr, data_for_backtest, callout_price = backtester.market_env.fetch_market_data(row['Ticker'], pd.to_datetime(row['created_at']))
    if data_for_atr is not None and data_for_backtest is not None:
        return optimize_strategy(row['Ticker'], row['created_at'], data_for_atr, data_for_backtest, callout_price, param_ranges, backtester)
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
        self.driver = self.initialize_webdriver()
        self.user_id = self.get_user_id()
        self.tweets = self.get_tweets()
        self.df = pd.DataFrame(self.tweets.data)
        self.heisenberg_tweets = None
        self.ht_dynamic = None

    def get_user_id(self):
        try:
            user = self.client.get_user(username=self.username)
            user_id = user.data.id
            print("User ID for", self.username, ":", user_id)
            return user_id
        except Exception as e:
            print("Error:", e)
            return None

    def get_tweets(self):
        return self.client.get_users_tweets(
            id=self.user_id,
            max_results=20,  # Number of tweets to retrieve (adjust as needed)
            tweet_fields=['id', 'text', 'created_at', 'entities', 'attachments'],  # Fields you want to retrieve for each tweet
            media_fields=['preview_image_url', 'url'],
            exclude=['retweets', 'replies'],
            expansions=['attachments.media_keys', 'author_id']
        )

    def get_display_url(self, entities):
        if isinstance(entities, dict) and "urls" in entities:
            urls = entities["urls"]
            display_urls = [url.get("display_url") for url in urls]
            return display_urls
        else:
            return 0

    def initialize_webdriver(self):
        options = Options()
        options.add_argument("--headless")  # Run in background without opening a browser window
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-dev-tools")
        options.add_argument('blink-settings=imagesEnabled=false')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def get_jpg_url(self, link):
        max_attempts = 1
        initial_wait, wait_time = 0.01, 0.01
        max_wait = 1
        for attempt in range(max_attempts):
            url = None
            try:
                if isinstance(link, list):  # Extract first element if URL is a list
                    for u in link:
                        if "twitter" in u:
                            url = u
                            break
                elif not link:
                    # print("No Link Provided")
                    return None
                else:
                    url = link
                if not url:
                    return None
                url = f'https://{url}'
                # print(f"Attempting to access URL: {url}")
                self.driver.get(url)
                # Wait for potential redirects and the page to stabilize
                WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
                time.sleep(wait_time)  # Extended sleep to ensure the page is loaded
                images = self.driver.find_elements(By.TAG_NAME, "img")
                for img in images:
                    img_src = img.get_attribute('src')
                    if 'media' in img_src:
                        print("img_src: " + img_src)
                        return img_src
                # # Wait for potential redirects and the page to stabilize
                # WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
                # time.sleep(wait_time)  # Extended sleep to ensure the page is loaded

                # # Navigate through each 'article' tag; articles in retweets are nested within the original tweet 'article'
                # articles = self.driver.find_elements(By.TAG_NAME, "article")
                # for article in articles:
                #     images = article.find_elements(By.TAG_NAME, "img")
                #     for img in images:
                #         img_src = img.get_attribute('src')
                #         if 'media' in img_src:
                #             print("Image source found:", img_src)
                #             img_sources.append(img_src)
                print("No media images found on the page")
                return None
            except Exception as e:
                print(f"General error processing URL {url}: {e}")
            wait_time = min(initial_wait * 2, max_wait)
            # print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        print("Maximum attempts reached, aborting")
        return None

    def get_response_image(self, text) :
        if text == 0:
            return "No image available"
        if not isinstance(text, str):
            print(text)
            text = text[0]
        try:
            response = openai.chat.completions.create(model="gpt-4o", messages=[
                                {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What kind of stock purchase is this image describing? If it is an option play like a call or a put please specify."},
                                    {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": text,
                                    },
                                    },
                                ],
                                }
                            ], max_tokens=300,)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            pass

    def dynamic_prompting(self, text, sys_prompt, user_prompt):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "text", "text": text}
                        ]
                    }
                ],
                max_tokens=300,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            pass

    def process_tweets(self):
        self.df["image_url"] = self.df["entities"].apply(self.get_display_url)
        selected_columns = ['id', 'text', 'created_at', 'image_url']
        self.heisenberg_tweets = self.df[selected_columns].copy()

        try:
            self.heisenberg_tweets.loc[:, 'jpg_url'] = self.heisenberg_tweets['image_url'].apply(lambda x: self.get_jpg_url(x))
        finally:
            self.driver.quit()


        self.heisenberg_tweets.loc[:,'image_response'] = self.heisenberg_tweets['jpg_url'].apply(lambda x: self.get_response_image(x) if x != None else None)
        self.heisenberg_tweets['image_response'] = self.heisenberg_tweets['image_response'].apply(lambda x: x.replace('$', '\$') if x and pd.notna(x) else x)
        self.heisenberg_tweets['text'] = self.heisenberg_tweets['text'].apply(lambda x: x.replace('$', '\$') if pd.notna(x) else x)
        self.heisenberg_tweets['full_response'] = self.heisenberg_tweets['text'].astype(str) + '  TRANSCRIBED IMAGE DATA: ' + self.heisenberg_tweets['image_response'].astype(str)


    def dynamic_prompt_and_save(self, sys_prompt, user_prompt):
        cols = ['id', 'created_at', 'full_response']
        self.ht_dynamic = self.heisenberg_tweets[cols]
        self.ht_dynamic['result'] = self.ht_dynamic['full_response'].apply(
            lambda text: self.dynamic_prompting(text, sys_prompt, user_prompt)
        )
        self.ht_dynamic['created_at'] = pd.to_datetime(self.ht_dynamic['created_at']).dt.tz_localize(None)
        self.ht_dynamic = self.ht_dynamic.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
