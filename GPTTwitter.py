import tweepy
import os
import pandas as pd
import yfinance as yf
import openai
import big_baller_moves
import json
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup

class GPTTwitter:
    def __init__(self, username):
        self.username = username
        self.client1 = tweepy.Client(bearer_token=big_baller_moves.bossman_tingz["twitter_api_key"])
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
        self.ht_dynamic = []  # Define ht_dynamic as a class attribute

    def get_user_id(self):
        try:
            user = self.client1.get_user(username=self.username)
            user_id = user.data.id
            print("User ID for", self.username, ":", user_id)
            return user_id
        except Exception as e:
            print("Error:", e)
            return None

    def get_tweets(self):
        return self.client1.get_users_tweets(
            id=self.user_id,
            max_results=100,  # Number of tweets to retrieve (adjust as needed)
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

    
    def contains_twitter_pic(url_list):
        if not url_list:
            return False
        return any('pic.twitter' in url for url in url_list)


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

    def get_jpg_url(driver, link):
        max_attempts = 10
        initial_wait = 0.5
        max_wait = 10
        for attempt in range(max_attempts):
            url = None
            try:
                if isinstance(link, list):  # Extract first element if URL is a list
                    for u in link:
                        if "twitter" in u:
                            url = u
                            break
                elif not link:
                    print("No Link Provided")
                    return None
                else:
                    url = link
                if not url:
                    return None
                url = f'https://{url}'
                print(f"Attempting to access URL: {url}")
                driver.get(url)
                # Wait for potential redirects and the page to stabilize
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
                time.sleep(1)  # Extended sleep to ensure the page is loaded
                images = driver.find_elements(By.TAG_NAME, "img")
                for img in images:
                    img_src = img.get_attribute('src')
                    if 'media' in img_src:
                        print("img_src: " + img_src)
                        return img_src

                print("No media images found on the page")
                return None
            except Exception as e:
                print(f"General error processing URL {url}: {e}")
            wait_time = initial_wait * (2 ** attempt) + random.uniform(0, 1)
            wait_time = min(wait_time, max_wait)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        print("Maximum attempts reached, aborting")
        return None


    def get_response_image(self, text):
        if text == 0:
            return "No image available"
        if not isinstance(text, str):
            print(text)
            text = text[0]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "What kind of stock purchase is this image describing? If it is an option play like a call or a put please specify."},
                    {"role": "user", "content": {"type": "image_url", "image_url": {"url": text}}}
                ],
                max_tokens=300
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(e)
            return None

    def dynamic_prompting(self, text, sys_prompt, user_prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt + "\n" + text}
                ],
                max_tokens=300
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(e)
            return None

    def process_tweets(self):
        self.df["image_url"] = self.df["entities"].apply(self.get_display_url)
        selected_columns = ['id', 'text', 'created_at', 'image_url']
        self.heisenberg_tweets = self.df[selected_columns].copy()

        try:
            self.heisenberg_tweets.loc[:, 'jpg_url'] = self.heisenberg_tweets['image_url'].apply(lambda x: self.get_jpg_url(x))
        finally:
            self.driver.quit()

        self.heisenberg_tweets['contains_image_url'] = self.heisenberg_tweets.apply['image_url'].apply(lambda x: self.contains_twitter_pic(x) if x != None else False)
        self.heisenberg_tweets = heisenberg_tweets[heisenberg_tweets['contains_image_url'] & heisenberg_tweets['jpg_url'].notna()]
        self.heisenberg_tweets['image_response'] = self.heisenberg_tweets['jpg_url'].apply(lambda x: self.get_response_image(x) if x != None else None)
        self.heisenberg_tweets['image_response'] = self.heisenberg_tweets['image_response'].apply(lambda x: x.replace('$', '\$') if x and pd.notna(x) else x)
        self.heisenberg_tweets['text'] = self.heisenberg_tweets['text'].apply(lambda x: x.replace('$', '\$') if pd.notna(x) else x)
        self.heisenberg_tweets['full_response'] = self.heisenberg_tweets['text'].astype(str) + '  TRANSCRIBED IMAGE DATA: ' + self.heisenberg_tweets['image_response'].astype(str)

    def save_to_excel(self, filename='ht_unprocessed.xlsx'):
        self.heisenberg_tweets['created_at'] = pd.to_datetime(self.heisenberg_tweets['created_at']).dt.tz_localize(None)
        self.heisenberg_tweets = self.heisenberg_tweets.applymap(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
        cwd = os.getcwd()
        file_path = os.path.join(cwd, filename)
        print(file_path)
        self.heisenberg_tweets.to_excel(file_path, index=False)

    def dynamic_prompt_and_save(self, sys_prompt, user_prompt, filename='out.xlsx'):
        cols = ['id', 'created_at', 'full_response']
        heisenberg_tweets = self.heisenberg_tweets[cols]
        heisenberg_tweets['result'] = heisenberg_tweets['full_response'].apply(
            lambda text: self.dynamic_prompting(text, sys_prompt, user_prompt)
        )
        heisenberg_tweets['created_at'] = pd.to_datetime(heisenberg_tweets['created_at']).dt.tz_localize(None)
        heisenberg_tweets = heisenberg_tweets.applymap(lambda x: x.encode('utf-8') if isinstance(x, str) else x)
        cwd = os.getcwd()
        file_path = os.path.join(cwd, filename)
        print(file_path)
        heisenberg_tweets.to_excel(file_path, index=False)

# Usage example:
#username = "Heisenberg_100k"
#gpt_twitter = GPTTwitter(username)
#gpt_twitter.process_tweets()
#gpt_twitter.save_to_excel()
 
