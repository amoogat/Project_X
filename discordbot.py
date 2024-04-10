import requests
import json
import pandas as pd
import re
import robin_stocks
from robin_stocks import *
import robin_stocks.robinhood as r
import configur
import time

# Initiliazations for message list and stock dataframe
message_list = {*()}
stocks_df = pd.DataFrame({'stock': ['string stock'],'message':['string message'],'guess':['string guess']})

# Logs into Robinhood for portfolio managing
# print(r.login(username='EMAIL', password=configur.thepass['pass2'], expiresIn = 86400, by_sms= True,store_session=False))

# These are some words for triggering buy/sell
buy_words = {'buy', 'lotto', 'patience', 'patient', 'entering','swing','grabbing','pt','pt1','took','taking','add','adding',
            'hold', 'call','opening','buying','put','p','c','bto','entry','filled','starter','added',}
bought_words = {'tapped','bought','printed','paying', 'up', 'hit', 'profits', 'risky', 'profit','already','printing'}
sell_words = {'take', 'cut', 'out','sold', 'sell', 'short','shorting','closing', 'close','cutting','green','lock',
            'taking','congrats','congratz','trim','trimming','stc','bearish','exit', 'stoploss'}
manage_words = {'risky', 'if', 'looks','look','based'}
holding_words = {'holding','hold','still','swinging'}
BTO_or_STC = ['BTO', 'STC']

# These are all from the discord rooms that we want messages from
message_room_ID_list = ['966301538337312818','966301653416431656','974016202487767110','1010201818464276550','953812898059276369','966301749713444914','966301785142734939','805508737506345000',
                        '797550631074005002','988842693494988830','997513089450778764','1001839099289804961']

# Uses Requests package to get the last 50 messages of a Discord channel
def retrieve_messages(channel_id):
    global message_list
    headers = {'authorization' : "Njc3MzAyOTEyNzAzMjY2ODE3.YfB1YQ.Xd7JAD96Xu-E04sFSuHA7ooWsoc"}
    r = requests.get(f'https://discord.com/api/v9/channels/{channel_id}/messages', headers = headers)

    jsonn = json.loads(r.text)
    jsonn = jsonn[:50]
    
    for value in jsonn:
        # Adds the new text if it is not already in the list
        if (value['content'] not in message_list):
            message_list.add(value['content'])
            print(value['content'])

def robin_stocks_buy(ticker):
    open_positions = r.account.build_holdings()
    # print(open_positions)
    if ticker not in open_positions.keys():
        dictro = r.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=True)
        close = float(dictro[0])
        print (r.orders.order_buy_market(ticker, float("{:.1f}".format(40/close)),timeInForce='gfd'))
        time.sleep(12)

        open_positions = r.account.build_holdings()

        if ticker not in open_positions.keys():
            r.cancel_all_stock_orders()
            robin_stocks_buy(ticker)

def robin_stocks_sell(ticker):
    open_positions = r.account.build_holdings()
    # print(open_positions)
    if ticker in open_positions.keys():
        dictro = r.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=True)
        close = float(dictro[0])
        print (r.orders.order_sell_market(ticker, float("{:.1f}".format(40/close)),timeInForce='gfd'))
        time.sleep(10)

        open_positions = r.account.build_holdings()

        if ticker in open_positions.keys():
            r.cancel_all_stock_orders()
            robin_stocks_sell(ticker)



def update():
    global stocks_df
    global message_list
    # Uses the function defined above to get messages from multiple channels
    for msg in message_room_ID_list:
        try:
            retrieve_messages(msg)
        except Exception as e:
            print(e)

    for i in message_list:
        stock = ''
        bought = False
        holding = False
        date_msg = False
        first = True
        send_text = False
        iterator = len(stocks_df)
        message = i.replace('!','').replace('.','').replace(':','').replace(',','').replace('\n', ' ').replace('stop loss','stoploss').replace('Stop Loss','stoploss').replace('(', '').replace(')', '').split(' ')
        

        if (len(message) <= 50):
            for word in message:
                # Finds the relevant stock by checking if it is the first uppercase word
                if word.isupper() and first and (word not in BTO_or_STC):
                    stock = word
                    first = False
                # Store boolean: is there a numerical date in the message (option date)?
                if ('/' in word) and len(word) < 10:
                    if any(ch.isdigit() for ch in word):
                        date_msg= True
                        bought = True

            if (len(stock) > 0 and stock != 'I'):
                stocks_df.at[iterator,'stock'] = stock
                stocks_df.at[iterator, 'message'] = (' ').join(message)

                message_set = {*()}
                message_set = message_set.union(set(x.lower() for x in message))

                # Handles buy logic - 'bought' means buy words AND not bought words
                if ((message_set) & buy_words) and (not(message_set & bought_words)):
                    bought = True
                # Handles hold logic
                if message_set & holding_words:
                    holding = True
                # Store boolean: is there a date in the message (option date)?
                if (re.findall(r'\s(?:jan|feb|mar\s|apr|may|jun\s|jul|aug|sep\s|oct|nov|dec\s)', ((' ').join(x.lower() for x in message)))):
                    bought = True
                    date_msg = True
                    

                # Handles sell logic. NOTE: bought variable means buy words AND not bought words
                if (message_set & sell_words):
                    if bought:
                        stocks_df.at[iterator, 'guess'] = 'selling and buying?'
                        send_text = True
                    else:
                        if holding:
                            stocks_df.at[iterator, 'guess'] = 'holding, selling?'
                            send_text = True
                        else:
                            if (message_set & bought_words):
                                stocks_df.at[iterator, 'guess'] = 'bought, selling?'
                                send_text = True
                            else:
                                stocks_df.at[iterator, 'guess'] = 'selling'
                                # robin_stocks_sell(stock)
                else:
                    if bought:
                        if date_msg:
                            stocks_df.at[iterator, 'guess'] = 'option date buying'
                            # robin_stocks_buy(stock)
                            if holding:
                                stocks_df.at[iterator, 'guess'] = 'option date holding?'
                                send_text = True
                            else:
                                if (message_set & bought_words):
                                    send_text = True
                                    stocks_df.at[iterator, 'guess'] = 'option date bought?'
                        else:
                            stocks_df.at[iterator, 'guess'] = 'buying'
                            # robin_stocks_buy(stock)

                    else:
                        if holding:
                            stocks_df.at[iterator, 'guess'] = 'holding?'
                            send_text = True
                        else:
                            if (message_set & bought_words):
                                stocks_df.at[iterator, 'guess'] = 'bought'
                                send_text = True
                            else:
                                stocks_df.at[iterator, 'guess'] = 'not buying or selling?'
                                send_text = True

                # More powerful logic at the end can override previous guess
                if('pt' in message_set) or ('pt1' in message_set):
                    if (message_set & bought_words):
                        stocks_df.at[iterator, 'guess'] = 'bought pt?'
                        send_text = True
                        if holding:
                            stocks_df.at[iterator, 'guess'] = 'holding pt?'
                            send_text = True
                    else:
                        if holding:
                            stocks_df.at[iterator, 'guess'] = 'holding or buying pt?'
                            send_text = True
                        else:
                            stocks_df.at[iterator, 'guess'] = 'buying'
                            # robin_stocks_buy(stock)

                if ('bto' in message_set):
                    stocks_df.at[iterator, 'guess'] = 'buying'
                if ('stc' in message_set):
                    stocks_df.at[iterator, 'guess'] = 'selling'
    print(stocks_df)
    # Removes the initilization row and pushes df to excel
    stocks_df = stocks_df.iloc[1: , :]
    stocks_df.to_csv(r"C:\Users\amoog\Desktop\discord_bot\stocks_df.csv") 
         
update()          

