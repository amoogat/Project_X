from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
import logging, pytz
import pandas as pd
import numpy as np
from .models import BacktestResult, get_default_strategy, StockData
from .services import parallel_optimize_strategy, GPTTwitter, Backtester
from .serializers import BacktestResultSerializer
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.http import JsonResponse
from datetime import timedelta
from django.utils.decorators import method_decorator

debug_level = 0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def return_stock(message):
    # Gets ticker name from message
    message = message.decode('utf-8')
    if message:
        try:
            parts = message.split()
            if len(parts) > 1:
                stock = parts[1].replace('[','').replace(']','').replace('$', '').replace('\\', '').replace('*','').upper()
                # Further validation to check if the cleaned stock symbol is alphanumeric
                if stock.isalnum():
                    if stock.upper() in ['VIX','VVIX']:
                        stock = 'UVXY'
                    return stock
        except Exception as e:
            logging.error(f"Error processing stock information from message: {message}, error: {str(e)}")
    return None

def return_open_close(message):
    # Gets open or close data from message
    message = message.decode('utf-8')
    if not message:
        return 0
    cleaned_message = message.replace('[','').replace(']','').replace('$', '').replace('\\', '').replace('*','')
    if len(cleaned_message.split()) > 1:
        return 1 if ('Open' in cleaned_message) and ('Long' in cleaned_message) else 0
    
def ensure_timezone(dt, tz):
    # Ensure datetime is timezone-aware
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.tz_localize(tz)
    else:
        dt = dt.tz_convert(tz)
    return dt

def round_to_nearest_minute(dt):
    if dt.second >= 30:
        dt += timedelta(minutes=1)
    return dt.replace(second=0, microsecond=0)

@csrf_exempt
@api_view(['POST'])
def upload_file(request):
    try:
        username = request.data.get('username')
        if not username:
            return Response({'status': 'error', 'message': 'Username is required'}, status=400)
        # Checks if the username already has data in the database - skips process in production
        if debug_level < 2:
            existing_results = BacktestResult.objects.filter(username=username).order_by('ticker', 'created_at').distinct('ticker', 'created_at')
            if existing_results.exists():
                # Serialize and return the existing results
                serializer = BacktestResultSerializer(existing_results, many=True)
                portfolio_chart_data = existing_results.first().portfolio_chart_data
                serializer.data[0].pop('portfolio_chart_data', None)
                return Response({'status': 'success', 'data': serializer.data, 'portfolio_chart_data': portfolio_chart_data}, status=200)

        twitter_processor = GPTTwitter(username)
        try:
            twitter_processor.process_tweets()
        except Exception as e:
            logging.error(str(e))

        if twitter_processor.heisenberg_tweets.empty:
            return Response({'status': 'error', 'message': 'Unable to fetch tweets or no tweets found.'}, status=404)

        param_ranges = {
            'atr_multiplier': np.arange(1, 4, 1),
            'atr_periods': [14, 50, 100, 200, 400, 650]
        }
        
        sys_prompt = """You are parsing tweets to interpret and synthesize information about stock plays. Reference examples as a guide to understand the format of the output. If the text and image description Ticker differ, go with the text, unless there is no ticker mentioned in the text.
        Example:
        Text: $PARA Closed\n\nIn 13.11 (yesterday)\n\nOut 13.24\n\n+1%\n+$65 profit\n\nJust trying to reduce long exposure heading into tomorrow. 
        https://t.co/GpCKwDrfky 
        TRANSCRIBED IMAGE DATA: This image describes a stock sale transaction, not an option play like a call or a put. 
        Specifically, it details the sale of 500 shares of PARA (which is the ticker symbol for a stock) at an average fill price of $13.2401. 
        It's a limit order set to sell at $13.24. The negative quantity (-500) indicates that shares are being sold rather than purchased.
        Correct Output: [Close PARA Long]

        Text: If $META closes above 450, I will do 1,000 jumping jacks.\n\n
        TRANSCRIBED IMAGE DATA: None
        Correct Output: [Neither]

        Text: \\$GOOG Closed\n\nIn .32 credit (Apr 30th)\n\nOut .05 debit\n\n+\\$135 profit\n\nSmall. Got dangerously close. But never worried. TRANSCRIBED IMAGE DATA: The image describes a vertical options spread for Google (GOOG) stock, specifically a put spread. Here are the details:\n\n- **Type of Spread**: Vertical Put Spread\n- **Underlying Asset**: GOOG (Google)\n- **Expiry Date**: May 3, 2024\n- **Strike Prices**: 165 (bought) and 162.5 (sold)\n\nDetails of the trade:\n- **Quantity**: 5 contracts\n- **Total Cost**: Since you are paying 0.05 per contract for 5 contracts, the total debit is 5 * 0.05 = \\$25.\n\nIn this trade, you are buying 5 put options at a strike price of 165 and selling 5 put options at a strike price of 162.5 for the same expiration date of May 3, 2024. This creates a vertical spread (specifically a bear put spread), aiming to profit from a decrease in the price of the underlying stock, Google, down to or below the lower strike price of 162.5.
        Correct Output: [Close GOOG Short]

        Text: \\$AMD Eyeing this name.\n\nIf it can get to 137-138ish tomorrow or next week, then I would buy there. \n\nWhy? 200dma magnet could act as major support.  TRANSCRIBED IMAGE DATA: None
        Correct Output: [Neither]

        Text: \\$DIS Open\n\nEarnings play\n\nRisk \\$1,670 to make \\$330.\n\nBmo, implied about a 7pt move. https://t.co/nyMZGS23OZ  TRANSCRIBED IMAGE DATA: The image describes a vertical put spread (also known as a bull put spread). This is a type of options strategy that involves selling one put option and buying another put option at a lower strike price but with the same expiration date.\n\nHere's the breakdown of the trade based on the image:\n\n- This is for the stock with the ticker symbol "DIS" (Walt Disney Company), with options expiring on May 10, 2024.\n- The vertical spread involves the 108 and 106 strike prices, meaning you are dealing with 108/106 put options.\n- The strategy specified is a put vertical spread:\n  - Selling 10 put options at the 108 strike price.\n  - Buying 10 put options at the 106 strike price.\n- The prices for the transactions are:\n  - Sold (shorted) the 108 strike put options at \\$0.86 each.\n  - Bought (long) the 106 strike put options at \\$0.53 each.\n- The net credit received for the spread is \\$0.33 per share (since options typically represent 100 shares, the total net credit is \\$33 per contract).\n\nIn summary, the strategy is a bull put spread where you hope the price of Disney (DIS) stays above the higher strike price (108) by the expiration date so that both options expire worthless, and you keep the credit

        Text: Watchlist for next week:\n\nLong: \\$VIX\n\nShort: \\$FXI, \\$SPY\n\nNeutral: \\$NVDA\n\nSpeculative bounce play on watch: \\$SHOP  TRANSCRIBED IMAGE DATA: None
        Correct Output: [Neither]

        Text: Two Biggest Losers this week:\n\nSPY call credit spread: -\\$5,130\n\nDIS put credit spread: -\\$1,670\n\nOuch.\n\nAccount Balance: \\$114,888.39.\n\n-\\$5,097.20 week over week.\n\nNot happy, but we look forward to next week. https://t.co/X67fSZOUY6  TRANSCRIBED IMAGE DATA: None
        Correct Output: [Neither]

        Text: \\\\$FXI Open\\n\\nJust some yolo puts here looking for a healthy small pullback.\\n\\nIndex up 10 of last 11 days. https://t.co/PWuTtGhm5t  TRANSCRIBED IMAGE DATA: The image describes the purchase of a put option. Specifically, the details are as follows:\\n\\n- Ticker Symbol: FXI 100 (This commonly refers to an exchange-traded fund (ETF) that tracks the performance of the top 100 Chinese companies).\\n- Weekly options expiring on: May 10th, 2024.\\n- Strike Price: 27.5.\\n- Type: Put option (designated by the "P").\\n- Quantity: 15 contracts.\\n- Price: \\\\$0.44 per contract (with a limit order as indicated by "LMT").\\n- Trade fill date and time: May 6th, 2024, at 6:32 AM.\\n\\nThis indicates that the trader bought 15 put options at a strike price of 27.5, and they paid \\\\$0.44 per option contract.
        Correct Output: [Open FXI Short]

        Text: Hang Seng going bonkers overnight.\\nIf it sticks by US open, these FXI calls should be close to 100%!  TRANSCRIBED IMAGE DATA: None
        Correct Output: [Neither]

        Text: \\\\$GME selling call spreads on watch.  TRANSCRIBED IMAGE DATA: None
        Correct Output: [Neither]

        Text: For those who want to play the VIX but cannot because cough Robinhood cough, then I suggest SPY puts. It's pretty much the next best alternative.\\n\\nDon't give me that UVXY crap imo.  TRANSCRIBED IMAGE DATA: None
        Correct Output: [Neither]
        response: FXI Open

        Text: FXI Open Would not be surprised we get a decent bounce in the Hang Seng tonight and/or tomorrow night. https://t.co/voe0o6sdyQ TRANSCRIBED IMAGE DATA: The image describes the sale of 25 call options for the VIX with a strike price of 12.5 expiring on August 21 2024. It shows that 25 call options were sold (as indicated by the -25) at a price of \$3.35 each. This is evident from the designation C in the options contract which stands for Call.
        Corect Output: [FXI Open Long]
        """
        user_prompt = "Is this tweet referring to the opening or closing of a stock position? If it is, please also list the corresponding ticker and whether it is long or short. If it is not referring to the opening or closing of a position, simply put neither. Please respond in the possible formats: [Open/Close TICKER Long/Short] or [Neither]. If the tweet refers to multiple positions, list them all in a comma separated list."
        
        # Dynamically prompt openAI and retrieve stock trades synthesis
        twitter_processor.dynamic_prompt_and_save(sys_prompt, user_prompt)
        df = twitter_processor.heisenberg_tweets.copy()
        df['ticker'] = df['result'].apply(return_stock)
        df['buy'] = df['result'].apply(return_open_close)
        df = df.loc[(df['ticker'].notnull()) & (df['buy'] == 1)]
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
        
        # Optimizes strategy so we can backtest username
        results = parallel_optimize_strategy(df, param_ranges)
        best_results = results.loc[results.groupby(['ticker', 'created_at'])['total_return'].idxmax()]
        best_results = best_results.sort_values(by='total_return', ascending=False)
        results_list = best_results.to_dict('records')
        default_strategy_id = get_default_strategy()
        best_results_sorted_by_date = best_results.sort_values(by='created_at')

        # Whole market backtest, needs first bought and last sold timestamps
        tz = pytz.timezone('America/New_York')
        portfolio_backtester = Backtester()
        first_bought_at = best_results_sorted_by_date['created_at'].min()
        last_sold_at = best_results_sorted_by_date['sold_at_date'].max()
        first_bought_at = round_to_nearest_minute(ensure_timezone(first_bought_at, tz))
        last_sold_at = round_to_nearest_minute(ensure_timezone(last_sold_at, tz))

        portfolio_backtester.set_first_bought_at(first_bought_at)
        portfolio_backtester.set_last_sold_at(last_sold_at)
        portfolio_backtester.initialize_portfolio()

        # Loops through each trade and fetches stock data during this time period
        market_data_cache = {}
        for result in best_results_sorted_by_date.itertuples():
            ticker = result.ticker
            start_date = round_to_nearest_minute(ensure_timezone(result.created_at, tz))
            end_date = round_to_nearest_minute(ensure_timezone(result.sold_at_date, tz))
            if debug_level > 0:
                logging.info(f"Processing trade for {ticker} from {start_date} to {end_date}")
            cache_key = f"{ticker}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
            if cache_key not in market_data_cache:
                market_data = portfolio_backtester.fetch_market_data(ticker, start_date, end_date)
                market_data_cache[cache_key] = market_data
            if market_data_cache[cache_key] is not None:
                trades = [{
                    'entry_date': start_date,
                    'exit_date': end_date,
                }]
                portfolio_backtester.evaluate_all_trades(trades, market_data_cache[cache_key])

        # Finalizes and exports portfolio data for chart view
        portfolio_backtester.finalize_portfolio()
        portfolio_df = portfolio_backtester.portfolio.reset_index()
        portfolio_df.columns = ['date', 'value']
        portfolio_chart_data = { 'dates': portfolio_df['date'].astype(str).tolist(),
                                'values': portfolio_df['value'].tolist() }
        
         # Individual trades serialized for chart view
        first_result_saved = False
        for result in results_list:
            result_data = {
                'username': username, 
                'strategy': default_strategy_id,
                'ticker': result.get('ticker', ''),
                'created_at': result.get('created_at'),
                'atr_multiplier': result.get('atr_multiplier', 0.0),
                'atr_period': result.get('atr_period', 0),
                'total_return':  result.get('total_return',0),
                'portfolio_variance': result.get('portfolio_variance', 0.0),
                'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                'maximum_drawdown': result.get('maximum_drawdown', 0.0),
                'minutes_taken': result.get('minutes_taken', 0),
                'sold_at_date':result.get('sold_at_date'),
                'score': result.get('score', 0.0)
                }
            if not first_result_saved:
                result_data.update({'portfolio_chart_data': portfolio_chart_data})
                first_result_saved = True
            serializer = BacktestResultSerializer(data=result_data)
            if serializer.is_valid():
                serializer.save()
            else:
                logging.error(f"Serializer error: {str(serializer.errors)}")
                return Response(serializer.errors, status=400)
        return Response({'status': 'success', 'data': results_list, 'portfolio_chart_data': portfolio_chart_data}, status=201)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return Response({'status': 'error', 'message': 'An internal error occurred.'})

@api_view(['GET'])
def results_view(request):
    results = BacktestResult.objects.all()
    return render(request, 'results.html', {'results': results})

@csrf_exempt  # Consider CSRF implications
def upload_form_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        return redirect('success_url')  # Redirect after POST
    return render(request, 'upload.html')

@method_decorator(csrf_exempt, name='dispatch')
class StockDataView(APIView):
    # Will display chart on click for individual stock plays
    def get(self, request, ticker):
        if debug_level > 0:
            logging.info(f"Received request for ticker: {ticker}")

        stock_data = StockData.objects.filter(ticker=ticker).order_by('date')
        dates, prices = [], []
        for data in stock_data:
            dates.append(data.date)
            prices.append(float(data.close))
            tweet_text = data.tweet_text.decode('utf-8') if isinstance(data.tweet_text, bytes) else data.tweet_text
        response_data = {
            'ticker': ticker,
            'tweet_text': tweet_text.replace('\\n', '\n').replace('\n', ' '),
            'chartData': {
                'dates': dates,
                'prices': prices
            }
        }
        return JsonResponse(response_data)