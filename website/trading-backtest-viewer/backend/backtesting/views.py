from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
import logging, pytz, json
import pandas as pd
import numpy as np
from .models import BacktestResult, get_default_strategy, StockData
from .services import parallel_optimize_strategy, GPTTwitter 
from .serializers import BacktestResultSerializer
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.db import transaction
from datetime import datetime, timedelta
from django.utils.decorators import method_decorator
from concurrent.futures import ThreadPoolExecutor

debug_mode = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def return_stock(message):
    if message:
        try:
            # Basic validation to check if the message format is as expected
            parts = message.split()
            if len(parts) > 1:
                # Cleaning up the stock ticker symbol
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
    if not message:
        return 0
    cleaned_message = message.replace('[','').replace(']','').replace('$', '').replace('\\', '').replace('*','')
    if len(cleaned_message.split()) > 1:
        return 1 if ('Open' in cleaned_message) and ('Long' in cleaned_message) else 0

@csrf_exempt
@api_view(['POST'])
def upload_file(request):
    try:
        username = request.data.get('username')
        if not username:
            return Response({'status': 'error', 'message': 'Username is required'}, status=400)
        
        if not debug_mode:
            # Check if the username already has data in the database
            existing_results = BacktestResult.objects.filter(username=username).distinct('ticker', 'created_at')
            if existing_results.exists():
                # Serialize and return the existing results
                serializer = BacktestResultSerializer(existing_results, many=True)
                return Response({'status': 'success', 'data': serializer.data}, status=200)

        twitter_processor = GPTTwitter(username)
        try:
            twitter_processor.process_tweets()
        except Exception as e:
            logging.error(str(e))
        finally:
            twitter_processor.close_drivers()

        if twitter_processor.heisenberg_tweets.empty:
            return Response({'status': 'error', 'message': 'Unable to fetch tweets or no tweets found.'}, status=404)

        param_ranges = {
            'atr_multiplier': np.arange(1, 3.5, 0.5),
            'trailing_stop_loss_multiplier': np.arange(1, 3.5, 0.5),
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
        
        twitter_processor.dynamic_prompt_and_save(sys_prompt, user_prompt)
        
        df = twitter_processor.ht_dynamic.copy()
        df['ticker'] = df['result'].apply(return_stock)
        df['buy'] = df['result'].apply(return_open_close)
        df = df.loc[(df['ticker'].notnull()) & (df['buy'] == 1)]

        # Convert 'created_at' to datetime and remove timezone
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
        
        results = parallel_optimize_strategy(df, param_ranges)
        best_results = results.loc[results.groupby(['ticker', 'created_at'])['total_return'].idxmax()]
        best_results = best_results.sort_values(by='final_equity', ascending=False)
        results_list = best_results.to_dict('records')
        default_strategy_id = get_default_strategy()

        for result in results_list:
            result_data = {
                'username': username, 
                'strategy': default_strategy_id,
                'ticker': result.get('ticker', ''),
                'created_at': result.get('created_at'),
                'atr_multiplier': result.get('atr_multiplier', 0.0),
                'trailing_stop_multiplier': result.get('trailing_stop_multiplier', 0.0),
                'atr_period': result.get('atr_period', 0),
                'total_return': result.get('total_return', 0.0),
                'portfolio_variance': result.get('portfolio_variance', 0.0),
                'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                'final_equity': result.get('final_equity', 0.0),
                'maximum_drawdown': result.get('maximum_drawdown', 0.0),
                'successful_trades': result.get('successful_trades', 0),
                'minutes_taken': result.get('minutes_taken', 0),
                'sold_at_date' : result.get('sold_at_date'),
                'score': result.get('score', 0.0)
            }

            serializer = BacktestResultSerializer(data=result_data)
            if serializer.is_valid():
                serializer.save()
            else:
                logging.error(f"Serializer error: {str(serializer.errors)}")
                return Response(serializer.errors, status=400)

        return Response({'status': 'success', 'data': results_list}, status=201)
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
        # Process the username, possibly calling an external API or service
        return redirect('success_url')  # Redirect after POST
    return render(request, 'upload.html')


@method_decorator(csrf_exempt, name='dispatch')
class StockDataView(APIView):
    def get(self, request, ticker):
        if debug_mode:
            logging.info(f"Received request for ticker: {ticker}")

        stock_data = StockData.objects.filter(ticker=ticker).order_by('date')
        dates = [(data.date) for data in stock_data]
        prices = [float(data.close) for data in stock_data]

        response_data = {
            'ticker': ticker,
            'chartData': {
                'dates': dates,
                'prices': prices,
            }
        }

        return JsonResponse(response_data)

@csrf_exempt
def batch_upload(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ticker = data.get('ticker')
            stock_data = data.get('stock_data')

            def save_batch(batch):
                with transaction.atomic():
                    for entry in batch:
                        date = (entry.get('date'))
                        if isinstance(date, str):
                            date = datetime.fromisoformat(date)
                        if not date.tzinfo:
                            date = pytz.UTC.localize(date)
                        date = (date - timedelta(hours=4)).strftime('%m-%d %I:%M:%S %p')
                        StockData.objects.update_or_create(
                            ticker=ticker,
                            date=date,
                            defaults={
                                'close': entry.get('close', 0.0)
                            }
                        )

            BATCH_SIZE = 1000
            batches = [stock_data[i:i + BATCH_SIZE] for i in range(0, len(stock_data), BATCH_SIZE)]

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.map(save_batch, batches)

            return JsonResponse({'status': 'success', 'message': 'Batch upload successful'}, status=201)
        except Exception as e:
            logging.error(f"Failed to save batch stock data: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)