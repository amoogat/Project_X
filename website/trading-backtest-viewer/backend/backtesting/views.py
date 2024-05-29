from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os, logging
import numpy as np
from .models import BacktestResult
from .services import parallel_optimize_strategy, GPTTwitter 

def return_stock(message):
    if message and len(message.split()) > 1:
        return message.split()[1].replace('[','').replace(']','').replace('$', '').replace('\\', '').replace('*','').upper()
    return None

def return_open_close(message):
    if not message:
        return 0
    cleaned_message =  message.replace('[','').replace(']','')
    if len(cleaned_message.split()) > 1:
        return 1 if ('Open' in cleaned_message) and ('Long' in cleaned_message) else 0
    
@api_view(['GET', 'POST'])
def upload_file(request):
    try:
        if request.method == 'POST':
            username = request.data.get('username')
            if not username:
                return Response({'status': 'error', 'message': 'Username is required'})

            twitter_processor = GPTTwitter(username)
            twitter_processor.process_tweets()  # Process and prepare tweets

            if twitter_processor.heisenberg_tweets.empty:
                return Response({'status': 'error', 'message': 'Unable to fetch tweets or no tweets found.'})

            param_ranges = {
                'atr_multiplier': np.arange(1, 3.5, 0.5),
                'trailing_stop_loss_multiplier': np.arange(1, 3.5, 0.5),
                'atr_periods': [14, 50, 100, 200, 400, 650]
            }
            sys_prompt = """You are parsing tweets to interpret and synthesize information about stock plays. Reference examples as a guide to understand the format of the output.
            Example:
            Text: $PARA Closed\n\nIn 13.11 (yesterday)\n\nOut 13.24\n\n+1%\n+$65 profit\n\nJust trying to reduce long exposure heading into tomorrow. 
            https://t.co/GpCKwDrfky 
            TRANSCRIBED IMAGE DATA: This image describes a stock sale transaction, not an option play like a call or a put. 
            Specifically, it details the sale of 500 shares of PARA (which is the ticker symbol for a stock) at an average fill price of $13.2401. 
            It's a limit order set to sell at $13.24. The negative quantity (-500) indicates that shares are being sold rather than purchased.
            Correct Output: [Close PARA Long]

            Text: If $META closes above 450, I will do 1,000 jumping jacks.\n\n🙏
            TRANSCRIBED IMAGE DATA: None
            Correct Output: [Neither]

            Text: \\$GOOG Closed\n\nIn .32 credit (Apr 30th)\n\nOut .05 debit\n\n+\\$135 profit\n\nSmall. Got dangerously close. But never worried. TRANSCRIBED IMAGE DATA: The image describes a vertical options spread for Google (GOOG) stock, specifically a put spread. Here are the details:\n\n- **Type of Spread**: Vertical Put Spread\n- **Underlying Asset**: GOOG (Google)\n- **Expiry Date**: May 3, 2024\n- **Strike Prices**: 165 (bought) and 162.5 (sold)\n\nDetails of the trade:\n- **Quantity**: 5 contracts\n- **Total Cost**: Since you are paying 0.05 per contract for 5 contracts, the total debit is 5 * 0.05 = \\$25.\n\nIn this trade, you are buying 5 put options at a strike price of 165 and selling 5 put options at a strike price of 162.5 for the same expiration date of May 3, 2024. This creates a vertical spread (specifically a bear put spread), aiming to profit from a decrease in the price of the underlying stock, Google, down to or below the lower strike price of 162.5.
            Correct Output: [Close GOOG Short]

            Text: \\$AMD Eyeing this name.\n\nIf it can get to 137-138ish tomorrow or next week, then I would buy there. \n\nWhy? 200dma magnet could act as major support.  TRANSCRIBED IMAGE DATA: None
            Correct Output: [Neither]

            Text: \\$DIS Open\n\nEarnings play\n\nRisk \\$1,670 to make \\$330.\n\nBmo, implied about a 7pt move. https://t.co/nyMZGS23OZ  TRANSCRIBED IMAGE DATA: The image describes a vertical put spread (also known as a bull put spread). This is a type of options strategy that involves selling one put option and buying another put option at a lower strike price but with the same expiration date.\n\nHere's the breakdown of the trade based on the image:\n\n- This is for the stock with the ticker symbol "DIS" (Walt Disney Company), with options expiring on May 10, 2024.\n- The vertical spread involves the 108 and 106 strike prices, meaning you are dealing with 108/106 put options.\n- The strategy specified is a put vertical spread:\n  - Selling 10 put options at the 108 strike price.\n  - Buying 10 put options at the 106 strike price.\n- The prices for the transactions are:\n  - Sold (shorted) the 108 strike put options at \\$0.86 each.\n  - Bought (long) the 106 strike put options at \\$0.53 each.\n- The net credit received for the spread is \\$0.33 per share (since options typically represent 100 shares, the total net credit is \\$33 per contract).\n\nIn summary, the strategy is a bull put spread where you hope the price of Disney (DIS) stays above the higher strike price (108) by the expiration date so that both options expire worthless, and you keep the credit received upfront.
            Correct Output: [Open DIS Long]

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
            """
            user_prompt = "Is this tweet referring to the opening or closing of a stock position? If it is, please also list the corresponding ticker and whether it is long or short. If it is not referring to the opening or closing of a position, simply put neither. Please respond in the possible formats: [Open/Close TICKER Long/Short] or [Neither]. If the tweet refers to multiple positions, list them all in a comma separated list."
            
            twitter_processor.dynamic_prompt_and_save(sys_prompt, user_prompt)
            
            df = twitter_processor.ht_dynamic.copy()
            df['Ticker'] = df['result'].apply(return_stock)
            df['Buy'] = df['result'].apply(return_open_close)
            df = df[(df['Ticker'].notnull()) & (df['Buy'] == 1)]
            
            results = parallel_optimize_strategy(df, param_ranges)
            best_results = results.loc[results.groupby(['ticker', 'created_at'])['total_return'].idxmax()]
            best_results = best_results.sort_values(by='final_equity', ascending=False)
            results_list = best_results.to_dict('records')
            for _, result in best_results.iterrows():
                BacktestResult.objects.create(
                    ticker=result['ticker'],
                    created_at=result['created_at'],
                    atr_multiplier=result['atr_multiplier'],
                    trailing_stop_multiplier=result['trailing_stop_multiplier'],
                    atr_period=result['atr_period'],
                    total_return=result['total_return'],
                    portfolio_variance=result['portfolio_variance'],
                    sharpe_ratio=result['sharpe_ratio'],
                    final_equity=result['final_equity'],
                    maximum_drawdown=result['maximum_drawdown'],
                    successful_trades=result['successful_trades'],
                    minutes_taken=result['minutes_taken'],
                    score=result['score']
                )

            return render(request, 'results.html', {'results': results_list})
        else:
            return render(request, 'upload.html')
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return Response({'status': 'error', 'message': 'An internal error occurred.'})

@api_view(['GET'])
def results_view(request):
    results = BacktestResult.objects.all()
    return render(request, 'results.html', {'results': results})
