import streamlit as st
import subprocess
import axios 
import requests
import os
import sys

# get the full path of the python executable
python_path = sys.executable

# use the full path in the command variable
command = python_path + ' bot_v7.py'
# use shell=True since this is a string
subprocess.Popen(command, shell=True)


 # Define the API endpoint
url = 'http://127.0.0.1:5000/config';
response = requests.get(url)
data = response.json()
data['ask_trade_size']




# Set up the sidebar
st.sidebar.title('Bot Trading and Backtesting')
st.sidebar.subheader('Enter your input below:')

col1, col2, col3 = st.columns(3)

# Create the input fields and store their values in a dictionary
fields = {}
with col1:
    fields['Field 1'] = st.text_input(label='Exchange', value= data['exchange'])
    fields['Field 2'] = st.text_input(label='Base Symbol', value=data['base_symbol'])
    fields['Field 3'] = st.text_input(label='Quote Symbol', value=data['quote_symbol'])
    fields['Field 4'] = st.text_input(label='Initial Balance Base', value=data['initial_balance_base'])
    fields['Field 5'] = st.text_input(label='Initial Balance Quote', value=data['initial_balance_quote'])
    fields['Field 6'] = st.text_input(label='Ask Trade Size', value=data['ask_trade_size'])
    fields['Field 7'] = st.text_input(label='Bid Trades Size', value=data['bid_trade_size'])
    fields['Field 8'] = st.text_input(label='Max Number of orders/symbol', value=data['max_number_of_orders_per_symbol'])
    fields['Field 9'] = st.text_input(label='Order Fee', value=data['order_fee'])

with col2:
    fields['Field 10'] = st.text_input(label='Buy Order Price offset', value=data['buy_order_price_offset'])
    fields['Field 11'] = st.text_input(label='Sell Order Price Offset',value=data['sell_order_price_offset'])
    fields['Field 12'] = st.text_input(label='Max no of Open Trades for sysmbol', value=data['max_number_of_open_trades_for_symbol'])
    fields['Field 13'] = st.text_input(label='max open order age', value=data['max_open_order_age'])
    fields['Field 14'] = st.text_input(label='From Date', value=data['from_date'])
    fields['Field 15'] = st.text_input(label='To Date', value=data['to_date'])
    fields['Field 16'] = st.text_input(label='Time Out',value=data['timeout'])
    fields['Field 17'] = st.text_input(label='Direction', value=data['direction'])
    fields['Field 18'] = st.text_input(label='Live Mode',value=data['live_mode'])

with col3:
    fields['Field 19'] = st.text_input(label='Api Key', value=data['api_key'])
    fields['Field 20'] = st.text_input(label='Api Secret', value=data['api_secret'])
    fields['Field 21'] = st.text_input(label='Api Password', value=data['api_password'])
    fields['Field 22'] = st.text_input(label='Enable Psar',value=data['enable_psar'])
    fields['Field 23'] = st.text_input(label='minimum pnl', value=data['minimum_pnl'])
    fields['Field 24'] = st.text_input(label='max no of trade for period', value=data['max_number_of_trades_for_period'])
    fields['Field 25'] = st.text_input(label='max no of trade period',value=data['max_number_of_trades_period'])
    fields['field 26'] = st.text_input(label='Trade Channel',value=data['trade_channels'])
    
    

# Add a button that will execute the script with the input field values as arguments

# Add a button that will execute the script with the input field values as arguments
#if st.button('Run Bot'):
 #   arg_list = [f"--{key}='{value}'" for key, value in fields.items()]
 #   command = f"/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 ~/Desktop/ALL/bot_collecton/bot_v7.py {' '.join(arg_list)}"
 #   subprocess.call(command, shell=True)
 #   with open('logs/backtester.log') as f:
 #     log_contents = f.read()
 #  st.text_area('Log', value=log_contents, height=400)
#else:
#   st.write("Click the button to run the bot.")


# Define a function to execute when the button is clicked

def run_bot():
    arg_list = [f"--{key}='{value}'" for key, value in fields.items()]
    command = f"/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 ~/Desktop/ALL/bot_collecton/bot_v7.py {' '.join(arg_list)}"
    subprocess.call(command, shell=True)
    with open('command.log') as f:
        log_contents = f.read()
        st.text_area('Log', value=log_contents, height=400)

# Create a button to run the function
if st.button('Run Bot'):
    run_bot()
    st.write("Bot complete.")
else:
    st.write("Click the button to run the bot.")