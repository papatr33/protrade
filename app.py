import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import re
from datetime import datetime, timedelta
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="ProTrade Portfolio Engine", layout="wide", initial_sidebar_state="expanded")

# --- CONSTANTS ---
IGNORED_TICKERS = ['002594.SZ', '1211.HK']

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { color: #8b949e; font-size: 14px; margin-bottom: 5px; }
    .metric-value { color: #f0f6fc; font-size: 24px; font-weight: bold; }
    .metric-green { color: #3fb950; }
    .metric-red { color: #f85149; }
    .stButton button { width: 100%; background-color: #238636; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 1. PARSING & UTILS ---

def parse_ticker_info(bb_key):
    if pd.isna(bb_key): return None, None, 'USD'
    bb_key = str(bb_key).strip()
    is_option = any(x in bb_key for x in [' C ', ' P ', ' Call ', ' Put ']) or len(bb_key.split()) > 3
    asset_type = 'Option' if is_option else 'Stock'
    equity_pattern = r"^(\w+)\s+(\w+)\s+Equity"
    match = re.match(equity_pattern, bb_key)
    
    yf_ticker = None
    currency = 'USD' 
    if match:
        symbol, country = match.groups()
        if country == 'US': yf_ticker = symbol; currency = 'USD'
        elif country == 'HK': yf_ticker = f"{symbol.zfill(4)}.HK"; currency = 'HKD'
        elif country == 'CH': yf_ticker = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"; currency = 'CNY'
        elif country == 'JP': yf_ticker = f"{symbol}.T"; currency = 'JPY'
        elif country == 'LN': yf_ticker = f"{symbol}.L"; currency = 'GBP'
    
    if is_option and not yf_ticker:
        parts = bb_key.split()
        raw_sym = parts[0]
        if ' HK ' in bb_key: yf_ticker = f"{raw_sym.zfill(4)}.HK"; currency = 'HKD'
        elif ' CH ' in bb_key: yf_ticker = f"{raw_sym}.SS" if raw_sym.startswith('6') else f"{raw_sym}.SZ"; currency = 'CNY'
        else: yf_ticker = raw_sym; currency = 'USD'

    return yf_ticker, asset_type, currency

def parse_option_details(bb_key):
    if pd.isna(bb_key): return None, None, None
    bb_key = str(bb_key)
    std_pattern = r"(\d{2}/\d{2}/\d{2})\s+(C|P)(\d+\.?\d*)"
    match1 = re.search(std_pattern, bb_key)
    if match1:
        date_str, type_code, strike_str = match1.groups()
        try:
            return datetime.strptime(date_str, "%m/%d/%y"), float(strike_str), 'Call' if type_code == 'C' else 'Put'
        except: pass
    otc_pattern = r"(\d{4}-\d{2}-\d{2})\s+(\d+\.?\d*)\s+(Call|Put)"
    match2 = re.search(otc_pattern, bb_key, re.IGNORECASE)
    if match2:
        date_str, strike_str, type_str = match2.groups()
        try:
            return datetime.strptime(date_str, "%Y-%m-%d"), float(strike_str), type_str.capitalize()
        except: pass
    return None, None, None

def normalize_fx_pair(currency):
    if currency == 'USD': return None
    map_fx = {'HKD': 'HKD=X', 'CNY': 'CNY=X', 'JPY': 'JPY=X', 'GBP': 'GBPUSD=X', 'EUR': 'EURUSD=X'}
    return map_fx.get(currency)

def get_metric_html(label, value, is_currency=True, is_delta=False):
    val_str = f"${value:,.0f}" if is_currency else f"{value:,.0f}"
    color_class = ""
    if is_delta or is_currency:
        color_class = "metric-green" if value >= 0 else "metric-red"
    return f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{val_str}</div>
    </div>
    """

# --- 2. DATA LOADING ---

@st.cache_data
def load_and_process_data(soy_file, trade_file):
    soy = pd.read_csv(soy_file)
    trades = pd.read_csv(trade_file)

    soy['YF_Ticker'], soy['Type'], soy['Currency'] = zip(*soy['BB Yellow Key New'].apply(parse_ticker_info))
    trades['YF_Ticker'], trades['Type'], trades['Currency'] = zip(*trades['BB Yellow Key'].apply(parse_ticker_info))
    trades['Opt_Expiry'], trades['Opt_Strike'], trades['Opt_Type'] = zip(*trades['BB Yellow Key'].apply(parse_option_details))
    
    soy = soy[~soy['YF_Ticker'].isin(IGNORED_TICKERS)]
    trades = trades[~trades['YF_Ticker'].isin(IGNORED_TICKERS)]
    
    soy['Quantity'] = pd.to_numeric(soy['Quantity'], errors='coerce').fillna(0)
    soy['Start_USD_Value'] = pd.to_numeric(soy['$ NMV'], errors='coerce').fillna(0)

    trades['Trade Date'] = pd.to_datetime(trades['Trade Date'])
    trades['Qty_Signed'] = pd.to_numeric(trades.get('Notional Quantity', 0), errors='coerce').fillna(0)
    abs_proceeds = pd.to_numeric(trades.get('$ Absolute Trading Notional Net Proceeds', 0), errors='coerce').fillna(0)
    
    def get_cashflow_sign(row):
        txn = str(row['Txn Type']).lower()
        if 'buy' in txn or 'cover' in txn: return -1.0
        if 'sell' in txn or 'short' in txn: return 1.0
        return 0.0

    trades['USD_Cashflow'] = abs_proceeds * trades.apply(get_cashflow_sign, axis=1)
    trades['Price_Local'] = pd.to_numeric(trades['Trade Price'], errors='coerce').fillna(0)
    
    return soy, trades

@st.cache_data
def fetch_market_data(ticker, currency):
    start_date = "2024-12-20"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Fetch Stock Data
        stock_raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if stock_raw.empty: return None, None, None, None
        if isinstance(stock_raw.columns, pd.MultiIndex): stock_raw.columns = stock_raw.columns.get_level_values(0)
        
        valid_dates = stock_raw.index
        
        # Fetch Company Name via Ticker API
        try:
            t_info = yf.Ticker(ticker).info
            company_name = t_info.get('shortName') or t_info.get('longName') or ticker
        except:
            company_name = ticker

        # Reindex for PnL Calculation
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        stock_filled = stock_raw.reindex(all_dates).ffill().bfill() 

        fx_filled = None
        fx_ticker = normalize_fx_pair(currency)
        if fx_ticker:
            fx_raw = yf.download(fx_ticker, start=start_date, end=end_date, progress=False)
            if isinstance(fx_raw.columns, pd.MultiIndex): fx_raw.columns = fx_raw.columns.get_level_values(0)
            fx_filled = fx_raw['Close'].reindex(all_dates).ffill().bfill()
            
        return stock_filled, fx_filled, valid_dates, company_name
    except:
        return None, None, None, None

# --- 3. CORE CALCULATION ---

def calculate_ticker_metrics(ticker, soy, trades, stock_df, fx_df, company_name):
    t_soy = soy[soy['YF_Ticker'] == ticker]
    t_trades = trades[trades['YF_Ticker'] == ticker]
    currency = t_trades['Currency'].iloc[0] if not t_trades.empty else t_soy['Currency'].iloc[0]

    dates = stock_df.index
    analysis = pd.DataFrame(index=dates)
    analysis['Close_Local'] = stock_df['Close']
    analysis['Open'] = stock_df['Open']
    analysis['High'] = stock_df['High']
    analysis['Low'] = stock_df['Low']
    
    if currency == 'GBP':
        analysis['Close_Local'] = analysis['Close_Local'] / 100.0
        analysis['Open'] = analysis['Open'] / 100.0
        analysis['High'] = analysis['High'] / 100.0
        analysis['Low'] = analysis['Low'] / 100.0
    
    if fx_df is not None:
        analysis['FX'] = fx_df.reindex(analysis.index).ffill()
    else:
        analysis['FX'] = 1.0
        
    if currency in ['HKD', 'CNY', 'JPY']: analysis['Close_USD'] = analysis['Close_Local'] / analysis['FX']
    elif currency in ['GBP', 'EUR']: analysis['Close_USD'] = analysis['Close_Local'] * analysis['FX']
    else: analysis['Close_USD'] = analysis['Close_Local']
    
    analysis['Close_USD'] = analysis['Close_USD'].bfill()

    # --- Stock PnL ---
    stock_soy = t_soy[t_soy['Type'] == 'Stock']
    stock_trades = t_trades[t_trades['Type'] == 'Stock']
    
    init_stock_qty = stock_soy['Quantity'].sum()
    first_valid_price = analysis['Close_USD'].iloc[0]
    init_simulated_cost = init_stock_qty * first_valid_price
    
    current_qty = init_stock_qty
    current_cash = -init_simulated_cost 
    
    daily_activity = pd.DataFrame()
    if not stock_trades.empty:
        daily_activity = stock_trades.groupby(stock_trades['Trade Date'].dt.date)[['Qty_Signed', 'USD_Cashflow']].sum()

    pos_list = []
    cash_list = []
    
    for d in analysis.index:
        d_date = d.date()
        if d_date in daily_activity.index:
            current_qty += daily_activity.loc[d_date, 'Qty_Signed']
            current_cash += daily_activity.loc[d_date, 'USD_Cashflow']
        pos_list.append(current_qty)
        cash_list.append(current_cash)
        
    analysis['Stock_Pos'] = pos_list
    analysis['Stock_Cashflow_Accum'] = cash_list
    analysis['Stock_Mkt_Value_USD'] = analysis['Stock_Pos'] * analysis['Close_USD']
    analysis['Stock_PnL_Cumulative'] = analysis['Stock_Mkt_Value_USD'] + analysis['Stock_Cashflow_Accum']
    
    # --- Option PnL ---
    opt_soy = t_soy[t_soy['Type'] == 'Option']
    opt_trades = t_trades[t_trades['Type'] == 'Option']
    opt_current_cash = -opt_soy['Start_USD_Value'].sum()
    
    opt_activity = pd.DataFrame()
    if not opt_trades.empty:
        opt_activity = opt_trades.groupby(opt_trades['Trade Date'].dt.date)[['USD_Cashflow']].sum()
        
    opt_cash_list = []
    for d in analysis.index:
        d_date = d.date()
        if d_date in opt_activity.index:
            opt_current_cash += opt_activity.loc[d_date, 'USD_Cashflow']
        opt_cash_list.append(opt_current_cash)
        
    analysis['Option_PnL_Cumulative'] = opt_cash_list 
    analysis['Total_PnL_Cumulative'] = analysis['Stock_PnL_Cumulative'] + analysis['Option_PnL_Cumulative']

    last_row = analysis.iloc[-1]
    
    summary = {
        'Ticker': ticker,
        'Name': company_name, # Uses the yfinance fetched name
        'Total PnL': last_row['Total_PnL_Cumulative'],
        'Stock PnL': last_row['Stock_PnL_Cumulative'],
        'Option PnL': last_row['Option_PnL_Cumulative'],
        'Position': last_row['Stock_Pos'],
        'Price': last_row['Close_Local'],
        'Currency': currency
    }
    
    analysis['Ticker'] = ticker
    return analysis, stock_trades, opt_trades, summary

# --- 4. PLOTTING ---

def plot_ticker_deep_dive(ticker, df_analysis, stock_trades, opt_trades, valid_dates):
    df_view = df_analysis[df_analysis.index >= '2025-01-01']
    df_price_plot = df_view.loc[df_view.index.intersection(valid_dates)]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price & Executions", "Total Cumulative PnL (USD)", "PnL Breakdown (USD)")
    )

    fig.add_trace(go.Candlestick(
        x=df_price_plot.index, 
        open=df_price_plot['Open'], high=df_price_plot['High'],
        low=df_price_plot['Low'], close=df_price_plot['Close_Local'], 
        name='Price', hoverinfo='x+y'
    ), row=1, col=1)

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    if not stock_trades.empty:
        buys = stock_trades[stock_trades['Qty_Signed'] > 0]
        sells = stock_trades[stock_trades['Qty_Signed'] < 0]
        
        t_curr = stock_trades['Currency'].iloc[0] if 'Currency' in stock_trades.columns else 'USD'
        price_mult = 0.01 if t_curr == 'GBP' else 1.0
        
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys['Trade Date'], y=buys['Price_Local'] * price_mult, mode='markers', 
                marker=dict(symbol='triangle-up', color='#00e676', size=14, line=dict(width=1.5, color='black')),
                name='Stock Buy', hovertemplate="Buy<br>Qty: %{customdata[0]}<br>Val: $%{customdata[1]:,.0f}",
                customdata=np.stack((buys['Qty_Signed'], buys['USD_Cashflow']), axis=-1)), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells['Trade Date'], y=sells['Price_Local'] * price_mult, mode='markers', 
                marker=dict(symbol='triangle-down', color='#ff1744', size=14, line=dict(width=1.5, color='black')),
                name='Stock Sell', hovertemplate="Sell<br>Qty: %{customdata[0]}<br>Val: $%{customdata[1]:,.0f}",
                customdata=np.stack((sells['Qty_Signed'], sells['USD_Cashflow']), axis=-1)), row=1, col=1)

    if not opt_trades.empty:
        for _, row in opt_trades.iterrows():
            if pd.notna(row['Opt_Strike']) and pd.notna(row['Opt_Expiry']):
                line_color = 'rgba(0, 230, 118, 0.6)' if row['Opt_Type'] == 'Call' else 'rgba(255, 23, 68, 0.6)'
                fig.add_trace(go.Scatter(x=[row['Trade Date'], row['Opt_Expiry']], y=[row['Opt_Strike'], row['Opt_Strike']],
                    mode='lines', line=dict(color=line_color, width=2, dash='solid' if row['Opt_Type'] == 'Call' else 'dash'),
                    showlegend=False, hoverinfo='text', text=f"{row['Opt_Type']} {row['Opt_Strike']}"), row=1, col=1)
        
        opt_plot = opt_trades.copy()
        opt_plot['Ref'] = df_view['Close_Local'].reindex(opt_plot['Trade Date'], method='nearest').values
        fig.add_trace(go.Scatter(x=opt_plot['Trade Date'], y=opt_plot['Ref'], mode='markers', 
            marker=dict(symbol='circle', color='#2979ff', size=10, line=dict(width=1.5, color='white')),
            name='Option Exec', text=opt_plot['Description']), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Total_PnL_Cumulative'], mode='lines', name='Total PnL',
        line=dict(color='#ffab00', width=2), fill='tozeroy', fillcolor='rgba(255, 171, 0, 0.15)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Stock_PnL_Cumulative'], mode='lines', name='Stock PnL',
        line=dict(color='#00e676', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Option_PnL_Cumulative'], mode='lines', name='Option PnL',
        line=dict(color='#2979ff', width=1.5)), row=3, col=1)

    fig.update_layout(height=900, template="plotly_dark", hovermode="closest", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_portfolio_overview(summary_df):
    df_sorted = summary_df.sort_values('Total PnL', ascending=False)
    
    df_sorted['Color'] = df_sorted['Total PnL'].apply(lambda x: '#00e676' if x >= 0 else '#ff1744')
    
    fig_bar = go.Figure(go.Bar(
        x=df_sorted['Ticker'], y=df_sorted['Total PnL'],
        marker_color=df_sorted['Color'],
        text=df_sorted['Total PnL'], texttemplate='%{y:$,.0f}', textposition='auto',
        customdata=df_sorted['Name'],
        hovertemplate="<b>%{customdata}</b><br>Ticker: %{x}<br>PnL: %{y:$,.0f}"
    ))
    fig_bar.update_layout(title="PnL Leaders & Laggards (USD)", template="plotly_dark", height=500, yaxis_title="PnL ($)")
    
    df_sorted['Abs PnL'] = df_sorted['Total PnL'].abs()
    cap_val = df_sorted['Abs PnL'].quantile(0.9) if not df_sorted.empty else 10000
    
    fig_tree = px.treemap(
        df_sorted, path=['Ticker'], values='Abs PnL', color='Total PnL',
        color_continuous_scale=['#ff1744', '#1e2130', '#00e676'],
        color_continuous_midpoint=0, range_color=[-cap_val, cap_val],
        title="Portfolio Heatmap (Size = Impact, Color = PnL)",
        hover_data={'Name': True, 'Total PnL': ':$,.0f', 'Abs PnL': False}
    )
    fig_tree.update_layout(template="plotly_dark", height=500)
    
    return fig_bar, fig_tree

def plot_portfolio_performance(history_df):
    portfolio_daily = history_df.groupby(history_df.index).agg({
        'Total_PnL_Cumulative': 'sum',
        'Stock_PnL_Cumulative': 'sum',
        'Option_PnL_Cumulative': 'sum',
        'Stock_Mkt_Value_USD': lambda x: x.tolist() 
    })
    
    portfolio_daily['Daily_Total_PnL'] = portfolio_daily['Total_PnL_Cumulative'].diff().fillna(0)

    fig_pnl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig_pnl.add_trace(go.Scatter(x=portfolio_daily.index, y=portfolio_daily['Total_PnL_Cumulative'],
        mode='lines', name='Total Equity Curve',
        line=dict(color='#00e676', width=3), fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)'), row=1, col=1)
    
    colors = ['#00e676' if v >= 0 else '#ff1744' for v in portfolio_daily['Daily_Total_PnL']]
    fig_pnl.add_trace(go.Bar(x=portfolio_daily.index, y=portfolio_daily['Daily_Total_PnL'],
        marker_color=colors, name='Daily PnL'), row=2, col=1)
    
    fig_pnl.update_layout(title="Portfolio PnL Performance", template="plotly_dark", height=600)

    def get_exposures(val_list):
        longs = sum([v for v in val_list if v > 0])
        shorts = sum([v for v in val_list if v < 0])
        return pd.Series([longs, shorts])

    exp_df = portfolio_daily['Stock_Mkt_Value_USD'].apply(get_exposures)
    exp_df.columns = ['Long', 'Short']
    exp_df['Net'] = exp_df['Long'] + exp_df['Short']

    fig_exp = go.Figure()
    fig_exp.add_trace(go.Bar(x=exp_df.index, y=exp_df['Long'], name='Long Exposure', marker_color='#00e676'))
    fig_exp.add_trace(go.Bar(x=exp_df.index, y=exp_df['Short'], name='Short Exposure', marker_color='#ff1744'))
    fig_exp.add_trace(go.Scatter(x=exp_df.index, y=exp_df['Net'], name='Net Exposure', line=dict(color='white', width=2, dash='dot')))
    fig_exp.update_layout(title="Stock Notional Exposure (USD)", template="plotly_dark", barmode='relative', height=400)

    return fig_pnl, fig_exp

# --- MAIN APP ---

st.sidebar.title("📁 Data Import")
soy_file = st.sidebar.file_uploader("Soy Positions (CSV)", type=['csv'])
trade_file = st.sidebar.file_uploader("Trade Record (CSV)", type=['csv'])

if soy_file and trade_file:
    try:
        soy_df, trade_df = load_and_process_data(soy_file, trade_file)
        all_tickers = sorted(list(set(soy_df['YF_Ticker'].dropna()) | set(trade_df['YF_Ticker'].dropna())))
        
        if 'portfolio_summary' not in st.session_state:
            st.session_state['portfolio_summary'] = None
        if 'full_history_df' not in st.session_state:
            st.session_state['full_history_df'] = None

        tab1, tab2, tab3 = st.tabs(["🔎 Ticker Deep Dive", "📊 Portfolio Overview", "📈 Portfolio Performance"])

        with tab1:
            col_sel, col_empty = st.columns([1, 3])
            with col_sel:
                selected_ticker = st.selectbox("Select Ticker to Analyze", all_tickers)
            
            if selected_ticker:
                s_df, f_df, valid_dates, c_name = fetch_market_data(selected_ticker, soy_df[soy_df['YF_Ticker'] == selected_ticker]['Currency'].iloc[0] if selected_ticker in soy_df['YF_Ticker'].values else trade_df[trade_df['YF_Ticker'] == selected_ticker]['Currency'].iloc[0])
                
                if s_df is not None:
                    analysis, s_trades, o_trades, _ = calculate_ticker_metrics(selected_ticker, soy_df, trade_df, s_df, f_df, c_name)
                    
                    curr_total = analysis['Total_PnL_Cumulative'].iloc[-1]
                    curr_stock = analysis['Stock_PnL_Cumulative'].iloc[-1]
                    curr_opt = analysis['Option_PnL_Cumulative'].iloc[-1]
                    
                    st.subheader(f"{c_name} ({selected_ticker})")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.markdown(get_metric_html("Ticker Total PnL", curr_total), unsafe_allow_html=True)
                    with c2: st.markdown(get_metric_html("Stock PnL", curr_stock), unsafe_allow_html=True)
                    with c3: st.markdown(get_metric_html("Option PnL", curr_opt), unsafe_allow_html=True)
                    with c4: st.markdown(get_metric_html("Current Position", analysis['Stock_Pos'].iloc[-1], is_currency=False), unsafe_allow_html=True)
                    
                    st.plotly_chart(plot_ticker_deep_dive(selected_ticker, analysis, s_trades, o_trades, valid_dates), use_container_width=True)
                    
                    with st.expander("View Trade Log"):
                         st.dataframe(trade_df[trade_df['YF_Ticker'] == selected_ticker].sort_values('Trade Date', ascending=False))
                else:
                    st.error("Data not available for this ticker.")

        with tab2:
            if st.session_state['portfolio_summary'] is None:
                st.info("Click below to run the full portfolio analysis (Takes ~30-60s).")
                if st.button("Generate Portfolio Report"):
                    summary_list = []
                    history_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, ticker in enumerate(all_tickers):
                        status_text.text(f"Analyzing {ticker}...")
                        currency = soy_df[soy_df['YF_Ticker'] == ticker]['Currency'].iloc[0] if ticker in soy_df['YF_Ticker'].values else trade_df[trade_df['YF_Ticker'] == ticker]['Currency'].iloc[0]
                        s_df, f_df, _, c_name = fetch_market_data(ticker, currency) 
                        
                        if s_df is not None:
                            analysis_df, _, _, stats = calculate_ticker_metrics(ticker, soy_df, trade_df, s_df, f_df, c_name)
                            summary_list.append(stats)
                            history_list.append(analysis_df)
                            
                        progress_bar.progress((i + 1) / len(all_tickers))
                    
                    st.session_state['portfolio_summary'] = pd.DataFrame(summary_list)
                    st.session_state['full_history_df'] = pd.concat(history_list) if history_list else pd.DataFrame()
                    status_text.success("Analysis Complete!")
                    progress_bar.empty()
                    st.rerun()

            if st.session_state['portfolio_summary'] is not None:
                summary_df = st.session_state['portfolio_summary']
                total_pnl = summary_df['Total PnL'].sum()
                best_trade = summary_df.loc[summary_df['Total PnL'].idxmax()]
                worst_trade = summary_df.loc[summary_df['Total PnL'].idxmin()]
                
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(get_metric_html("Portfolio Total PnL", total_pnl), unsafe_allow_html=True)
                with c2: st.markdown(get_metric_html(f"Best: {best_trade['Name']}", best_trade['Total PnL']), unsafe_allow_html=True)
                with c3: st.markdown(get_metric_html(f"Worst: {worst_trade['Name']}", worst_trade['Total PnL']), unsafe_allow_html=True)
                
                st.divider()
                fig_bar, fig_tree = plot_portfolio_overview(summary_df)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.plotly_chart(fig_tree, use_container_width=True)
                
                st.subheader("Detailed Performance Table")
                styled_df = summary_df.sort_values(by='Total PnL', ascending=False).style \
                    .format({'Total PnL': '${:,.0f}', 'Stock PnL': '${:,.0f}', 'Option PnL': '${:,.0f}', 'Position': '{:,.0f}', 'Price': '{:,.2f}'}) \
                    .bar(subset=['Stock PnL'], align='mid', color=['#ff1744', '#00e676']) \
                    .bar(subset=['Total PnL'], align='mid', color=['#ff1744', '#00e676'])
                
                st.dataframe(styled_df, column_config={"Ticker": st.column_config.TextColumn("Ticker"), "Name": "Company"}, hide_index=True, use_container_width=True)
                if st.button("Refresh Data"):
                    st.session_state['portfolio_summary'] = None
                    st.session_state['full_history_df'] = None
                    st.rerun()

        with tab3:
            if st.session_state['full_history_df'] is not None and not st.session_state['full_history_df'].empty:
                hist_df = st.session_state['full_history_df']
                hist_df = hist_df[hist_df.index >= '2025-01-01']
                f1, f2 = plot_portfolio_performance(hist_df)
                st.plotly_chart(f1, use_container_width=True)
                st.plotly_chart(f2, use_container_width=True)
            else:
                st.info("Please generate the Portfolio Report in the 'Portfolio Overview' tab first.")
                if st.button("Go to Generate Report", key='goto_gen'):
                    st.warning("Go to 'Portfolio Overview' tab and click Generate.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
else:
    st.info("Please upload your `soy_positions.csv` and `trade_record.csv` files in the sidebar to begin.")