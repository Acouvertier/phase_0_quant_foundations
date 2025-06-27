"""
market_behavior_utils.py

Cumulative of Phase 1 - Building Intuition with Returns, Risk & Rates
This is a self-teaching project to transition from Quantum Physics to Quant Finance
Includes yfinance data cleaning, basic return calculations, 
"""

# === Imports ===
import math
import numpy as np
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime as dt

from sympy import symbols, simplify, diff, lambdify
import statsmodels.api as sm
import scipy.stats as sc
from scipy.differentiate import derivative as deriv

# === Constants ===
RISK_FREE_TICKER = "DGS3MO" # 3 Month Treasury Bond Yields, FRED Ticker

# === Bond Symbols ===
fv, c, r, t, n = symbols('f c r t n',real=True)

# === Symbolic Bond Present Value (r > 0) ===
_present_value_sym = (n*c/r) + (fv - (n*c/r))*(1+(r/n))**(-n*t)
_duration_sym = simplify(-diff(_present_value_sym,r)/_present_value_sym)
_convexity_sym = simplify(diff(_present_value_sym,r,r)/_present_value_sym)

# === Lambda Functions of Bond Present Vale (r > 0) ===
_present_value_f = lambdify((r,fv,c,t,n),_present_value_sym)
_duration_f = lambdify((r,fv,c,t,n),_duration_sym)
_convexity_f = lambdify((r,fv,c,t,n),_convexity_sym)

# === Vecotrized Functions of Bond Present Vale (r > 0) ===
_present_value_fv = np.vectorize(_present_value_f)
_duration_fv = np.vectorize(_duration_f)
_convexity_fv = np.vectorize(_convexity_f)

# === Helper Functions ===
def _close_column(df:pd.DataFrame, ticker:str) -> pd.DataFrame:
    """
    Healper function to format yfinance.download() DataFrame

    Parameters:
    df (DataFrame): Dataframe returned from yfinance.download()
    ticker (str): String of the ticker symbol (must be in df)

    Returns:
    DataFrame: Pruned yfinance.download() with only close price with column = CLOSE_{ticker}
    """
    new_df = df[ticker]['Close'].to_frame()
    new_df.rename(columns={'Close':f"CLOSE_{ticker}"},inplace=True)
    return new_df

def _terminal_value(growth_rate:float, discount_rate:float, final_flow:float) -> float:
    """
    Calculates the terminal value of a cash flow under an assumed constant growth rate and discount rate

    Parameters:
    growth_rate (float): constant growth % as a decimal
    discount_rate (float): Risk free interest rate as a decimal
    final_flow (float): Last known value of the cash flow
    
    """
    return ((1+growth_rate)/(discount_rate - growth_rate))*final_flow

def _single_moment_set(df:pd.DataFrame, column:str) -> pd.DataFrame:
    """
    Creates a DataFrame for the four moments of the desired column

    Parameters:
    df (DataFrame): DataFrame of numerical values (at least in desired column)
    column (str): Column in df that contains numerical rows

    Returns:
    DataFrame: A new DataFrame with the original column and new rows for each moment (mean, std, skew, kurt)
    """

    column_data = df[column]
    return pd.DataFrame(
        data=[column_data.mean(),column_data.std(),column_data.skew(),column_data.kurtosis()],
        index=["mean","std","skew","kurt"],
        columns=[column]
        )

def _merge_stock_bond(stock_df:pd.DataFrame, bond_df:pd.DataFrame) -> pd.DataFrame:
    """
    Returns the merge of the result of grab_ticker with get_bond_data

    Parameters:
    stock_df (DataFrame): DataFrame formatted from grab_ticker
    bond_df (DataFrame): DataFrame formatted from get_bond_data

    Returns:
    DataFrame: The result of an inner merge on their Date index
    """
    merged_df = stock_df.merge(bond_df,how='inner',on='Date')
    merged_df = merged_df.ffill()
    return merged_df

# === Core Functions ===

def grab_ticker(start:dt, end:dt, ticker_list:list[str]) -> pd.DataFrame:
    """
    Create a DataFrame with close prices and basic returns for each stock in ticker_list from start to end

    Parameters: 
    start (datetime.dateime): Start date for data call
    end (datetime.datetime): End date for data call
    ticker_list (list): List of str representing valid yfinance ticker symbols

    Returns:
    DataFrame: Processed DataFrame with close prices and basic returns for each stock, columns appropriately matched to ticker
    
    """
    raw_data = yf.download(ticker_list,start=start,end=end,group_by='ticker',keepna=True)
    pruned_data = None
    for ticker in ticker_list:
        if pruned_data is None:
            pruned_data = _close_column(raw_data,ticker)
        else:
            pruned_data = pruned_data.merge(_close_column(raw_data,ticker),how='outer',on='Date')

        pruned_data[f"SIMPLE_{ticker}"] = pruned_data[f"CLOSE_{ticker}"].pct_change()
        pruned_data[f"LOG_{ticker}"] = np.log(1+pruned_data[f"SIMPLE_{ticker}"])

        pruned_data[f"CUM_LOG_{ticker}"] = pruned_data[f"LOG_{ticker}"].fillna(0).cumsum()
        pruned_data[f"CUM_SIMPLE_{ticker}"] = (pruned_data[f"SIMPLE_{ticker}"].fillna(0) + 1).cumprod() - 1

    return pruned_data

def single_year_data(df:pd.DataFrame, year:int) -> pd.DataFrame:
    """
    Grab single year worth of data

    Parameters:
    df (DataFrame): DataFrame with datetime as index
    year (int): calendar year in YYYY format
    
    Returns:
    DataFrame: Only rows with the provided year
    """
    return df.iloc[df.index.year == year]

def npv(discount_rate:float, cash_flow:np.ndarray, interpret:bool) -> float:
    """
    Calculates the Net present Vale (NPV) of an equally spaced cash flow assuming constant discount rate.

    Parameters:
    discount_rate (float): Risk free interest rate as a decimal
    cash_flow (np.ndarray): An array of the equally spaced cash flows
    interpret (bool): If True, prints sentence interpreting the cash flow

    Returns:
    float: NPV
    """

    flow_size = len(cash_flow)
    discount_multipliers = np.array([1/((1+discount_rate)**i) for i in range(flow_size)])
    pre_sum = discount_multipliers * cash_flow
    npv = np.sum(pre_sum)
    if interpret:
        if npv > 0:
            print(f"The net present value is {npv}, which represents a sound investment.")
        elif npv < 0:
            print(f"The net present value is {npv}, which represents a bad investment.")
        else:
            print("The net present value is 0, the investment is ambiguous under this measure.")
        
    return npv

def dcf(growth_rate:float, discount_rate:float, free_cash_flow:np.ndarray) -> float:
    """
    Calculates the Discounted Cash Flow (DCF) associated with a companies Free Cash Flow (FCF), asummed perpetuity growth rate and discount rate
    
    Parameters:
    growth_rate (float): constant growth % as a decimal
    discount_rate (float): Risk free interest rate as a decimal
    free_cash_flow (np.ndarray): An array of the equally spaced cash flows

    Returns:
    float: DCF
    """

    flow_size = len(free_cash_flow)
    discount_multiplier = np.array([1/((1+discount_rate)**i) for i in range(1,flow_size+1)])
    t_v = _terminal_value(growth_rate, discount_rate, free_cash_flow[-1])
    pre_sum = discount_multiplier*free_cash_flow
    return np.sum(pre_sum) + (t_v/(1+discount_rate)**(flow_size))

def fair_stock(debt:float, shares_outstanding:float, growth_rate:float, capital_cost:float, free_cash_flow:np.ndarray) -> float:
    """
    Calculates the 'fair' stock price using DCF and known company debt and shares outstanding

    Parameters:
    debt (float): Dollar amount of company debt in same units as free_cash_flow
    shares_outstanding (float): Number of shares outstanding (same scale as debt and free_cash_flow)
    growth_rate (float): constant growth % as a decimal
    discount_rate (float): Risk free interest rate as a decimal
    free_cash_flow (np.ndarray): An array of the equally spaced cash flows

    Returns:
    float: The fair price of a single stock
    """
    enterprise_value = dcf(growth_rate, capital_cost, free_cash_flow)
    return (enterprise_value - debt) / shares_outstanding

def moments_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    Creates a dataframe with all four moments of the provided dataframe

    Parameters:
    df (DataFrame): DataFrame of numerical values

    Returns:
    DataFrame: same columns as df with four indexes representing the four moments of the provided columns
    """

    columns = df.columns
    new_df = None
    for column in columns:
        if new_df is None:
            new_df = _single_moment_set(df,column)
        else:
            new_df = new_df.join(_single_moment_set(df,column))
    
    return new_df

def gaussian_overlay_data(df:pd.DataFrame, column_name:str) -> tuple:
    """
    Formatted column data and simulated normal data required to perform a Gaussian Overlay

    Parameters:
    df (DataFrame): DataFrame of numerical values
    column_name (str): Column in df to be overlayed
    
    Returns (tuple):
    formatted_data (Series): Column Data with nan removed
    plot_domain (np.arange): X-Values of the Normal Data from df[column_name].min() to df[column_name].max() 
    normal_range (np.array): Y-Values of the Normal Data using df[column_name].mean() and df[column_name].std()

    """
    series = df[column_name]
    
    mu = series.mean()
    std = series.std()
    plot_domain = np.arange(series.min(),series.max(),(series.max()-series.min())/200)
    normal_range = sc.norm.pdf(plot_domain,loc=mu,scale=std)
    
    formatted_data = series.dropna().to_numpy()
    
    return (formatted_data,plot_domain,normal_range)

def gaussian_qqplot(df:pd.DataFrame, column_name:str) -> None:
    """
    QQ Plot against a normal distribution with equal mean

    Parameters:
    df (DataFrame): DataFrame of numerical values
    column_name (str): Column in df to be overlayed
    
    Returns:
    None (In Juypter Notebook the output is displayed)

    """
    series = df[column_name]
    mu = series.mean()
    
    pruned_series = series.dropna().to_numpy()
    _ = sm.qqplot(pruned_series,line='s',loc=mu)
    return None

def get_bond_data(start:dt, end:dt, fred_ticker: str = RISK_FREE_TICKER) -> pd.DataFrame:
    """
    Created DataFrame of Daily Bond Rates using FRED Database

    Parameters:
    start (datetime.datetime): initial date for data query
    end (datetime.datetime): final date for data query
    fred_ticker (str): Valid FRED ticker for Bond Rates

    Returns:
    DataFrame: Daily Bond Rates with index matching grab_ticker
    """
    
    if fred_ticker not in [RISK_FREE_TICKER,"DGS6MO"] + [f"DGS{i}" for i in [1,2,5,7,10,20,30]]:
        fred_ticker = RISK_FREE_TICKER

    raw_data = pdr.fred.FredReader([fred_ticker],start=start,end=end).read()
    daily_data = ((1+raw_data[fred_ticker]/100)**(1/252) - 1)
    return daily_data.rename_axis("Date")

def static_sharpe_df(start:dt, end:dt, stocks:list[str], fred_ticker: str = RISK_FREE_TICKER) -> pd.DataFrame:
    """
    Calculate Sharpe Ratio of all stocks from start to end date using fred_ticker as risk free

    Parameters:
    start (datetime.datetime): initial date for data query
    end (datetime.datetime): final date for data query
    stocks (list): list of valid yfinance stock tickers as str
    fred_ticker (str): Valid FRED ticker for Bond Rates

    Returns:
    DataFrame: New Dataframe single column of static_sharpe and index of stock name
    
    """
    all_stocks = grab_ticker(start,end,stocks)
    bond_data = get_bond_data(start,end,fred_ticker)
    
    stock_and_bond = _merge_stock_bond(all_stocks,bond_data)

    moments = moments_df(stock_and_bond[[f"SIMPLE_{stock}" for stock in stocks] + [fred_ticker]])

    sharpes = [(moments[f"SIMPLE_{stock}"]["mean"] - moments[fred_ticker]["mean"])/(moments[f"SIMPLE_{stock}"]["std"]) for stock in stocks]
    return pd.DataFrame(data=sharpes,index=stocks,columns=['static_sharpe'])

def rolling_statistics(start:dt, end:dt, stocks:list[str], rolling_window:str, fred_ticker: str = RISK_FREE_TICKER) -> pd.DataFrame:
    """
    Calculated rolling mean, std dev, excess returns, and sharpe ratios for all stocks against risk free defined by fred_ticker from start to end

    Parameters:
    start (datetime.datetime): initial date for data query
    end (datetime.datetime): final date for data query
    stocks (list): list of valid yfinance stock tickers as str
    rolling_window (str): Valid window for pd.DataFrame.rolling
    fred_ticker (str): Valid FRED ticker for Bond Rates

    Returns:
    DataFrame: Results of grab_ticker and get_bond_data with rolling statistics as new columns

    """
    
    stock_data = grab_ticker(start,end,stocks)
    bond_data = get_bond_data(start,end,fred_ticker)

    all_stocks = _merge_stock_bond(stock_data,bond_data)

    rolling_data = all_stocks.rolling(window=rolling_window)
    all_stocks[f"{rolling_window}_MEAN_{fred_ticker}"] = rolling_data[fred_ticker].mean()
    
    for stock in stocks:
        all_stocks[f"{rolling_window}_STD_{stock}"] = rolling_data[f"SIMPLE_{stock}"].std()
        all_stocks[f"{rolling_window}_MEAN_{stock}"] = rolling_data[f"SIMPLE_{stock}"].mean()
        all_stocks[f"{rolling_window}_EXCESS_RETURN_{stock}"] = all_stocks[f"{rolling_window}_MEAN_{stock}"]- all_stocks[f"{rolling_window}_MEAN_{fred_ticker}"]
        all_stocks[f"{rolling_window}_SHARPE_{stock}"] = (all_stocks[f"{rolling_window}_EXCESS_RETURN_{stock}"])/all_stocks[f"{rolling_window}_STD_{stock}"]
    
    return all_stocks

def bond_pv(interest_rate:float, face_value:float, time:int, periods:int = 1, coupon: float = 0.0) -> float:
    """
    Calculates the present value of a bond given a current interest rate and coupon payment

    Parameters:
    interest_rate (float): current bond interest rate
    face_value (float): minted value of the bond
    time (int): Number of years to maturity
    periods (int): Number of coupon payments per year (default 1)
    coupon (float): fixed bond payment (default 0.0)

    Returns:
    pv (float): present value of the bond
    """
    
    n = math.floor(time * periods)
    cash_flow = [coupon for _ in range(n)]
    cash_flow[-1] = cash_flow[-1] + face_value
    cash_flow = [0] + cash_flow
    pv = npv(interest_rate/periods, np.array(cash_flow),False)

    return pv

def duration_pv(interest_rates:np.ndarray, face_value:float, time:int, periods:int = 1, coupon: float = 0.0) -> tuple:
    """
    Calculates the duration of a given bond and ndarray of interest rates

    Parameters:
    interest_rates (np.ndarray): All interest rates for bond price derivatives
    face_value (float): minted value of the bond
    time (int): Number of years to maturity
    periods (int): Number of coupon payments per year (default 1)
    coupon (float): fixed bond payment (default 0.0)   

    Returns:
    tuple: (interest_rates, durations)
    """

    pvs = np.vectorize(bond_pv)(interest_rates, face_value, time, periods, coupon)
    derivative_results = deriv(np.vectorize(bond_pv), interest_rates,args=(face_value, time, periods, coupon))
    return (derivative_results.x,-np.divide(derivative_results.df,pvs))

def convexity_pv(interest_rates:np.ndarray, face_value:float, time:int, periods:int = 1, coupon: float = 0.0) -> tuple:
    """
    Calculates the convexity of a given bond and ndarray of interest rates

    Parameters:
    interest_rates (np.ndarray): All interest rates for bond price derivatives
    face_value (float): minted value of the bond
    time (int): Number of years to maturity
    periods (int): Number of coupon payments per year (default 1)
    coupon (float): fixed bond payment (default 0.0)   

    Returns:
    tuple: (interest_rates, convexities)
    """
        
    pvs = np.vectorize(bond_pv)(interest_rates, face_value, time, periods, coupon)
    derivative_results = deriv(np.vectorize(bond_pv), interest_rates,args=(face_value, time, periods, coupon))
    return (derivative_results.x, np.divide(np.gradient(derivative_results.df,derivative_results.x),pvs))


# === Main Guard ===
if __name__  == '__main__':
    print("This module is based on yfinance, pandas, and numpy.")
    print("It is a learning project made by Austen Couvertier.")
    print(f"It can calculate the Net Present Value of cash flow: [-100,20,20,80,30] with discount rate of 10%")
    print(f"npv([-100,20,20,80,30],0.10,True) = {npv([-100,20,20,80,30],0.10,True)}")