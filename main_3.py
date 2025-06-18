from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from arch import arch_model
from scipy.stats import linregress
import xgboost as xgb
import pickle
from io import BytesIO
import os
from time import time
import base64

# Initialize FastAPI app
app = FastAPI(
    title="Options Trading API",
    description="FastAPI backend for options trading strategies and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class LoginRequest(BaseModel):
    access_token: str

class TradeLogRequest(BaseModel):
    strategy: str
    instrument_token: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    realized_pnl: Optional[float] = 0
    unrealized_pnl: Optional[float] = 0
    status: str = "open"
    regime_score: Optional[float] = 0
    notes: Optional[str] = ""

class JournalEntryRequest(BaseModel):
    title: str
    content: str
    mood: str
    tags: Optional[str] = ""

class StrategyRequest(BaseModel):
    strategy_name: str
    lots: int = 1

class OrderPlacementRequest(BaseModel):
    orders: List[Dict[str, Any]]

class DashboardResponse(BaseModel):
    vix: Optional[float]
    nifty: Optional[float]
    seller_metrics: Dict[str, Any]
    market_metrics: Dict[str, Any]
    volatility_data: Dict[str, Any]
    regime_data: Dict[str, Any]
    suggested_strategies: List[str]
    rationale: str
    event_warning: Optional[str]
    upcoming_events: List[Dict[str, Any]]
    full_chain_data: List[Dict[str, Any]]

# Global variables for configuration
current_config = None
all_strategies = [
    "Iron Fly", "Iron Condor", "Jade Lizard", "Straddle",
    "Calendar Spread", "Bull Put Spread", "Wide Strangle", "ATM Strangle"
]

def get_config(access_token: str):
    """Get configuration with access token"""
    config = {
        "access_token": access_token,
        "base_url": "https://api.upstox.com/v2",
        "headers": {
            "accept": "application/json",
            "Api-Version": "2.0",
            "Authorization": f"Bearer {access_token}"
        },
        "instrument_key": "NSE_INDEX|Nifty 50",
        "event_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv",
        "ivp_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv",
        "nifty_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv",
        "total_funds": 2000000,
        "risk_config": {
            "Iron Fly": {"capital_pct": 0.30, "risk_per_trade_pct": 0.01},
            "Iron Condor": {"capital_pct": 0.25, "risk_per_trade_pct": 0.015},
            "Jade Lizard": {"capital_pct": 0.20, "risk_per_trade_pct": 0.01},
            "Straddle": {"capital_pct": 0.15, "risk_per_trade_pct": 0.02},
            "Calendar Spread": {"capital_pct": 0.10, "risk_per_trade_pct": 0.01},
            "Bull Put Spread": {"capital_pct": 0.15, "risk_per_trade_pct": 0.01},
            "Wide Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015},
            "ATM Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015}
        },
        "daily_risk_limit_pct": 0.02,
        "weekly_risk_limit_pct": 0.03,
        "lot_size": 75
    }

    def get_next_expiry_internal():
        try:
            url = f"{config['base_url']}/option/contract"
            params = {"instrument_key": config['instrument_key']}
            res = requests.get(url, headers=config['headers'], params=params)
            if res.status_code == 200:
                expiries = sorted(res.json()["data"], key=lambda x: datetime.strptime(x["expiry"], "%Y-%m-%d"))
                today = datetime.now()
                for expiry in expiries:
                    expiry_dt = datetime.strptime(expiry["expiry"], "%Y-%m-%d")
                    if expiry_dt.weekday() == 3 and expiry_dt > today:
                        return expiry["expiry"]
                return datetime.now().strftime("%Y-%m-%d")
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            return datetime.now().strftime("%Y-%m-%d")

    config['expiry_date'] = get_next_expiry_internal()
    return config

def get_current_config():
    """Dependency to get current configuration"""
    if current_config is None:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login first.")
    return current_config

# Data fetching functions
def fetch_option_chain(config):
    """Fetch option chain data"""
    try:
        url = f"{config['base_url']}/option/chain"
        params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        return []
    except Exception as e:
        return []

def get_indices_quotes(config):
    """Get VIX and Nifty quotes"""
    try:
        url = f"{config['base_url']}/market-quote/quotes?instrument_key=NSE_INDEX|India VIX,NSE_INDEX|Nifty 50"
        res = requests.get(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json().get("data", {})
            vix = data["NSE_INDEX:India VIX"]["last_price"] if "NSE_INDEX:India VIX" in data else None
            nifty = data["NSE_INDEX:Nifty 50"]["last_price"] if "NSE_INDEX:Nifty 50" in data else None
            return vix, nifty
        return None, None
    except Exception as e:
        return None, None

def load_upcoming_events(config):
    """Load upcoming events"""
    try:
        df = pd.read_csv(config['event_url'])
        df["Datetime"] = pd.to_datetime(df["Date"].str.strip() + " " + df["Time"].str.strip(), format="%d-%b %H:%M", errors="coerce")
        current_year = datetime.now().year
        df["Datetime"] = df["Datetime"].apply(
            lambda dt: dt.replace(year=current_year) if pd.notnull(dt) and dt.year == 1900 else dt
        )
        now = datetime.now()
        expiry_dt = datetime.strptime(config['expiry_date'], "%Y-%m-%d")
        mask = (df["Datetime"] >= now) & (df["Datetime"] <= expiry_dt)
        filtered = df.loc[mask, ["Datetime", "Event", "Classification", "Forecast", "Prior"]]
        return filtered.sort_values("Datetime").reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])

def load_ivp(config, avg_iv):
    """Load implied volatility percentile"""
    try:
        iv_df = pd.read_csv(config['ivp_url'])
        iv_df.dropna(subset=["ATM_IV"], inplace=True)
        iv_df = iv_df.tail(30)
        ivp = round((iv_df["ATM_IV"] < avg_iv).sum() / len(iv_df) * 100, 2)
        return ivp
    except Exception as e:
        return 0

def load_xgboost_model():
    """Load XGBoost model for volatility prediction"""
    try:
        model_url = "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"
        response = requests.get(model_url)
        if response.status_code == 200:
            model = pickle.load(BytesIO(response.content))
            return model
        return None
    except Exception as e:
        return None

def predict_xgboost_volatility(model, atm_iv, realized_vol, ivp, pcr, vix, days_to_expiry, garch_vol):
    """Predict volatility using XGBoost model"""
    try:
        features = pd.DataFrame({
            'ATM_IV': [atm_iv],
            'Realized_Vol': [realized_vol],
            'IVP': [ivp],
            'PCR': [pcr],
            'VIX': [vix],
            'Days_to_Expiry': [days_to_expiry],
            'GARCH_Predicted_Vol': [garch_vol]
        })
        if model is not None:
            prediction = model.predict(features)[0]
            return round(float(prediction), 2)
        return 0
    except Exception as e:
        return 0

def extract_seller_metrics(option_chain, spot_price):
    """Extract ATM seller metrics"""
    try:
        atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
        call = atm["call_options"]
        put = atm["put_options"]
        return {
            "strike": atm["strike_price"],
            "straddle_price": call["market_data"]["ltp"] + put["market_data"]["ltp"],
            "avg_iv": (call["option_greeks"]["iv"] + put["option_greeks"]["iv"]) / 2,
            "theta": call["option_greeks"]["theta"] + put["option_greeks"]["theta"],
            "vega": call["option_greeks"]["vega"] + put["option_greeks"]["vega"],
            "delta": call["option_greeks"]["delta"] + put["option_greeks"]["delta"],
            "gamma": call["option_greeks"]["gamma"] + put["option_greeks"]["gamma"],
            "pop": ((call["option_greeks"]["pop"] + put["option_greeks"]["pop"]) / 2),
        }
    except Exception as e:
        return {}

def full_chain_table(option_chain, spot_price):
    """Generate full option chain table"""
    try:
        chain_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            if abs(strike - spot_price) <= 300:
                call = opt["call_options"]
                put = opt["put_options"]
                chain_data.append({
                    "Strike": strike,
                    "Call IV": call["option_greeks"]["iv"],
                    "Put IV": put["option_greeks"]["iv"],
                    "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"],
                    "Total Theta": call["option_greeks"]["theta"] + put["option_greeks"]["theta"],
                    "Total Vega": call["option_greeks"]["vega"] + put["option_greeks"]["vega"],
                    "Straddle Price": call["market_data"]["ltp"] + put["market_data"]["ltp"],
                    "Total OI": call["market_data"]["oi"] + put["market_data"]["oi"]
                })
        return chain_data
    except Exception as e:
        return []

def market_metrics(option_chain, expiry_date):
    """Calculate market metrics"""
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain if "call_options" in opt and "market_data" in opt["call_options"])
        put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain if "put_options" in opt and "market_data" in opt["put_options"])
        pcr = put_oi / call_oi if call_oi != 0 else 0
        
        # Calculate max pain
        strikes = sorted(set(opt["strike_price"] for opt in option_chain))
        max_pain_strike = 0
        min_pain = float('inf')
        for strike in strikes:
            pain_at_strike = 0
            for opt in option_chain:
                if "call_options" in opt:
                    pain_at_strike += max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"]
                if "put_options" in opt:
                    pain_at_strike += max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]
            if pain_at_strike < min_pain:
                min_pain = pain_at_strike
                max_pain_strike = strike
        
        return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain_strike}
    except Exception as e:
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

def calculate_volatility(config, seller_avg_iv):
    """Calculate historical and GARCH volatility"""
    try:
        df = pd.read_csv(config['nifty_url'])
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y")
        df = df.sort_values('Date')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
        
        hv_7 = np.std(df["Log_Returns"][-7:]) * np.sqrt(252) * 100
        
        model = arch_model(df["Log_Returns"], vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=7)
        garch_7d = np.mean(np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252) * 100)
        
        iv_rv_spread = round(seller_avg_iv - hv_7, 2)
        return hv_7, garch_7d, iv_rv_spread
    except Exception as e:
        return 0, 0, 0

def calculate_iv_skew_slope(full_chain_data):
    """Calculate IV skew slope"""
    try:
        if not full_chain_data:
            return 0
        df = pd.DataFrame(full_chain_data)
        slope, _, _, _, _ = linregress(df["Strike"], df["IV Skew"])
        return round(slope, 4)
    except Exception as e:
        return 0

def calculate_regime(atm_iv, ivp, realized_vol, garch_vol, straddle_price, spot_price, pcr, vix, iv_skew_slope):
    """Calculate market regime"""
    expected_move = (straddle_price / spot_price) * 100
    vol_spread = atm_iv - realized_vol
    regime_score = 0
    
    regime_score += 10 if ivp > 80 else -10 if ivp < 20 else 0
    regime_score += 10 if vol_spread > 10 else -10 if vol_spread < -10 else 0
    regime_score += 10 if vix > 20 else -10 if vix < 10 else 0
    regime_score += 5 if pcr > 1.2 else -5 if pcr < 0.8 else 0
    regime_score += 5 if abs(iv_skew_slope) > 0.001 else 0
    regime_score += 10 if expected_move > 0.05 else -10 if expected_move < 0.02 else 0
    regime_score += 5 if garch_vol > realized_vol * 1.2 else -5 if garch_vol < realized_vol * 0.8 else 0
    
    if regime_score > 20:
        return regime_score, "High Vol Trend", "Market in high volatility — ideal for premium selling.", "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
    elif regime_score > 10:
        return regime_score, "Elevated Volatility", "Above-average volatility — favor range-bound strategies.", "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
    elif regime_score > -10:
        return regime_score, "Neutral Volatility", "Balanced market — flexible strategy selection.", "IV and RV aligned, with moderate PCR and skew."
    else:
        return regime_score, "Low Volatility", "Low volatility — cautious selling or long vega plays.", "Low IVP, tight straddle, and low VIX suggest limited movement."

def suggest_strategy(regime_label, ivp, iv_minus_rv, days_to_expiry, event_df, expiry_date, straddle_price, spot_price):
    """Suggest trading strategies based on market conditions"""
    strategies = []
    rationale = []
    event_warning = None
    
    event_window = 3 if ivp > 80 else 2
    high_impact_event_near = False
    event_impact_score = 0
    
    for _, row in event_df.iterrows():
        try:
            dt = pd.to_datetime(row["Datetime"])
            level = row["Classification"]
            if level == "High" and (0 <= (datetime.strptime(expiry_date, "%Y-%m-%d") - dt).days <= event_window):
                high_impact_event_near = True
            if level == "High" and pd.notnull(row["Forecast"]) and pd.notnull(row["Prior"]):
                forecast = float(str(row["Forecast"]).strip("%")) if "%" in str(row["Forecast"]) else float(row["Forecast"])
                prior = float(str(row["Prior"]).strip("%")) if "%" in str(row["Prior"]) else float(row["Prior"])
                if abs(forecast - prior) > 0.5:
                    event_impact_score += 1
        except Exception as e:
            continue
    
    if high_impact_event_near:
        event_warning = f"High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
    
    if event_impact_score > 0:
        rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")
    
    expected_move_pct = (straddle_price / spot_price) * 100
    
    if regime_label == "High Vol Trend":
        strategies = ["Iron Fly", "Wide Strangle"]
        rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif regime_label == "Elevated Volatility":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif regime_label == "Neutral Volatility":
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced — slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Fly"]
            rationale.append("Tight expiry — quick theta-based capture via short Iron Fly.")
    elif regime_label == "Low Volatility":
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry — benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")
    
    if event_impact_score > 0 and not high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
    
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) — avoid unhedged selling.")
    
    rationale.append(f"Expected move: ±{expected_move_pct:.2f}% based on straddle price.")
    
    return strategies, " | ".join(rationale), event_warning

# Strategy calculation functions
def find_option_by_strike(option_chain, strike, option_type):
    """Find option by strike and type"""
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                if option_type == "CE" and "call_options" in opt:
                    return opt["call_options"]
                elif option_type == "PE" and "put_options" in opt:
                    return opt["put_options"]
        return None
    except Exception as e:
        return None

def get_dynamic_wing_distance(ivp, straddle_price):
    """Calculate dynamic wing distance based on IVP"""
    if ivp >= 80:
        multiplier = 0.35
    elif ivp <= 20:
        multiplier = 0.2
    else:
        multiplier = 0.25
    raw_distance = straddle_price * multiplier
    return int(round(raw_distance / 50.0)) * 50

def get_strategy_details(strategy_name, option_chain, spot_price, config, lots=1):
    """Get strategy details with orders and pricing"""
    func_map = {
        "Iron Fly": _iron_fly_calc,
        "Iron Condor": _iron_condor_calc,
        "Jade Lizard": _jade_lizard_calc,
        "Straddle": _straddle_calc,
        "Calendar Spread": _calendar_spread_calc,
        "Bull Put Spread": _bull_put_spread_calc,
        "Wide Strangle": _wide_strangle_calc,
        "ATM Strangle": _atm_strangle_calc
    }
    
    if strategy_name not in func_map:
        return None
    
    try:
        detail = func_map[strategy_name](option_chain, spot_price, config, lots=lots)
    except Exception as e:
        return None
    
    if detail:
        # Update orders with current prices
        ltp_map = {}
        for opt in option_chain:
            if "call_options" in opt and "market_data" in opt["call_options"]:
                ltp_map[opt["call_options"]["instrument_key"]] = opt["call_options"]["market_data"].get("ltp", 0)
            if "put_options" in opt and "market_data" in opt["put_options"]:
                ltp_map[opt["put_options"]["instrument_key"]] = opt["put_options"]["market_data"].get("ltp", 0)
        
        updated_orders = []
        for order in detail["orders"]:
            key = order["instrument_key"]
            ltp = ltp_map.get(key, 0)
            updated_orders.append({**order, "current_price": ltp})
        
        detail["orders"] = updated_orders
        
        # Calculate premium
        premium = 0
        for order in detail["orders"]:
            price = order["current_price"]
            qty = order["quantity"]
            if order["transaction_type"] == "SELL":
                premium += price * qty
            else:
                premium -= price * qty
        
        detail["premium"] = premium / config["lot_size"]
        detail["premium_total"] = premium
        
        # Calculate max loss and profit
        if strategy_name in ["Iron Fly", "Iron Condor", "Jade Lizard", "Bull Put Spread"]:
            wing_width = abs(detail["strikes"][0] - detail["strikes"][-1])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
            detail["max_profit"] = detail["premium_total"]
        elif strategy_name in ["Straddle", "Wide Strangle", "ATM Strangle"]:
            detail["max_loss"] = float("inf")
            detail["max_profit"] = detail["premium_total"]
        elif strategy_name == "Calendar Spread":
            detail["max_loss"] = detail["premium"]
            detail["max_profit"] = float("inf")
    
    return detail

# Strategy calculation functions
def _iron_fly_calc(option_chain, spot_price, config, lots):
    """Calculate Iron Fly strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"]["ltp"] + atm["put_options"]["market_data"]["ltp"]
    
    # Calculate IVP for wing distance
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    ivp = load_ivp(config, seller_metrics.get("avg_iv", 20))
    wing_distance = get_dynamic_wing_distance(ivp, straddle_price)
    
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing_distance, "PE")
    
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    
    return {"strategy": "Iron Fly", "strikes": [strike + wing_distance, strike, strike - wing_distance], "orders": orders}

def _iron_condor_calc(option_chain, spot_price, config, lots):
    """Calculate Iron Condor strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"]["ltp"] + atm["put_options"]["market_data"]["ltp"]
    
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    ivp = load_ivp(config, seller_metrics.get("avg_iv", 20))
    short_wing_distance = get_dynamic_wing_distance(ivp, straddle_price)
    long_wing_distance = int(round(short_wing_distance * 1.5 / 50)) * 50
    
    ce_short_opt = find_option_by_strike(option_chain, strike + short_wing_distance, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike - short_wing_distance, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + long_wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - long_wing_distance, "PE")
    
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    
    return {"strategy": "Iron Condor", "strikes": [strike + long_wing_distance, strike + short_wing_distance, strike - short_wing_distance, strike - long_wing_distance], "orders": orders}

def _jade_lizard_calc(option_chain, spot_price, config, lots):
    """Calculate Jade Lizard strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    put_long_strike = atm["strike_price"] - 100
    
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, put_long_strike, "PE")
    
    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    
    return {"strategy": "Jade Lizard", "strikes": [call_strike, put_strike, put_long_strike], "orders": orders}

def _straddle_calc(option_chain, spot_price, config, lots):
    """Calculate Straddle strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    
    if not all([ce_short_opt, pe_short_opt]):
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    
    return {"strategy": "Straddle", "strikes": [strike, strike], "orders": orders}

def _calendar_spread_calc(option_chain, spot_price, config, lots):
    """Calculate Calendar Spread strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    
    near_leg = find_option_by_strike(option_chain, strike, "CE")
    far_leg = find_option_by_strike(option_chain, strike, "CE")
    
    if not all([near_leg, far_leg]):
        return None
    
    orders = [
        {"instrument_key": near_leg["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": far_leg["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    
    return {"strategy": "Calendar Spread", "strikes": [strike, strike], "orders": orders}

def _bull_put_spread_calc(option_chain, spot_price, config, lots):
    """Calculate Bull Put Spread strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_strike = atm["strike_price"] - 50
    long_strike = atm["strike_price"] - 100
    
    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")
    
    if not all([pe_short_opt, pe_long_opt]):
        return None
    
    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    
    return {"strategy": "Bull Put Spread", "strikes": [short_strike, long_strike], "orders": orders}

def _wide_strangle_calc(option_chain, spot_price, config, lots):
    """Calculate Wide Strangle strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 100
    put_strike = atm["strike_price"] - 100
    
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    
    if not all([ce_short_opt, pe_short_opt]):
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    
    return {"strategy": "Wide Strangle", "strikes": [call_strike, put_strike], "orders": orders}

def _atm_strangle_calc(option_chain, spot_price, config, lots):
    """Calculate ATM Strangle strategy"""
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    
    if not all([ce_short_opt, pe_short_opt]):
        return None
    
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    
    return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike], "orders": orders}

def place_multi_leg_orders(config, orders):
    """Place multi-leg orders"""
    try:
        sorted_orders = sorted(orders, key=lambda x: 0 if x["transaction_type"] == "BUY" else 1)
        payload = []
        for idx, order in enumerate(sorted_orders):
            correlation_id = f"s{idx}_{int(time()) % 1000000}"
            payload.append({
                "quantity": abs(order["quantity"]),
                "product": "D",
                "validity": "DAY",
                "price": order.get("current_price", 0),
                "tag": f"{order['instrument_key']}_leg_{idx}",
                "slice": False,
                "instrument_token": order["instrument_key"],
                "order_type": "MARKET",
                "transaction_type": order["transaction_type"],
                "disclosed_quantity": 0,
                "trigger_price": 0,
                "is_amo": False,
                "correlation_id": correlation_id
            })
        
        url = f"{config['base_url']}/order/multi/place"
        res = requests.post(url, headers=config['headers'], json=payload)
        if res.status_code == 200:
            return {"success": True, "message": "Multi-leg order placed successfully!"}
        else:
            return {"success": False, "message": f"Failed to place multi-leg order: {res.status_code} - {res.text}"}
    except Exception as e:
        return {"success": False, "message": f"Error placing multi-leg order: {e}"}

def get_funds_and_margin(config):
    """Get funds and margin information"""
    try:
        url = f"{config['base_url']}/user/get-funds-and-margin?segment=SEC"
        res = requests.get(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json().get("data", {})
            equity_data = data.get("equity", {})
            return {
                "available_margin": float(equity_data.get("available_margin", 0)),
                "used_margin": float(equity_data.get("used_margin", 0)),
                "total_funds": float(equity_data.get("notional_cash", 0))
            }
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}
    except Exception as e:
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}

# API Endpoints
@app.post("/login")
async def login(request: LoginRequest):
    """Login with access token"""
    global current_config
    try:
        current_config = get_config(request.access_token)
        # Test the token by making a simple API call
        vix, nifty = get_indices_quotes(current_config)
        if vix is None and nifty is None:
            current_config = None
            raise HTTPException(status_code=401, detail="Invalid access token")
        
        return {"message": "Login successful", "status": "authenticated"}
    except Exception as e:
        current_config = None
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@app.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(config = Depends(get_current_config)):
    """Get dashboard data with market analysis"""
    try:
        # Fetch basic data
        option_chain = fetch_option_chain(config)
        vix, nifty = get_indices_quotes(config)
        
        if not option_chain or nifty is None:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        # Calculate metrics
        seller_metrics = extract_seller_metrics(option_chain, nifty)
        market_data = market_metrics(option_chain, config['expiry_date'])
        full_chain_data = full_chain_table(option_chain, nifty)
        
        # Load additional data
        upcoming_events = load_upcoming_events(config)
        ivp = load_ivp(config, seller_metrics.get("avg_iv", 20))
        
        # Calculate volatility
        hv_7, garch_7d, iv_rv_spread = calculate_volatility(config, seller_metrics.get("avg_iv", 20))
        
        # Calculate IV skew slope
        iv_skew_slope = calculate_iv_skew_slope(full_chain_data)
        
        # Calculate regime
        regime_score, regime_label, regime_desc, regime_detail = calculate_regime(
            seller_metrics.get("avg_iv", 20), ivp, hv_7, garch_7d,
            seller_metrics.get("straddle_price", 0), nifty,
            market_data.get("pcr", 0), vix or 0, iv_skew_slope
        )
        
        # Suggest strategies
        strategies, rationale, event_warning = suggest_strategy(
            regime_label, ivp, iv_rv_spread, market_data.get("days_to_expiry", 0),
            upcoming_events, config['expiry_date'],
            seller_metrics.get("straddle_price", 0), nifty
        )
        
        # Prepare response
        volatility_data = {
            "historical_vol_7d": round(hv_7, 2),
            "garch_vol_7d": round(garch_7d, 2),
            "iv_rv_spread": iv_rv_spread,
            "ivp": ivp
        }
        
        regime_data = {
            "score": regime_score,
            "label": regime_label,
            "description": regime_desc,
            "detail": regime_detail
        }
        
        # Convert upcoming events to dict format
        events_list = []
        for _, row in upcoming_events.iterrows():
            events_list.append({
                "datetime": row["Datetime"].isoformat() if pd.notnull(row["Datetime"]) else None,
                "event": row["Event"],
                "classification": row["Classification"],
                "forecast": row["Forecast"],
                "prior": row["Prior"]
            })
        
        return DashboardResponse(
            vix=vix,
            nifty=nifty,
            seller_metrics=seller_metrics,
            market_metrics=market_data,
            volatility_data=volatility_data,
            regime_data=regime_data,
            suggested_strategies=strategies,
            rationale=rationale,
            event_warning=event_warning,
            upcoming_events=events_list,
            full_chain_data=full_chain_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {str(e)}")

@app.get("/strategies")
async def get_all_strategies():
    """Get list of all available strategies"""
    return {"strategies": all_strategies}

@app.post("/strategy-details")
async def get_strategy_details_endpoint(request: StrategyRequest, config = Depends(get_current_config)):
    """Get detailed strategy information with orders and pricing"""
    try:
        option_chain = fetch_option_chain(config)
        vix, nifty = get_indices_quotes(config)
        
        if not option_chain or nifty is None:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        strategy_details = get_strategy_details(request.strategy_name, option_chain, nifty, config, request.lots)
        
        if not strategy_details:
            raise HTTPException(status_code=400, detail=f"Strategy {request.strategy_name} not supported or invalid")
        
        return strategy_details
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating strategy details: {str(e)}")

@app.post("/place-orders")
async def place_orders(request: OrderPlacementRequest, config = Depends(get_current_config)):
    """Place multi-leg orders"""
    try:
        result = place_multi_leg_orders(config, request.orders)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing orders: {str(e)}")

@app.get("/funds-margin")
async def get_funds_margin(config = Depends(get_current_config)):
    """Get funds and margin information"""
    try:
        funds_data = get_funds_and_margin(config)
        return funds_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching funds data: {str(e)}")

@app.get("/option-chain")
async def get_option_chain(config = Depends(get_current_config)):
    """Get full option chain data"""
    try:
        option_chain = fetch_option_chain(config)
        vix, nifty = get_indices_quotes(config)
        
        if not option_chain:
            raise HTTPException(status_code=500, detail="Failed to fetch option chain")
        
        return {
            "option_chain": option_chain,
            "spot_price": nifty,
            "vix": vix,
            "expiry_date": config['expiry_date']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching option chain: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

