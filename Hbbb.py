Import os
import logging
import pandas as pd
import numpy as np
import httpx  # Asynchronous HTTP client
import pickle
from io import BytesIO
from datetime import datetime, timedelta
from arch import arch_model
from scipy.stats import linregress
from fastapi import FastAPI, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, Body, APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError  # Import ValidationError
from typing import List, Optional, Dict, Any, Set
import websockets

# --- Environment Variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
XG_BOOST_MODEL_URL = os.getenv("XG_BOOST_MODEL_URL", "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl")
NIFTY_HIST_URL = os.getenv("NIFTY_HIST_URL", "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
IVP_HIST_URL = os.getenv("IVP_HIST_URL", "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv")
UPCOMING_EVENTS_URL = os.getenv("UPCOMING_EVENTS_URL", "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set as environment variables.")

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SUPABASE CLIENT ---
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

async def log_trade_to_supabase(data: dict):
    data["timestamp_entry"] = datetime.utcnow().isoformat() + "Z"  # ISO 8601 with Z for UTC
    data["timestamp_exit"] = datetime.utcnow().isoformat() + "Z"
    data["status"] = "closed"  # Assuming logs are for closed trades
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{SUPABASE_URL}/rest/v1/trade_logs", json=data, headers=SUPABASE_HEADERS)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            logger.info(f"Trade logged to Supabase: {response.json()}")
            return response.status_code, response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error logging trade to Supabase: {e.response.status_code} - {e.response.text}")
        return e.response.status_code, {"error": e.response.text}
    except httpx.RequestError as e:
        logger.error(f"Network error logging trade to Supabase: {e}")
        return 500, {"error": str(e)}

async def add_journal_to_supabase(data: dict):
    """Adds a journal entry to Supabase."""
    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{SUPABASE_URL}/rest/v1/journals", json=data, headers={**SUPABASE_HEADERS, "Authorization": f"Bearer {SUPABASE_KEY}"})
            response.raise_for_status()
            logger.info(f"Journal entry added to Supabase: {response.json()}")
            return response.status_code, response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error adding journal to Supabase: {e.response.status_code} - {e.response.text}")
        return e.response.status_code, {"error": e.response.text}
    except httpx.RequestError as e:
        logger.error(f"Network error adding journal to Supabase: {e}")
        return 500, {"error": str(e)}

# --- FastAPI Setup ---
app = FastAPI(title="Volguard Trading API", description="API for Volguard Trading Copilot", version="1.0.0")

# --- Pydantic Models for Request/Response Validation ---

class LoginRequest(BaseModel):
    access_token: str

class StrategyOrderRequest(BaseModel):
    strategy: str
    lots: int = 1
    sl_percentage: float = 10.0
    order_type: str = "MARKET"
    validity: str = "DAY"

class JournalEntryRequest(BaseModel):
    title: str
    content: str
    mood: str
    tags: Optional[str] = ""

class UpstoxOrderRequest(BaseModel):
    quantity: int
    product: str
    validity: str
    price: float
    tag: Optional[str] = None
    slice: Optional[bool] = False
    instrument_key: str
    order_type: str
    transaction_type: str
    disclosed_quantity: Optional[int] = Field(0, ge=0)
    trigger_price: Optional[float] = Field(0, ge=0)
    is_amo: Optional[bool] = False
    correlation_id: Optional[str] = None

class StrategyLeg(BaseModel):
    instrument_key: str
    quantity: int
    transaction_type: str
    product: str
    validity: str
    order_type: str
    price: float
    current_price: Optional[float] = None

class StrategyDetailResponse(BaseModel):
    strategy: str
    strikes: List[float]
    orders: List[StrategyLeg]
    premium: float # Premium per lot
    premium_total: float # Total premium for all lots
    max_profit: float
    max_loss: float
    margin: float

class DashboardResponse(BaseModel):
    nifty: float
    vix: float
    atm_strike: float
    straddle_price: float
    atm_iv: float
    ivp: float
    days_to_expiry: int
    pcr: float
    iv_rv_spread: float
    iv_skew_slope: float
    regime: str
    regime_score: float
    regime_note: str
    regime_explanation: str
    events: List[Dict[str, Any]]
    strategies: List[str]
    strategy_rationale: str
    event_warning: Optional[str]

class OptionChainDataRow(BaseModel):
    Strike: float
    Call_IV: float = Field(..., alias="Call IV")
    Put_IV: float = Field(..., alias="Put IV")
    IV_Skew: float = Field(..., alias="IV Skew")
    Total_Theta: float = Field(..., alias="Total Theta")
    Total_Vega: float = Field(..., alias="Total Vega")
    Straddle_Price: float = Field(..., alias="Straddle Price")
    Total_OI: int = Field(..., alias="Total OI")

class OptionChainResponse(BaseModel):
    option_chain: List[OptionChainDataRow]
    theta_vega_ranking: List[Dict[str, Any]]

class TradeLogEntry(BaseModel):
    strategy: str
    instrument_token: str
    entry_price: float
    quantity: float
    realized_pnl: float
    unrealized_pnl: float
    regime_score: Optional[float] = None
    notes: Optional[str] = ""
    capital_used: Optional[float] = None
    potential_loss: Optional[float] = None
    sl_hit: Optional[bool] = False
    vega: Optional[float] = None
    timestamp_entry: Optional[str] = None
    timestamp_exit: Optional[str] = None
    status: Optional[str] = "closed"

class JournalLogEntry(BaseModel):
    title: str
    content: str
    mood: str
    tags: Optional[str] = ""
    timestamp: Optional[str] = None # Added for consistency if Supabase returns it

class PortfolioStrategySummary(BaseModel):
    Strategy: str
    Capital_Used: float = Field(..., alias="Capital Used")
    Cap_Limit: float = Field(..., alias="Cap Limit")
    Percent_Used: float = Field(..., alias="% Used")
    Potential_Risk: float = Field(..., alias="Potential Risk")
    Risk_Limit: float = Field(..., alias="Risk Limit")
    Realized_P_L: float = Field(..., alias="Realized P&L")
    Unrealized_P_L: float = Field(..., alias="Unrealized P&L")
    Vega: float
    Risk_OK: str = Field(..., alias="Risk OK?")

class PortfolioSummaryResponse(BaseModel):
    Total_Funds: float = Field(..., alias="Total Funds")
    Capital_Deployed: float = Field(..., alias="Capital Deployed")
    Exposure_Percent: float = Field(..., alias="Exposure Percent")
    Risk_on_Table: float = Field(..., alias="Risk on Table")
    Risk_Percent: float = Field(..., alias="Risk Percent")
    Daily_Risk_Limit: float = Field(..., alias="Daily Risk Limit")
    Weekly_Risk_Limit: float = Field(..., alias="Weekly Risk Limit")
    Total_Realized_P_L: float = Field(..., alias="Total Realized P&L")
    Total_Unrealized_P_L: float = Field(..., alias="Total Unrealized P&L")
    Drawdown_Rupees: float = Field(..., alias="Drawdown â‚¹")
    Drawdown_Percent: float = Field(..., alias="Drawdown Percent")
    Max_Drawdown_Allowed: float = Field(..., alias="Max Drawdown Allowed")
    Total_Vega_Exposure: float = Field(..., alias="Total Vega Exposure")
    Flags: List[str]

class PortfolioResponse(BaseModel):
    available_capital: float
    used_margin: float
    exposure_percent: float
    sharpe_ratio: float
    strategy_summary: List[PortfolioStrategySummary]
    risk_alerts: List[str]

# Dependency to validate access token
api_key_header = APIKeyHeader(name="X-Access-Token", auto_error=False)

async def get_access_token_dependency(token: str = Depends(api_key_header)):
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Access token required")
    config = await get_config(token)
    test_url = f"{config['base_url']}/user/profile"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(test_url, headers=config['headers'])
            if res.status_code != 200:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {res.status_code} - {res.text}")
            return config
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error validating token: {str(e)}")

# --- Helper Functions for Database (Supabase) ---
async def get_all_trades(status: Optional[str] = None):
    """Fetches all logged trade entries from Supabase, with optional status filter."""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{SUPABASE_URL}/rest/v1/trade_logs"
            headers = {**SUPABASE_HEADERS, "Authorization": f"Bearer {SUPABASE_KEY}"}
            if status:
                url += f"?status=eq.{status}"  # Supabase filter syntax
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            logger.info("Successfully fetched trades from Supabase.")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching trades: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching trades: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=f"Network error fetching trades: {str(e)}")

async def get_all_journals():
    """Fetches all logged journal entries from Supabase."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SUPABASE_URL}/rest/v1/journals", headers={**SUPABASE_HEADERS, "Authorization": f"Bearer {SUPABASE_KEY}"})
            response.raise_for_status()
            logger.info("Successfully fetched journals from Supabase.")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching journals: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching journals: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching journals: {e}")
        raise HTTPException(status_code=500, detail=f"Network error fetching journals: {str(e)}")

def trades_to_dataframe() -> pd.DataFrame:
    """Placeholder for converting Supabase trades to DataFrame. In a real app, you'd fetch from DB."""
    # For now, return an empty DataFrame or mock data
    logger.warning("trades_to_dataframe is a placeholder. Implement actual DB fetch and conversion.")
    return pd.DataFrame()

def journals_to_dataframe() -> pd.DataFrame:
    """Placeholder for converting Supabase journals to DataFrame. In a real app, you'd fetch from DB."""
    # For now, return an empty DataFrame or mock data
    logger.warning("journals_to_dataframe is a placeholder. Implement actual DB fetch and conversion.")
    return pd.DataFrame()


# --- CONFIGURATION & UPSTOX API SETUP ---
def get_upstox_headers(access_token: str) -> Dict[str, str]:
    """Returns standard headers for Upstox API calls, including Authorization."""
    return {
        "accept": "application/json",
        "Api-Version": "2.0",
        "Authorization": f"Bearer {access_token}"
    }

async def get_config(access_token: str) -> Dict[str, Any]:
    """
    Retrieves dynamic configuration, including Upstox API headers, instrument details,
    and calculates the next Thursday expiry date.
    """
    upstox_headers = get_upstox_headers(access_token)
    config = {
        "base_url": "https://api.upstox.com/v2",
        "headers": upstox_headers,
        "instrument_key": "NSE_INDEX|Nifty 50",  # Assuming Nifty 50 is the primary instrument
        "nifty_url": NIFTY_HIST_URL,
        "ivp_url": IVP_HIST_URL,
        "event_url": UPCOMING_EVENTS_URL,
        "total_funds": 2000000,  # Default total funds, can be made dynamic
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
        "lot_size": 50  # Nifty lot size as of recent changes (was 75) - adjust if needed
    }

    try:
        async with httpx.AsyncClient() as client:
            url = f"{config['base_url']}/option/contract"
            params = {"instrument_key": config['instrument_key']}
            res = await client.get(url, headers=config['headers'], params=params)
            res.raise_for_status()
            expiries = sorted(res.json()["data"], key=lambda x: datetime.strptime(x["expiry"], "%Y-%m-%d"))
            today = datetime.now()

            # Find next Thursday expiry
            next_expiry = None
            for expiry in expiries:
                expiry_dt = datetime.strptime(expiry["expiry"], "%Y-%m-%d")
                # Thursday is weekday 3. Ensure expiry_dt is today or in the future.
                if expiry_dt.weekday() == 3 and expiry_dt.date() >= today.date():
                    next_expiry = expiry["expiry"]
                    break

            if next_expiry:
                config['expiry_date'] = next_expiry
            else:
                logger.warning(f"No upcoming Thursday expiry found. Defaulting to current date: {today.strftime('%Y-%m-%d')}")
                # Fallback to the nearest available expiry if no Thursday or current date doesn't have options
                if expiries:
                    config['expiry_date'] = expiries[0]["expiry"]
                    logger.warning(f"Falling back to nearest available expiry: {config['expiry_date']}")
                else:
                    config['expiry_date'] = today.strftime("%Y-%m-%d")
                    logger.error("No expiries found at all. Defaulting to current date.")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching expiries in get_config: {e.response.status_code} - {e.response.text}")
        config['expiry_date'] = datetime.now().strftime("%Y-%m-%d")  # Fallback
    except httpx.RequestError as e:
        logger.error(f"Network error fetching expiries in get_config: {e}")
        config['expiry_date'] = datetime.now().strftime("%Y-%m-%d")  # Fallback
    except Exception as e:
        logger.error(f"Unexpected exception in get_config (expiry fetch): {e}")
        config['expiry_date'] = datetime.now().strftime("%Y-%m-%d")  # Fallback

    return config

# --- Data Fetching and Calculation Functions ---
async def fetch_option_chain_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetches the option chain for the configured instrument and expiry date."""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{config['base_url']}/option/chain"
            params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
            res = await client.get(url, headers=config['headers'], params=params)
            res.raise_for_status()
            return res.json()["data"]
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching option chain: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching option chain: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching option chain: {e}")
        raise HTTPException(status_code=500, detail=f"Network error fetching option chain: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected exception in fetch_option_chain_data: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching option chain: {str(e)}")

def extract_seller_metrics(option_chain: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
    """Extracts key metrics for option sellers from the option chain."""
    try:
        # Filter out strikes with missing data to prevent errors
        valid_options = [opt for opt in option_chain if opt.get("call_options") and opt.get("put_options") and \
                         opt["call_options"].get("market_data") and opt["put_options"].get("market_data") and \
                         opt["call_options"]["market_data"].get("ltp") is not None and opt["put_options"]["market_data"].get("ltp") is not None and \
                         opt["call_options"].get("option_greeks") and opt["put_options"].get("option_greeks") and \
                         opt["call_options"]["option_greeks"].get("iv") is not None and opt["put_options"]["option_greeks"].get("iv") is not None]

        if not valid_options:
            logger.warning("No valid options found in chain for seller metrics extraction.")
            return {}

        atm_strike_info = min(valid_options, key=lambda x: abs(x["strike_price"] - spot_price))

        call_atm = atm_strike_info["call_options"]
        put_atm = atm_strike_info["put_options"]

        return {
            "atm_strike": atm_strike_info["strike_price"],
            "straddle_price": call_atm["market_data"]["ltp"] + put_atm["market_data"]["ltp"],
            "avg_iv": (call_atm["option_greeks"]["iv"] + put_atm["option_greeks"]["iv"]) / 2,
            "theta": (call_atm["option_greeks"].get("theta", 0.0) or 0.0) + (put_atm["option_greeks"].get("theta", 0.0) or 0.0),
            "vega": (call_atm["option_greeks"].get("vega", 0.0) or 0.0) + (put_atm["option_greeks"].get("vega", 0.0) or 0.0),
            "delta": (call_atm["option_greeks"].get("delta", 0.0) or 0.0) + (put_atm["option_greeks"].get("delta", 0.0) or 0.0),
            "gamma": (call_atm["option_greeks"].get("gamma", 0.0) or 0.0) + (put_atm["option_greeks"].get("gamma", 0.0) or 0.0),
            "pop": ((call_atm["option_greeks"].get("pop", 0.0) or 0.0) + (put_atm["option_greeks"].get("pop", 0.0) or 0.0)) / 2,
        }
    except Exception as e:
        logger.error(f"Exception in extract_seller_metrics for spot {spot_price}: {e}")
        return {}

def calculate_market_metrics(option_chain: List[Dict[str, Any]], expiry_date: str) -> Dict[str, Any]:
    """Calculates broader market metrics like Days to Expiry, PCR, and Max Pain."""
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        days_to_expiry = max(0, days_to_expiry)  # Ensure it's not negative

        call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain if opt.get("call_options") and opt["call_options"].get("market_data") and opt["call_options"]["market_data"].get("oi") is not None)
        put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain if opt.get("put_options") and opt["put_options"].get("market_data") and opt["put_options"]["market_data"].get("oi") is not None)

        pcr = put_oi / call_oi if call_oi != 0 else 0

        strikes = sorted(list(set(opt["strike_price"] for opt in option_chain)))
        max_pain_strike = 0
        min_pain = float('inf')

        # Filter out strikes that might not have full data for max pain calc
        valid_strikes_for_pain = [opt for opt in option_chain if \
                                  opt.get("call_options") and opt["call_options"].get("market_data") and opt["call_options"]["market_data"].get("oi") is not None and \
                                  opt.get("put_options") and opt["put_options"].get("market_data") and opt["put_options"]["market_data"].get("oi") is not None]

        for strike in strikes:
            pain_at_strike = 0
            for opt in valid_strikes_for_pain:
                # Pain for call writers: loss if market closes above their strike
                pain_at_strike += max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"]
                # Pain for put writers: loss if market closes below their strike
                pain_at_strike += max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]

            if pain_at_strike < min_pain:
                min_pain = pain_at_strike
                max_pain_strike = strike

        return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain_strike}
    except Exception as e:
        logger.error(f"Exception in market_metrics: {e}")
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

async def fetch_india_vix_and_nifty_spot(config: Dict[str, Any]) -> tuple[float, float]:
    """
    Fetches India VIX and Nifty 50 spot from Upstox API.
    """
    try:
        vix_instrument_key_for_request = "NSE_INDEX|India VIX"
        nifty_instrument_key_for_request = "NSE_INDEX|Nifty 50"
        # Upstox API returns this format, note the colon
        vix_instrument_key_for_response_parsing = "NSE_INDEX:India VIX"
        nifty_instrument_key_for_response_parsing = "NSE_INDEX:Nifty 50"

        async with httpx.AsyncClient() as client:
            url = f"{config['base_url']}/market-quote/quotes"
            params = {"instrument_key": f"{vix_instrument_key_for_request},{nifty_instrument_key_for_request}"}
            logger.info(f"Attempting to fetch India VIX and Nifty Spot from: {url} with params: {params}")
            res = await client.get(url, headers=config['headers'], params=params)
            res.raise_for_status()
            data = res.json()

            vix_ltp = None
            nifty_ltp = None

            if data and data.get("data"):
                if data["data"].get(vix_instrument_key_for_response_parsing):
                    vix_ltp = data["data"][vix_instrument_key_for_response_parsing].get("last_price")
                if data["data"].get(nifty_instrument_key_for_response_parsing):
                    nifty_ltp = data["data"][nifty_instrument_key_for_response_parsing].get("last_price")

            if vix_ltp is not None and nifty_ltp is not None:
                logger.info(f"Successfully fetched India VIX: {vix_ltp} and Nifty Spot: {nifty_ltp}")
                return vix_ltp, nifty_ltp
            else:
                missing_data = []
                if vix_ltp is None:
                    missing_data.append("India VIX")
                if nifty_ltp is None:
                    missing_data.append("Nifty Spot")
                logger.error(f"Missing data for {', '.join(missing_data)} in API response. Full response: {data}")
                raise ValueError(f"Required data for {', '.join(missing_data)} not available in API response.")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching India VIX and Nifty Spot: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Failed to fetch India VIX and Nifty Spot due to HTTP error: {e.response.status_code}. Response: {e.response.text}") from e
    except httpx.RequestError as e:
        logger.error(f"Network error fetching India VIX and Nifty Spot: {e}")
        raise RuntimeError(f"Failed to fetch India VIX and Nifty Spot due to network error: {e}") from e
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing India VIX and Nifty Spot data from API response: {e}")
        raise RuntimeError(f"Failed to parse India VIX and Nifty Spot data: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching India VIX and Nifty Spot: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}") from e

async def calculate_volatility(config: Dict[str, Any], seller_avg_iv: float) -> tuple[float, float, float]:
    """Calculates Historical Volatility, GARCH Volatility, and IV-RV spread."""
    try:
        async with httpx.AsyncClient() as client:
            nifty_response = await client.get(config['nifty_url'])
            nifty_response.raise_for_status()
            df = pd.read_csv(BytesIO(nifty_response.content))

        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y")
        df = df.sort_values('Date').set_index('Date')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)  # Drop rows where 'Close' couldn't be converted

        if df.empty or len(df) < 2:  # Need at least 2 points for a return
            logger.warning("Not enough historical data for volatility calculation. Returning zeros.")
            return 0.0, 0.0, 0.0

        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

        if df.empty:
            logger.warning("No valid log returns after cleaning. Returning zeros.")
            return 0.0, 0.0, 0.0

        # Historical Volatility (7-day)
        if len(df["Log_Returns"]) >= 7:
            hv_7 = np.std(df["Log_Returns"].tail(7)) * np.sqrt(252) * 100
        else:
            logger.warning("Less than 7 days of log returns for HV_7. Using all available data.")
            hv_7 = np.std(df["Log_Returns"]) * np.sqrt(252) * 100 if len(df["Log_Returns"]) > 0 else 0.0

        # GARCH Model
        garch_7d = 0.0
        if len(df["Log_Returns"]) > 20:  # A common heuristic for minimum data points for GARCH
            try:
                # Scale returns for GARCH to prevent numerical issues with very small numbers
                model = arch_model(df["Log_Returns"] * 100, vol="Garch", p=1, q=1, dist='StudentsT')
                res = model.fit(disp="off", show_warning=False)
                forecast = res.forecast(horizon=7)
                if not forecast.variance.empty:
                    # Convert back from scaled returns, then annualize
                    garch_7d = np.mean(np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252))
                else:
                    logger.warning("GARCH forecast variance is empty.")
            except Exception as e:
                logger.warning(f"GARCH model fitting failed: {e}. GARCH volatility set to 0.")
        else:
            logger.warning("Not enough data for GARCH model. GARCH volatility set to 0.")

        iv_rv_spread = round(seller_avg_iv - hv_7, 2)
        return hv_7, garch_7d, iv_rv_spread

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching Nifty historical data: {e.response.status_code} - {e.response.text}")
        return 0.0, 0.0, 0.0
    except httpx.RequestError as e:
        logger.error(f"Network error fetching Nifty historical data: {e}")
        return 0.0, 0.0, 0.0
    except Exception as e:
        logger.error(f"Exception in calculate_volatility: {e}")
        return 0.0, 0.0, 0.0

_xgboost_model = None  # Global variable to store the loaded model

async def load_xgboost_model():
    """Loads the pre-trained XGBoost model for volatility prediction."""
    global _xgboost_model
    if _xgboost_model is not None:
        logger.info("XGBoost model already loaded.")
        return _xgboost_model

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(XG_BOOST_MODEL_URL)
            response.raise_for_status()
        _xgboost_model = pickle.load(BytesIO(response.content))
        logger.info("XGBoost model loaded successfully.")
        return _xgboost_model
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching XGBoost model: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Network error fetching XGBoost model: {e}")
        return None
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling XGBoost model: {e}. Model file might be corrupt or incompatible.")
        return None
    except Exception as e:
        logger.error(f"Unexpected exception in load_xgboost_model: {e}")
        return None

def predict_xgboost_volatility(model: Any, atm_iv: float, realized_vol: float, ivp: float, pcr: float, vix: float, days_to_expiry: int, garch_vol: float) -> float:
    """Predicts volatility using the loaded XGBoost model."""
    if model is None:
        logger.warning("XGBoost model not loaded, cannot predict volatility. Returning 0.0.")
        return 0.0
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
        prediction = model.predict(features)[0]
        return round(float(prediction), 2)
    except Exception as e:
        logger.error(f"Exception in predict_xgboost_volatility: {e}")
        return 0.0

async def calculate_ivp(config: Dict[str, Any], current_atm_iv: float) -> float:
    """Calculates Implied Volatility Percentile (IVP)."""
    try:
        async with httpx.AsyncClient() as client:
            ivp_response = await client.get(config['ivp_url'])
            ivp_response.raise_for_status()
            df_iv = pd.read_csv(BytesIO(ivp_response.content))

        df_iv.columns = df_iv.columns.str.strip()
        df_iv['Date'] = pd.to_datetime(df_iv['Date'])

        if 'ATM_IV' not in df_iv.columns:
            logger.error("IVP CSV must contain an 'ATM_IV' column. Cannot calculate IVP.")
            return 0.0

        historical_ivs = df_iv['ATM_IV'].dropna().values

        if len(historical_ivs) < 10:  # Need sufficient historical data for percentile
            logger.warning(f"Not enough historical data ({len(historical_ivs)} points) in IVP CSV for percentile calculation. Returning 0.0.")
            return 0.0

        percentile = np.mean(current_atm_iv > historical_ivs) * 100
        return round(percentile, 2)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching IVP historical data: {e.response.status_code} - {e.response.text}")
        return 0.0
    except httpx.RequestError as e:
        logger.error(f"Network error fetching IVP historical data: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Exception in calculate_ivp: {e}")
        return 0.0

def calculate_iv_skew_slope(full_chain_df: pd.DataFrame) -> float:
    """Calculates the slope of the IV skew."""
    try:
        if full_chain_df.empty or "Strike" not in full_chain_df.columns or "IV Skew" not in full_chain_df.columns:
            logger.warning("Full chain DataFrame is empty or missing required columns for IV skew slope calculation. Returning 0.0.")
            return 0.0

        df_filtered = full_chain_df[["Strike", "IV Skew"]].dropna()

        if len(df_filtered) < 2:
            logger.warning("Not enough valid data points for linear regression on IV Skew. Returning 0.0.")
            return 0.0

        slope, _, _, _, _ = linregress(df_filtered["Strike"], df_filtered["IV Skew"])
        return round(slope, 4)
    except Exception as e:
        logger.error(f"Exception in calculate_iv_skew_slope: {e}")
        return 0.0

def calculate_regime(atm_iv: float, ivp: float, realized_vol: float, garch_vol: float, straddle_price: float, spot_price: float, pcr: float, vix: float, iv_skew_slope: float) -> tuple[float, str, str, str]:
    """Determines the current market volatility regime based on various metrics."""
    expected_move = (straddle_price / spot_price) * 100 if spot_price else 0
    vol_spread = atm_iv - realized_vol

    regime_score = 0

    regime_score += 10 if ivp > 80 else (-10 if ivp < 20 else 0)
    regime_score += 10 if vol_spread > 10 else (-10 if vol_spread < -10 else 0)
    regime_score += 10 if vix > 20 else (-10 if vix < 10 else 0)
    regime_score += 5 if pcr > 1.2 else (-5 if pcr < 0.8 else 0)
    regime_score += 5 if abs(iv_skew_slope) > 0.001 else 0  # Significant skew indicates potential for moves
    regime_score += 10 if expected_move > 0.05 else (-10 if expected_move < 0.02 else 0)
    regime_score += 5 if garch_vol > realized_vol * 1.2 else (-5 if garch_vol < realized_vol * 0.8 else 0) # GARCH predicting higher/lower than realized

    if regime_score > 20:
        return regime_score, "High Vol Trend 🔥", "Market in high volatility â€” ideal for premium selling.", "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
    elif regime_score > 10:
        return regime_score, "Elevated Volatility ⚡", "Above-average volatility â€” favor range-bound strategies.", "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
    elif regime_score > -10:
        return regime_score, "Neutral Volatility 🙂", "Balanced market â€” flexible strategy selection.", "IV and RV aligned, with moderate PCR and skew."
    else:
        return regime_score, "Low Volatility 📉", "Low volatility â€” cautious selling or long vega plays.", "Low IVP, tight straddle, and low VIX suggest limited movement."


async def suggest_strategy(regime_label: str, ivp: float, iv_minus_rv: float, days_to_expiry: int, expiry_date: str, straddle_price: float, spot_price: float, config: Dict[str, Any]) -> tuple[List[str], str, Optional[str]]:
    """Suggests optimal trading strategies based on market conditions."""
    strategies = []
    rationale = []
    event_warning = None
    event_impact_score = 0

    try:
        async with httpx.AsyncClient() as client:
            events_response = await client.get(config['event_url'])
            events_response.raise_for_status()
            event_df = pd.read_csv(BytesIO(events_response.content))

        event_df.columns = event_df.columns.str.strip()
        if 'Date' in event_df.columns and 'Time' in event_df.columns:
            current_year = datetime.now().year
            event_df['CombinedDateTimeStr'] = event_df.apply(
                lambda row: f"{row['Date']}-{current_year} {row['Time']}" if pd.notna(row['Date']) and pd.notna(row['Time']) else np.nan,
                axis=1
            )
            event_df['Datetime'] = pd.to_datetime(event_df['CombinedDateTimeStr'], format='%d-%b-%Y %H:%M', errors='coerce')
            event_df = event_df.drop(columns=['CombinedDateTimeStr'])
        else:
            logger.error("Missing 'Date' or 'Time' columns in upcoming_events.csv. Cannot process event dates.")
            event_df['Datetime'] = pd.NaT

        event_window = 3 if ivp > 80 else 2 # Days before expiry to consider as "near"
        high_impact_event_near = False
        current_expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")

        for _, row in event_df.iterrows():
            if pd.isna(row["Datetime"]):
                continue
            event_dt = row["Datetime"]
            level = str(row["Classification"]).strip()
            days_until_event = (event_dt.date() - datetime.now().date()).days
            days_from_event_to_expiry = (current_expiry_dt.date() - event_dt.date()).days

            if level == "High" and (0 <= days_until_event <= event_window or (days_from_event_to_expiry >=0 and days_from_event_to_expiry <= event_window)):
                high_impact_event_near = True
                if pd.notnull(row.get("Forecast")) and pd.notnull(row.get("Prior")):
                    try:
                        forecast = float(str(row["Forecast"]).replace('%', '').strip())
                        prior = float(str(row["Prior"]).replace('%', '').strip())
                        if abs(forecast - prior) > 0.5: # Significant deviation
                            event_impact_score += 1
                    except ValueError:
                        logger.warning(f"Could not parse Forecast/Prior for event: {row.get('Event')}")

        if high_impact_event_near:
            event_warning = f"⚠️ High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
            rationale.append("Upcoming high-impact event suggests cautious, defined-risk approach.")
            if event_impact_score > 0:
                rationale.append(f"High-impact events with significant forecast deviations detected ({event_impact_score} events).")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching event data: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching event data: {e}")
    except Exception as e:
        logger.error(f"Exception in fetching/processing event data: {e}")

    expected_move_pct = (straddle_price / spot_price) * 100 if spot_price else 0

    # Strategy suggestions based on regime
    if "High Vol Trend" in regime_label:
        strategies = ["Iron Fly", "Wide Strangle"]
        rationale.append("Strong IV premium â€” neutral strategies for premium capture.")
    elif "Elevated Volatility" in regime_label:
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average â€” range-bound strategies offer favorable reward-risk.")
    elif "Neutral Volatility" in regime_label:
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced â€” slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Fly"]
            rationale.append("Tight expiry â€” quick theta-based capture via short Iron Fly.")
    elif "Low Volatility" in regime_label:
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry â€” benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV â€” premium collection favorable but monitor for breakout risk.")

    # Adjust strategies if high-impact event is near, prioritizing defined risk
    if high_impact_event_near:
        defined_risk_strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
        if defined_risk_strategies:
            strategies = defined_risk_strategies
        else:
            strategies = ["Iron Condor"] # Fallback to a generic defined risk
        rationale.append("Forcing defined-risk strategy due to high-impact event proximity.")

    # Additional rationale based on IVP and IV-RV spread
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) â€” ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) â€” avoid unhedged selling.")

    rationale.append(f"Expected market move based on straddle price: Â±{expected_move_pct:.2f}%.")

    return strategies, " | ".join(rationale), event_warning

def find_option_by_strike(option_chain: List[Dict[str, Any]], strike: float, option_type: str, tolerance: float = 0.01) -> Optional[Dict[str, Any]]:
    """Helper to find a specific option by strike and type with a tolerance."""
    for opt in option_chain:
        if abs(opt["strike_price"] - strike) < tolerance:
            if option_type == "CE" and "call_options" in opt and opt["call_options"].get("market_data") and opt["call_options"]["market_data"].get("ltp") is not None:
                return opt["call_options"]
            elif option_type == "PE" and "put_options" in opt and opt["put_options"].get("market_data") and opt["put_options"]["market_data"].get("ltp") is not None:
                return opt["put_options"]
    logger.warning(f"No valid option found for strike {strike} {option_type}")
    return None

def get_dynamic_wing_distance(ivp: float, straddle_price: float) -> int:
    """Calculates dynamic wing distance for multi-leg strategies."""
    # Modified to be less extreme for Iron Fly, and more responsive to IVP
    if ivp >= 80:
        multiplier = 0.30  # Slightly reduced for very high IV
    elif ivp <= 20:
        multiplier = 0.20
    else:
        multiplier = 0.25  # Default
    raw_distance = straddle_price * multiplier
    return int(round(raw_distance / 50.0)) * 50  # Round to nearest 50 for Nifty

# --- Strategy Detail Calculation Functions ---
def _iron_fly_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """Calculates details for Iron Fly strategy."""
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    atm_strike = atm_info["strike_price"]
    straddle_price = (atm_info["call_options"]["market_data"]["ltp"] + atm_info["put_options"]["market_data"]["ltp"]) if atm_info["call_options"] and atm_info["put_options"] else 0.0

    # Using a high IVP as input for wider wings for Iron Fly (as it's often used in high IV regimes)
    wing_distance = get_dynamic_wing_distance(80, straddle_price)

    ce_short_opt = find_option_by_strike(option_chain, atm_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, atm_strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, atm_strike + wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, atm_strike - wing_distance, "PE")

    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        logger.error("Invalid options for Iron Fly: One or more legs not found or invalid data.")
        return None

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_long_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Iron Fly", "strikes": [atm_strike - wing_distance, atm_strike, atm_strike + wing_distance], "orders": orders}

def _iron_condor_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """Calculates details for Iron Condor strategy."""
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    atm_strike = atm_info["strike_price"]
    straddle_price = (atm_info["call_options"]["market_data"]["ltp"] + atm_info["put_options"]["market_data"]["ltp"]) if atm_info["call_options"] and atm_info["put_options"] else 0.0

    # For Iron Condor, typically wider, so maybe a slightly lower IVP for calculation if not tied to actual IVP
    short_wing_distance = get_dynamic_wing_distance(50, straddle_price) # Moderate IVP for wider short wings
    long_wing_distance = int(round(short_wing_distance * 1.5 / 50)) * 50 # 1.5x the short wing distance

    ce_short_strike = atm_strike + short_wing_distance
    pe_short_strike = atm_strike - short_wing_distance
    ce_long_strike = atm_strike + long_wing_distance
    pe_long_strike = atm_strike - long_wing_distance

    ce_short_opt = find_option_by_strike(option_chain, ce_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, pe_short_strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, ce_long_strike, "CE")
    pe_long_opt = find_option_by_strike(option_chain, pe_long_strike, "PE")

    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        logger.error("Invalid options for Iron Condor: One or more legs not found or invalid data.")
        return None

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_long_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Iron Condor", "strikes": [pe_long_strike, pe_short_strike, ce_short_strike, ce_long_strike], "orders": orders}

def _jade_lizard_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """Calculates details for Jade Lizard strategy."""
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))

    # These strike offsets might need to be dynamic or configured based on market conditions
    call_short_strike = atm_info["strike_price"] + 100 # Slightly OTM Call
    pe_short_strike = atm_info["strike_price"] - 50  # Closer OTM Put
    pe_long_strike = atm_info["strike_price"] - 150 # Further OTM Put for hedge

    # Ensure strikes are rounded to nearest 50 for Nifty
    call_short_strike = int(round(call_short_strike / 50.0)) * 50
    pe_short_strike = int(round(pe_short_strike / 50.0)) * 50
    pe_long_strike = int(round(pe_long_strike / 50.0)) * 50

    ce_short_opt = find_option_by_strike(option_chain, call_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, pe_short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, pe_long_strike, "PE")

    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        logger.error("Invalid options for Jade Lizard: One or more legs not found or invalid data.")
        return None

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Jade Lizard", "strikes": [pe_long_strike, pe_short_strike, call_short_strike], "orders": orders}

def _straddle_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """Calculates details for Straddle strategy."""
    atm_info = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    atm_strike = atm_info["strike_price"]

    ce_short_opt = find_option_by_strike(option_chain, atm_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, atm_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for Straddle: One or both legs not found or invalid data.")
        return None

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Straddle", "strikes": [atm_strike, atm_strike], "orders": orders}

def _calendar_spread_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """
    Calculates details for Calendar Spread strategy.
    NOTE: This implementation requires fetching option chains for multiple expiry dates, which is not directly
    supported by the current single-expiry `fetch_option_chain_data` flow. It will always return None with a warning.
    """
    logger.error("Calendar Spread calculation requires fetching option chains for multiple expiry dates (e.g., near and far), which is not fully implemented in this single-expiry flow. Returning None.")
    return None

def _bull_put_spread_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """Calculates details for Bull Put Spread strategy."""
    # Find a strike roughly 1-2 standard deviations below spot price for the short put,
    # then go further down for the long put. Using fixed distances for simplicity.
    short_put_strike = spot_price - 100 # Sell slightly OTM Put
    long_put_strike = spot_price - 200 # Buy further OTM Put for protection

    # Round to nearest 50 for Nifty strikes
    short_put_strike = int(round(short_put_strike / 50.0)) * 50
    long_put_strike = int(round(long_put_strike / 50.0)) * 50

    pe_short_opt = find_option_by_strike(option_chain, short_put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_put_strike, "PE")

    if not all([pe_short_opt, pe_long_opt]):
        logger.error("Invalid options for Bull Put Spread: One or both legs not found or invalid data.")
        return None

    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_long_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Bull Put Spread", "strikes": [long_put_strike, short_put_strike], "orders": orders}

def _wide_strangle_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """Calculates details for Wide Strangle strategy."""
    # Fixed distances, could be dynamic based on volatility or expected move
    call_short_strike = spot_price + 200
    put_short_strike = spot_price - 200

    call_short_strike = int(round(call_short_strike / 50.0)) * 50
    put_short_strike = int(round(put_short_strike / 50.0)) * 50

    ce_short_opt = find_option_by_strike(option_chain, call_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_short_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for Wide Strangle: One or both legs not found or invalid data.")
        return None

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "Wide Strangle", "strikes": [put_short_strike, call_short_strike], "orders": orders}

def _atm_strangle_calc(option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int) -> Optional[Dict[str, Any]]:
    """Calculates details for ATM Strangle strategy."""
    # Closer to ATM, fixed distances
    call_short_strike = spot_price + 50
    put_short_strike = spot_price - 50

    call_short_strike = int(round(call_short_strike / 50.0)) * 50
    put_short_strike = int(round(put_short_strike / 50.0)) * 50

    ce_short_opt = find_option_by_strike(option_chain, call_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_short_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        logger.error("Invalid options for ATM Strangle: One or both legs not found or invalid data.")
        return None

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": ce_short_opt["market_data"]["ltp"]},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "D", "validity": "DAY", "order_type": "LIMIT", "price": pe_short_opt["market_data"]["ltp"]}
    ]
    return {"strategy": "ATM Strangle", "strikes": [put_short_strike, call_short_strike], "orders": orders}

async def get_strategy_details(strategy_name: str, option_chain: List[Dict[str, Any]], spot_price: float, config: Dict[str, Any], lots: int = 1) -> Optional[Dict[str, Any]]:
    """
    Retrieves detailed information (legs, premiums, max PnL) for a specific option strategy.
    """
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
        logger.warning(f"Strategy {strategy_name} not supported.")
        return None

    try:
        detail = func_map[strategy_name](option_chain, spot_price, config, lots=lots)
        if detail is None:
            return None # Strategy specific calculation returned None
    except Exception as e:
        logger.error(f"Error calculating {strategy_name} details: {e}")
        return None

    if detail:
        # Update LTPs in orders and calculate premiums, max loss/profit
        ltp_map = {}
        for opt in option_chain:
            if opt.get("call_options") and opt["call_options"].get("instrument_key"):
                ltp_map[opt["call_options"]["instrument_key"]] = opt["call_options"]["market_data"].get("ltp", 0.0)
            if opt.get("put_options") and opt["put_options"].get("instrument_key"):
                ltp_map[opt["put_options"]["instrument_key"]] = opt["put_options"]["market_data"].get("ltp", 0.0)

        updated_orders = []
        premium = 0.0
        for order in detail["orders"]:
            key = order["instrument_key"]
            ltp = ltp_map.get(key, 0.0)

            # Ensure price is set for order placement, especially for LIMIT orders
            if order.get("order_type") == "LIMIT" and order.get("price") is None:
                order["price"] = ltp # Default to LTP for limit orders if not specified
            updated_orders.append({**order, "current_price": ltp})

            # Calculate premium based on transaction type and current price
            if order["transaction_type"] == "SELL":
                premium += ltp * order["quantity"]
            else: # BUY
                premium -= ltp * order["quantity"]

        detail["orders"] = updated_orders
        detail["premium"] = premium / config["lot_size"] # Premium per lot
        detail["premium_total"] = premium # Total premium received/paid

        # Calculate Max Loss and Max Profit
        detail["max_loss"] = float("inf")
        detail["max_profit"] = float("inf")

        wing_width = 0
        if strategy_name == "Iron Fly":
            if len(detail["strikes"]) == 3:
                wing_width = abs(detail["strikes"][1] - detail["strikes"][0]) # Distance from ATM short to long wing
        elif strategy_name == "Iron Condor":
            if len(detail["strikes"]) == 4:
                put_wing_width = abs(detail["strikes"][0] - detail["strikes"][1])
                call_wing_width = abs(detail["strikes"][2] - detail["strikes"][3])
                wing_width = max(put_wing_width, call_wing_width) # Max of call or put spread
        elif strategy_name == "Jade Lizard":
            if len(detail["strikes"]) == 3:
                wing_width = abs(detail["strikes"][0] - detail["strikes"][1]) # Put spread width
        elif strategy_name == "Bull Put Spread":
            if len(detail["strikes"]) == 2:
                wing_width = abs(detail["strikes"][0] - detail["strikes"][1])

        # Apply general P&L calculations based on credit/debit and defined/undefined risk
        if detail["premium_total"] >= 0: # Credit strategies
            detail["max_profit"] = detail["premium_total"]
            if wing_width > 0: # Defined risk credit strategies (Iron Fly, Iron Condor, Jade Lizard, Bull Put Spread)
                detail["max_loss"] = (wing_width * config["lot_size"] * lots) - detail["premium_total"]
                if detail["max_loss"] < 0: # Ensure max_loss is non-negative, can happen if premium is very high
                    detail["max_loss"] = 0
            else: # Undefined risk credit strategies (Straddle, Strangle)
                detail["max_loss"] = float("inf") # Unlimited loss potential
        else: # Debit strategies (e.g., Calendar Spread, if implemented)
            detail["max_profit"] = float("inf") # Theoretically unlimited, practically capped by other factors
            detail["max_loss"] = abs(detail["premium_total"]) # Max loss is the debit paid

        # Calculate margin using a placeholder for now
        detail["margin"] = await calculate_strategy_margin(config, detail) # Call the new margin calculation

        logger.info(f"Calculated details for {strategy_name}: Premium={detail['premium']:.2f}, Max Profit={detail['max_profit']:.2f}, Max Loss={detail['max_loss']:.2f}, Margin={detail['margin']:.2f}")
    return detail

async def calculate_strategy_margin(config: Dict[str, Any], strategy_details: Dict[str, Any]) -> float:
    """
    Calculates the estimated margin required for a given strategy.
    This is a simplified estimation. A real-world calculation would involve
    Upstox's margin APIs or a more complex SPAN/VAR margin simulator.
    """
    total_margin = 0.0
    lot_size = config["lot_size"]

    # Basic estimation: sum of margins for each leg.
    # This is a very rough estimate; actual broker margins are complex.
    # For defined risk strategies, max loss is a good proxy for margin if net credit.
    # For undefined risk, it's typically SPAN + Exposure margin.

    strategy_name = strategy_details["strategy"]
    premium_total = strategy_details["premium_total"]
    max_loss = strategy_details["max_loss"]
    orders = strategy_details["orders"]

    if "Iron" in strategy_name or "Spread" in strategy_name or "Lizard" in strategy_name:
        # For defined risk strategies, margin is often close to the max loss
        # minus the premium received (if credit) or just the max loss (if debit)
        # However, brokers might block more. A common rough estimate for credit spreads
        # is (difference in strikes * lot_size) for one lot.
        if max_loss != float('inf'):
            total_margin = max_loss + abs(premium_total) # Simplified: max_loss + total premium
            # Add a buffer
            total_margin *= 1.05 # Add 5% buffer
        else:
            # Fallback for defined risk if max_loss is somehow inf (shouldn't happen)
            # Estimate based on short options' margins if max_loss is not clear
            for order in orders:
                if order["transaction_type"] == "SELL":
                    total_margin += order["current_price"] * order["quantity"] * 0.15 # Rough 15% of premium
    else: # Straddle, Strangle (undefined risk)
        # For undefined risk strategies, margin is substantial.
        # It's usually a percentage of the underlying value or based on SPAN/VAR.
        # Here, we'll use a very rough multiplier on premium or a fixed amount per lot.
        # This is highly speculative and requires actual broker margin APIs for accuracy.
        # Let's assume a fixed high margin per lot for now.
        estimated_margin_per_lot = 120000 # Example for Nifty Straddle/Strangle
        total_margin = strategy_details["lots"] * estimated_margin_per_lot

    # Ensure margin is never negative
    return max(0.0, round(total_margin, 2))


async def evaluate_full_risk(trades_data: List[TradeLogEntry], config: Dict[str, Any], regime_label: str, vix: float) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates the overall risk of the trading portfolio.
    `trades_data` should represent active/open positions for a realistic risk assessment.
    """
    try:
        # Convert list of Pydantic models to DataFrame
        trades_df = pd.DataFrame([t.dict() for t in trades_data])

        total_funds = config.get('total_funds', 2000000)
        daily_risk_limit = config['daily_risk_limit_pct'] * total_funds
        weekly_risk_limit = config['weekly_risk_limit_pct'] * total_funds

        # Dynamic Max Drawdown based on VIX
        if vix > 25:
            max_drawdown_pct = 0.06  # Higher drawdown tolerance in very high VIX
        elif vix > 20:
            max_drawdown_pct = 0.05
        elif vix > 12:
            max_drawdown_pct = 0.03
        else:
            max_drawdown_pct = 0.02
        max_drawdown = max_drawdown_pct * total_funds

        strategy_summary_records = []
        total_cap_used = 0.0
        total_risk_on_table = 0.0
        total_realized_pnl = 0.0  # This would typically be 0 for *active* trades or PnL_Day
        total_unrealized_pnl = 0.0
        total_vega_exposure = 0.0
        flags = []

        if trades_df.empty:
            logger.info("No active trades to evaluate risk for.")
            # Populate with default values for an empty portfolio
            strategy_summary_records.append({
                "Strategy": "None", "Capital Used": 0.0, "Cap Limit": total_funds, "% Used": 0.0,
                "Potential Risk": 0.0, "Risk Limit": total_funds * 0.01, "Realized P&L": 0.0,
                "Unrealized P&L": 0.0, "Vega": 0.0, "Risk OK?": "✅"
            })
        else:
            for _, row in trades_df.iterrows():
                strat = row["strategy"]
                capital_used = row.get("capital_used", 0.0)
                potential_risk = row.get("potential_loss", 0.0)  # For active trades
                realized_pnl = row.get("realized_pnl", 0.0)
                unrealized_pnl = row.get("unrealized_pnl", 0.0)
                sl_hit = row.get("sl_hit", False)
                vega = row.get("vega", 0.0)

                cfg = config['risk_config'].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
                risk_factor = 1.2 if "High Vol Trend" in regime_label else (0.8 if "Low Volatility" in regime_label else 1.0)
                max_cap = cfg["capital_pct"] * total_funds
                max_risk_per_trade = cfg["risk_per_trade_pct"] * max_cap * risk_factor

                risk_ok = "✅" if potential_risk <= max_risk_per_trade else "❌"

                strategy_summary_records.append({
                    "Strategy": strat,
                    "Capital Used": round(capital_used, 2),
                    "Cap Limit": round(max_cap, 2),
                    "% Used": round((capital_used / max_cap * 100) if max_cap else 0, 2),
                    "Potential Risk": round(potential_risk, 2),
                    "Risk Limit": round(max_risk_per_trade, 2),
                    "Realized P&L": round(realized_pnl, 2),
                    "Unrealized P&L": round(unrealized_pnl, 2),
                    "Vega": round(vega, 2),
                    "Risk OK?": risk_ok
                })

                total_cap_used += capital_used
                total_risk_on_table += potential_risk
                total_realized_pnl += realized_pnl
                total_unrealized_pnl += unrealized_pnl
                total_vega_exposure += vega

                if risk_ok == "❌":
                    flags.append(f"❌ {strat} exceeded its per-trade risk limit (Potential Risk: {potential_risk:.2f}, Limit: {max_risk_per_trade:.2f})")
                if sl_hit:
                    flags.append(f"⚠️ {strat} hit stop-loss. Review for potential revenge trading or strategy effectiveness.")

        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0 # Simplified
        exposure_pct = round(total_cap_used / total_funds * 100, 2) if total_funds else 0
        risk_pct = round(total_risk_on_table / total_funds * 100, 2) if total_funds else 0
        dd_pct = round(net_dd / total_funds * 100, 2) if total_funds else 0

        if total_risk_on_table > daily_risk_limit:
            flags.append(f"❌ Total portfolio risk ({total_risk_on_table:.2f}) exceeds daily limit ({daily_risk_limit:.2f}).")
        if net_dd > max_drawdown:
            flags.append(f"❌ Portfolio drawdown ({net_dd:.2f}) exceeds maximum allowed ({max_drawdown:.2f}).")

        portfolio_summary_data = {
            "Total Funds": round(total_funds, 2),
            "Capital Deployed": round(total_cap_used, 2),
            "Exposure Percent": exposure_pct,
            "Risk on Table": round(total_risk_on_table, 2),
            "Risk Percent": risk_pct,
            "Daily Risk Limit": round(daily_risk_limit, 2),
            "Weekly Risk Limit": round(weekly_risk_limit, 2),
            "Total Realized P&L": round(total_realized_pnl, 2),
            "Total Unrealized P&L": round(total_unrealized_pnl, 2),
            "Drawdown ₹": round(net_dd, 2),
            "Drawdown Percent": dd_pct,
            "Max Drawdown Allowed": round(max_drawdown, 2),
            "Total Vega Exposure": round(total_vega_exposure, 2),
            "Flags": flags if flags else ["✅ All risk parameters within limits."]
        }
        
        # Validate and convert to Pydantic model
        portfolio_summary = PortfolioSummaryResponse.parse_obj(portfolio_summary_data)


        logger.info("Full risk evaluation completed.")
        return strategy_summary_records, portfolio_summary
    except Exception as e:
        logger.error(f"Exception in evaluate_full_risk: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating full risk: {str(e)}")

async def get_funds_and_margin(config: Dict[str, Any]) -> Dict[str, float]:
    """Fetches funds and margin details from Upstox. (Placeholder for actual API call)"""
    # This function needs to interact with Upstox funds/margin API.
    # For now, providing dummy data.
    logger.warning("get_funds_and_margin is a placeholder. Implement actual Upstox API call.")
    try:
        async with httpx.AsyncClient() as client:
            url = f"{config['base_url']}/user/get-funds-and-margin" # Example endpoint, check Upstox docs
            res = await client.get(url, headers=config['headers'])
            res.raise_for_status()
            data = res.json()
            # Parse actual data from Upstox response
            return {
                "available_margin": data.get("available_margin", config["total_funds"]),
                "used_margin": data.get("used_margin", 0.0)
            }
    except Exception as e:
        logger.error(f"Failed to fetch funds/margin from Upstox: {e}. Using dummy data.")
        return {
            "available_margin": config["total_funds"] - 50000, # Example dummy
            "used_margin": 50000.0 # Example dummy
        }

def calculate_sharpe_ratio() -> float:
    """Calculates Sharpe Ratio. (Placeholder for actual calculation from trade data)"""
    logger.warning("calculate_sharpe_ratio is a placeholder. Implement actual calculation using trade logs.")
    # This would involve fetching historical PnL from trades, and a risk-free rate.
    # For now, return a dummy value.
    return 1.5 # Example dummy

async def place_multi_leg_orders(config: Dict[str, Any], orders: List[Dict[str, Any]]) -> bool:
    """Places multi-leg orders via Upstox API."""
    try:
        order_payload = [UpstoxOrderRequest(**order).dict(exclude_unset=True) for order in orders]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config['base_url']}/order/multi/place",
                json=order_payload,
                headers=config['headers']
            )
            response.raise_for_status()
            logger.info(f"Multi-leg order placement successful: {response.json()}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error placing multi-leg orders: {e.response.status_code} - {e.response.text}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Network error placing multi-leg orders: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error placing multi-leg orders: {e}")
        return False

async def create_gtt_order(config: Dict[str, Any], instrument_key: str, trigger_price: float, transaction_type: str, tag: Optional[str] = None) -> bool:
    """Creates a GTT order (Good Till Triggered). (Placeholder for actual Upstox API call)"""
    logger.warning("create_gtt_order is a placeholder. Implement actual Upstox GTT API call.")
    # This needs to be implemented using Upstox's GTT order API.
    # For now, it just logs a message.
    logger.info(f"Mock GTT order created for {instrument_key} at {trigger_price} ({transaction_type}) with tag: {tag}")
    return True

async def log_trade(trade_data: Dict[str, Any]):
    """Logs a trade. This will call your Supabase logging function."""
    status, result = await log_trade_to_supabase(trade_data)
    if status != 201:
        logger.error(f"Failed to log trade to Supabase: {result}")
    else:
        logger.info("Trade logged to Supabase successfully.")

async def add_journal_entry(journal_data: Dict[str, Any]):
    """Adds a journal entry. This will call your Supabase logging function."""
    status, result = await add_journal_to_supabase(journal_data)
    if status != 201:
        logger.error(f"Failed to add journal entry to Supabase: {result}")
        return False
    else:
        logger.info("Journal entry added to Supabase successfully.")
        return True

async def fetch_trade_data(config: Dict[str, Any], full_chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches live trade/position data from Upstox and integrates with market data.
    This is a placeholder and should fetch *live positions* from Upstox API.
    For this example, it will return an empty DataFrame or dummy data.
    """
    logger.warning("fetch_trade_data is a placeholder. Implement actual Upstox Positions API integration.")
    # Dummy live trade data for evaluation
    dummy_trades = [
        TradeLogEntry(
            strategy="Iron Fly",
            instrument_token="NSE_FO|NIFTY|2025JUN|23000CE", # Example instrument_token
            entry_price=22000.0,
            quantity=50.0,
            realized_pnl=0.0,
            unrealized_pnl=500.0,
            capital_used=60000.0,
            potential_loss=1000.0,
            sl_hit=False,
            vega=150.0,
            status="open"
        ),
        TradeLogEntry(
            strategy="Jade Lizard",
            instrument_token="NSE_FO|NIFTY|2025JUN|21500PE",
            entry_price=21800.0,
            quantity=50.0,
            realized_pnl=0.0,
            unrealized_pnl=-100.0,
            capital_used=45000.0,
            potential_loss=800.0,
            sl_hit=False,
            vega=100.0,
            status="open"
        )
    ]
    return pd.DataFrame([t.dict() for t in dummy_trades])

# --- WebSocket Manager for Broadcasting ---
class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Broadcasts a text message to all active WebSocket connections."""
        if not self.active_connections:
            # logger.debug("No active WebSocket connections to broadcast to.")
            return

        # Create a list to iterate over to allow modification during iteration if needed
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket {connection.client}. Disconnecting: {e}")
                self.disconnect(connection)

ws_manager = WebSocketManager()

# --- FastAPI Endpoints ---

@app.post("/auth/login", summary="Validate Upstox Access Token")
async def login(request: LoginRequest):
    config = await get_config(request.access_token)
    test_url = f"{config['base_url']}/user/profile"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(test_url, headers=config['headers'])
            if res.status_code == 200:
                return {"status": "success", "message": "Logged in successfully"}
            else:
                raise HTTPException(status_code=res.status_code, detail=f"Invalid token: {res.text}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {str(e)}")

@app.get("/dashboard", response_model=DashboardResponse, summary="Fetch Market Insights Dashboard Data")
async def get_dashboard(config: dict = Depends(get_access_token_dependency)):
    try:
        xgb_model = await load_xgboost_model()
        option_chain = await fetch_option_chain_data(config)
        if not option_chain:
            raise HTTPException(status_code=500, detail="Failed to fetch option chain data")

        spot_price = option_chain[0]["underlying_spot_price"]

        vix, nifty = await fetch_india_vix_and_nifty_spot(config)
        if not vix or not nifty:
            raise HTTPException(status_code=500, detail="Failed to fetch India VIX or Nifty 50 data")

        seller = extract_seller_metrics(option_chain, spot_price)
        if not seller:
            raise HTTPException(status_code=500, detail="Failed to extract seller metrics")

        full_chain_df_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            if abs(strike - spot_price) <= 500: # Consider relevant strikes for full chain
                call = opt.get("call_options")
                put = opt.get("put_options")
                if call and put and call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None:
                    full_chain_df_data.append({
                        "Strike": strike,
                        "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
                    })
        full_chain_df = pd.DataFrame(full_chain_df_data)

        market = calculate_market_metrics(option_chain, config['expiry_date'])
        ivp = await calculate_ivp(config, seller["avg_iv"])
        hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller["avg_iv"])
        
        xgb_vol = predict_xgboost_volatility(xgb_model, seller["avg_iv"], hv_7, ivp, market["pcr"], vix, market["days_to_expiry"], garch_7d)
        iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
        
        regime_score, regime, regime_note, regime_explanation = calculate_regime(
            seller["avg_iv"], ivp, hv_7, garch_7d, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope)
        
        # Load upcoming events for the dashboard
        event_df_raw = None
        try:
            async with httpx.AsyncClient() as client:
                events_response = await client.get(config['event_url'])
                events_response.raise_for_status()
                event_df_raw = pd.read_csv(BytesIO(events_response.content))
                event_df_raw.columns = event_df_raw.columns.str.strip()
                if 'Date' in event_df_raw.columns and 'Time' in event_df_raw.columns:
                    current_year = datetime.now().year
                    event_df_raw['CombinedDateTimeStr'] = event_df_raw.apply(
                        lambda row: f"{row['Date']}-{current_year} {row['Time']}" if pd.notna(row['Date']) and pd.notna(row['Time']) else np.nan,
                        axis=1
                    )
                    event_df_raw['Datetime'] = pd.to_datetime(event_df_raw['CombinedDateTimeStr'], format='%d-%b-%Y %H:%M', errors='coerce')
                    event_df_raw = event_df_raw.drop(columns=['CombinedDateTimeStr'])
                else:
                    logger.error("Missing 'Date' or 'Time' columns in upcoming_events.csv. Cannot process event dates for dashboard.")
                    event_df_raw['Datetime'] = pd.NaT
        except Exception as e:
            logger.error(f"Error loading upcoming events for dashboard: {e}")
            event_df_raw = pd.DataFrame() # Ensure it's an empty DataFrame on error

        # Convert relevant columns to string for JSON serialization if they contain non-serializable types
        if not event_df_raw.empty:
            # Only include columns that are directly useful for the dashboard or can be easily serialized
            event_cols = ['Event', 'Date', 'Time', 'Classification', 'Forecast', 'Prior']
            # Filter columns that exist in the DataFrame
            event_df_filtered = event_df_raw[[col for col in event_cols if col in event_df_raw.columns]]
            events_list = event_df_filtered.to_dict(orient="records")
        else:
            events_list = []


        strategies, strategy_rationale, event_warning = await suggest_strategy(
            regime, ivp, iv_rv_spread, market["days_to_expiry"], config['expiry_date'], seller["straddle_price"], spot_price, config)

        return DashboardResponse(
            nifty=nifty,
            vix=vix,
            atm_strike=seller['atm_strike'],
            straddle_price=seller['straddle_price'],
            atm_iv=seller['avg_iv'],
            ivp=ivp,
            days_to_expiry=market['days_to_expiry'],
            pcr=market['pcr'],
            iv_rv_spread=iv_rv_spread,
            iv_skew_slope=iv_skew_slope,
            regime=regime,
            regime_score=regime_score,
            regime_note=regime_note,
            regime_explanation=regime_explanation,
            events=events_list,
            strategies=strategies,
            strategy_rationale=strategy_rationale,
            event_warning=event_warning
        )
    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {str(e)}")

@app.get("/option-chain", response_model=OptionChainResponse, summary="Fetch Option Chain Analysis")
async def get_option_chain(config: dict = Depends(get_access_token_dependency)):
    try:
        option_chain = await fetch_option_chain_data(config)
        if not option_chain:
            raise HTTPException(status_code=500, detail="Failed to fetch option chain data")
        
        spot_price = option_chain[0]["underlying_spot_price"]
        
        full_chain_df_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            if abs(strike - spot_price) <= 500: # Limit to strikes around +/- 500 points from spot for relevance
                call = opt.get("call_options")
                put = opt.get("put_options")
                # Ensure required data exists before adding to list
                if call and put and \
                   call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None and \
                   call["market_data"].get("ltp") is not None and put["market_data"].get("ltp") is not None and \
                   call["market_data"].get("oi") is not None and put["market_data"].get("oi") is not None:
                    full_chain_df_data.append({
                        "Strike": strike,
                        "Call IV": round(call["option_greeks"]["iv"], 2),
                        "Put IV": round(put["option_greeks"]["iv"], 2),
                        "IV Skew": round(call["option_greeks"]["iv"] - put["option_greeks"]["iv"], 4),
                        "Total Theta": round((call["option_greeks"].get("theta", 0.0) or 0.0) + (put["option_greeks"].get("theta", 0.0) or 0.0), 2),
                        "Total Vega": round((call["option_greeks"].get("vega", 0.0) or 0.0) + (put["option_greeks"].get("vega", 0.0) or 0.0), 2),
                        "Straddle Price": round(call["market_data"]["ltp"] + put["market_data"]["ltp"], 2),
                        "Total OI": int((call["market_data"].get("oi", 0) or 0) + (put["market_data"].get("oi", 0) or 0))
                    })
        full_chain_df = pd.DataFrame(full_chain_df_data)
        full_chain_df = full_chain_df.sort_values(by="Strike").reset_index(drop=True)

        eff_df = full_chain_df.copy()
        # Handle division by zero for Theta/Vega
        eff_df["Theta/Vega"] = eff_df.apply(
            lambda row: row["Total Theta"] / row["Total Vega"] if row["Total Vega"] != 0 else (float('inf') if row["Total Theta"] > 0 else 0.0), axis=1
        )
        eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values("Theta/Vega", ascending=False)
        
        # Convert to Pydantic models
        option_chain_response_data = [OptionChainDataRow.parse_obj(row) for row in full_chain_df.to_dict(orient="records")]

        return OptionChainResponse(
            option_chain=option_chain_response_data,
            theta_vega_ranking=eff_df.to_dict(orient="records")
        )
    except Exception as e:
        logger.error(f"Error in option chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching option chain: {str(e)}")

@app.get("/strategies", summary="Fetch Strategy Suggestions with Details")
async def get_strategies_with_details(config: dict = Depends(get_access_token_dependency)):
    try:
        option_chain = await fetch_option_chain_data(config)
        if not option_chain:
            raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")

        spot_price = option_chain[0]["underlying_spot_price"]
        seller = extract_seller_metrics(option_chain, spot_price)
        if not seller or seller.get("avg_iv") is None:
            raise HTTPException(status_code=500, detail="Could not extract seller metrics from option chain for strategy suggestion.")

        market = calculate_market_metrics(option_chain, config['expiry_date'])
        ivp = await calculate_ivp(config, seller["avg_iv"])
        hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller["avg_iv"])

        full_chain_df_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            if abs(strike - spot_price) <= 300: # Limit range for skew calculation
                call = opt.get("call_options")
                put = opt.get("put_options")
                if call and put and call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None:
                    full_chain_df_data.append({
                        "Strike": strike,
                        "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
                    })
        full_chain_df = pd.DataFrame(full_chain_df_data)
        iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
        
        vix, _ = await fetch_india_vix_and_nifty_spot(config) # Get VIX for regime calculation

        regime_score, regime, regime_note, regime_explanation = calculate_regime(
            seller["avg_iv"], ivp, hv_7, garch_7d, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope)
        
        strategies_list, strategy_rationale, event_warning = await suggest_strategy(
            regime, ivp, iv_rv_spread, market["days_to_expiry"], config['expiry_date'], seller["straddle_price"], spot_price, config)

        strategy_details_responses: List[StrategyDetailResponse] = []
        for strat_name in strategies_list:
            detail = await get_strategy_details(strat_name, option_chain, spot_price, config, lots=1)
            if detail:
                try:
                    # Convert orders to StrategyLeg Pydantic models
                    detail_orders = [StrategyLeg(**order) for order in detail['orders']]
                    strategy_details_responses.append(
                        StrategyDetailResponse(
                            strategy=detail["strategy"],
                            strikes=detail["strikes"],
                            orders=detail_orders,
                            premium=detail["premium"],
                            premium_total=detail["premium_total"],
                            max_profit=detail["max_profit"],
                            max_loss=detail["max_loss"],
                            margin=detail["margin"]
                        )
                    )
                except ValidationError as e:
                    logger.error(f"Validation error for strategy {strat_name} details: {e}")
                    # Optionally, skip this strategy or raise a specific error
                except Exception as e:
                    logger.error(f"Unexpected error processing strategy {strat_name} details: {e}")

        response = {
            "regime": regime,
            "regime_score": regime_score,
            "regime_note": regime_note,
            "regime_explanation": regime_explanation,
            "strategies": strategies_list,
            "strategy_rationale": strategy_rationale,
            "event_warning": event_warning,
            "strategy_details": strategy_details_responses
        }
        return response
    except Exception as e:
        logger.error(f"Error in strategies endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching strategies: {str(e)}")


@app.post("/orders/place", summary="Place Multi-Leg Strategy Order")
async def place_strategy_order(request: StrategyOrderRequest, config: dict = Depends(get_access_token_dependency)):
    try:
        option_chain = await fetch_option_chain_data(config)
        if not option_chain:
            raise HTTPException(status_code=400, detail="Failed to fetch option chain data.")
        
        spot_price = option_chain[0]["underlying_spot_price"]
        detail = await get_strategy_details(request.strategy, option_chain, spot_price, config, lots=request.lots)
        
        if not detail:
            raise HTTPException(status_code=400, detail=f"Unable to generate order details for {request.strategy}")
        
        # Modify order_type and validity for all legs based on request
        for leg in detail["orders"]:
            leg["order_type"] = request.order_type
            leg["validity"] = request.validity

        success = await place_multi_leg_orders(config, detail["orders"])
        if success:
            trade_data = {
                "strategy": request.strategy,
                "instrument_token": config["instrument_key"], # Or derive from strategy legs
                "entry_price": detail["premium_total"] / request.lots if request.lots else 0, # Average entry price per lot
                "quantity": request.lots * config["lot_size"], # Total quantity
                "realized_pnl": 0.0, # Initial, to be updated on exit
                "unrealized_pnl": 0.0, # Initial, to be updated live
                # regime_score is not part of StrategyOrderRequest, so cannot directly use request.regime_score
                "notes": f"Strategy: {request.strategy}, Lots: {request.lots}",
                "capital_used": detail["margin"], # Use calculated margin as capital used
                "potential_loss": detail["max_loss"],
                "sl_hit": False,
                "vega": sum(o.get("vega", 0) for o in detail["orders"]) # Sum up vega from each leg if available
            }
            await log_trade(trade_data)
            
            # Place SL orders for SELL legs
            for leg in detail["orders"]:
                if leg["transaction_type"] == "SELL" and request.sl_percentage > 0:
                    sl_price = leg["current_price"] * (1 + request.sl_percentage / 100)
                    await create_gtt_order(config, leg["instrument_key"], sl_price, "BUY", tag=f"SL_{request.strategy}_{leg['instrument_key'].split('|')[-1]}")
            
            return {
                "status": "success",
                "message": f"Placed {request.strategy} order with {request.lots} lots",
                "margin": detail["margin"],
                "premium_collected": detail["premium_total"],
                "max_loss": detail["max_loss"],
                "max_profit": detail["max_profit"]
            }
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to place {request.strategy} order")
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error placing order: {str(e)}")

@app.get("/portfolio", response_model=PortfolioResponse, summary="Fetch Portfolio and Risk Summary")
async def get_portfolio(config: dict = Depends(get_access_token_dependency)):
    try:
        # Fetch live option chain for latest prices/greeks
        option_chain = await fetch_option_chain_data(config)
        if not option_chain:
            raise HTTPException(status_code=400, detail="Failed to fetch option chain data for portfolio evaluation.")
        spot_price = option_chain[0]["underlying_spot_price"]

        # Prepare full_chain_df for IV skew calculation
        full_chain_df_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            if abs(strike - spot_price) <= 300: # Limit range for skew calculation
                call = opt.get("call_options")
                put = opt.get("put_options")
                if call and put and call["option_greeks"].get("iv") is not None and put["option_greeks"].get("iv") is not None:
                    full_chain_df_data.append({
                        "Strike": strike,
                        "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"]
                    })
        full_chain_df = pd.DataFrame(full_chain_df_data)

        trades_df = await fetch_trade_data(config, full_chain_df) # This should fetch LIVE positions
        
        vix, _ = await fetch_india_vix_and_nifty_spot(config)
        seller = extract_seller_metrics(option_chain, spot_price)
        ivp = await calculate_ivp(config, seller["avg_iv"])
        hv_7, garch_7d, _ = await calculate_volatility(config, seller["avg_iv"])
        iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
        market_metrics_data = calculate_market_metrics(option_chain, config['expiry_date'])

        regime_score, regime, _, _ = calculate_regime(
            seller["avg_iv"], ivp, hv_7, garch_7d, seller["straddle_price"], spot_price, market_metrics_data["pcr"], vix, iv_skew_slope)
        
        # Convert trades_df records to list of TradeLogEntry models for type hinting in evaluate_full_risk
        active_trades_list = [TradeLogEntry(**row) for row in trades_df.to_dict(orient="records")]

        strategy_summary_records, portfolio_summary_pydantic = await evaluate_full_risk(active_trades_list, config, regime, vix)
        
        funds_data = await get_funds_and_margin(config)
        sharpe_ratio = calculate_sharpe_ratio() # This should be based on actual historical PnL

        # Ensure strategy_summary_records are converted to PortfolioStrategySummary
        strategy_summary_parsed = [PortfolioStrategySummary.parse_obj(rec) for rec in strategy_summary_records]

        return PortfolioResponse(
            available_capital=funds_data["available_margin"],
            used_margin=funds_data["used_margin"],
            exposure_percent=portfolio_summary_pydantic.Exposure_Percent, # Access by Pydantic attribute name
            sharpe_ratio=sharpe_ratio,
            strategy_summary=strategy_summary_parsed,
            risk_alerts=portfolio_summary_pydantic.Flags
        )
    except Exception as e:
        logger.error(f"Error in portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")

@app.get("/trades", summary="Fetch Trade Logs", response_model=List[TradeLogEntry])
async def get_trades(config: dict = Depends(get_access_token_dependency)):
    try:
        trades_data = await get_all_trades() # This fetches from Supabase now
        # Validate each trade entry against TradeLogEntry model
        validated_trades = []
        for trade in trades_data:
            try:
                validated_trades.append(TradeLogEntry(**trade))
            except ValidationError as e:
                logger.warning(f"Skipping invalid trade entry from Supabase: {trade}. Error: {e}")
        return validated_trades
    except Exception as e:
        logger.error(f"Error in fetching trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching trades: {str(e)}")

@app.post("/journal", summary="Add Journal Entry")
async def add_journal(request: JournalEntryRequest, config: dict = Depends(get_access_token_dependency)):
    try:
        journal_data = request.dict()
        if await add_journal_entry(journal_data):
            return {"status": "success", "message": "Journal entry saved"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save journal entry")
    except Exception as e:
        logger.error(f"Error adding journal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding journal: {str(e)}")

@app.get("/journal", summary="Fetch Journal Entries", response_model=List[JournalLogEntry])
async def get_journal(config: dict = Depends(get_access_token_dependency)):
    try:
        journals_data = await get_all_journals() # This fetches from Supabase now
        # Validate each journal entry against JournalLogEntry model
        validated_journals = []
        for journal in journals_data:
            try:
                validated_journals.append(JournalLogEntry(**journal))
            except ValidationError as e:
                logger.warning(f"Skipping invalid journal entry from Supabase: {journal}. Error: {e}")
        return validated_journals
    except Exception as e:
        logger.error(f"Error in fetching journal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching journal: {str(e)}")

# --- WebSocket Proxy Endpoint ---
@app.websocket("/ws/market")
async def market_data_websocket_proxy(websocket: WebSocket, access_token: str = Query(..., description="Upstox API Access Token")):
    """
    Establishes a WebSocket connection to Upstox Market Data Feed and proxies messages to connected clients.
    """
    await ws_manager.connect(websocket)
    upstox_ws_client = None  # To hold the Upstox client websocket object

    try:
        # Get authorized WebSocket URI from Upstox
        async with httpx.AsyncClient() as client:
            auth_url = "https://api.upstox.com/v2/feed/market-data-feed/authorize"
            headers = get_upstox_headers(access_token)
            auth_response = await client.get(auth_url, headers=headers)
            auth_response.raise_for_status()
            authorized_uri = auth_response.json()["data"]["authorized_redirect_uri"]

        # Connect to Upstox feed and rebroadcast
        logger.info(f"Attempting to connect to Upstox WebSocket: {authorized_uri}")
        async with websockets.connect(authorized_uri) as upstox_ws:
            upstox_ws_client = upstox_ws
            logger.info(f"Successfully connected to Upstox WebSocket: {authorized_uri}")
            while True:
                # Listen for messages from Upstox
                message_from_upstox = await upstox_ws.recv()
                # Broadcast the message to all connected FastAPI clients
                await ws_manager.broadcast(message_from_upstox)

    except WebSocketDisconnect:
        logger.info(f"Client WebSocket {websocket.client} disconnected.")
        ws_manager.disconnect(websocket)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during Upstox WS authorization: {e.response.status_code} - {e.response.text}")
        await websocket.close(code=1011, reason=f"Upstox Auth Error: {e.response.text}")
    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Upstox WebSocket connection closed gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Upstox WebSocket connection closed with error: {e.code} - {e.reason}")
        await websocket.close(code=1011, reason=f"Upstox WS Closed Error: {e.reason}")
    except Exception as e:
        logger.critical(f"Unexpected error in market_data_websocket_proxy: {e}", exc_info=True)
        await websocket.close(code=1011, reason=f"Internal Server Error: {str(e)}")
    finally:
        # Ensure the client websocket is disconnected from the manager
        ws_manager.disconnect(websocket)
        # No need to explicitly close upstox_ws here because `async with` handles it.

# You can keep other utility endpoints as needed, or remove them if their functionality
# is entirely covered by the new, more comprehensive endpoints.
# For instance, /predict/volatility, /suggest/strategy (already enhanced), /calculate/regime, etc.
# have their core logic now feeding into the dashboard or strategy suggestion endpoints.

# Example of an endpoint that might still be useful for granular testing/debugging
@app.get("/predict/volatility", summary="Predict Various Volatility Metrics")
async def predict_volatility_endpoint(access_token: str = Query(..., description="Upstox API Access Token")):
    logger.info("Predicting volatility...")
    config = await get_config(access_token)
    option_chain = await fetch_option_chain_data(config)
    if not option_chain:
        raise HTTPException(status_code=400, detail="Failed to fetch option chain or it's empty.")
    
    spot_price = option_chain[0]["underlying_spot_price"]
    seller_metrics = extract_seller_metrics(option_chain, spot_price)
    if not seller_metrics or seller_metrics.get("avg_iv") is None:
        raise HTTPException(status_code=500, detail="Could not extract seller metrics (e.g., average IV) from option chain for volatility calculation.")
    
    market_metrics_data = calculate_market_metrics(option_chain, config['expiry_date'])
    hv_7, garch_7d, iv_rv_spread = await calculate_volatility(config, seller_metrics["avg_iv"])
    xgb_model = await load_xgboost_model()
    
    vix, _ = await fetch_india_vix_and_nifty_spot(config) # Only need VIX here
    ivp = await calculate_ivp(config, seller_metrics["avg_iv"])
    
    xgb_vol = predict_xgboost_volatility(
        xgb_model, seller_metrics["avg_iv"], hv_7, ivp, market_metrics_data["pcr"],
        vix, market_metrics_data["days_to_expiry"], garch_7d
    )
    logger.info("Volatility prediction complete.")
    return {
        "volatility": {
            "hv_7": round(hv_7, 2),
            "garch_7d": round(garch_7d, 2),
            "xgb_vol": round(xgb_vol, 2),
            "ivp": round(ivp, 2),
            "vix": round(vix, 2),
            "iv_rv_spread": round(iv_rv_spread, 2)
        }
    }
