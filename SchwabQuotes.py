
import schwab
from schwab.auth import easy_client
from schwab.client import Client
import pandas as pd
import json
from datetime import datetime, date

# ── CONFIG ──────────────────────────────────────────────────────────────────
API_KEY      = "YOUR_APP_KEY"        # From developer.schwab.com
APP_SECRET   = "YOUR_APP_SECRET"
REDIRECT_URI = "https://127.0.0.1"
TOKEN_PATH   = "schwab_token.json"   # Stored locally after first login

# ── AUTHENTICATE ─────────────────────────────────────────────────────────────
# First run: opens a browser for OAuth login and saves the token.
# Subsequent runs: reuses the saved token automatically.
client = easy_client(
    api_key=API_KEY,
    app_secret=APP_SECRET,
    callback_url=REDIRECT_URI,
    token_path=TOKEN_PATH,
)

# ── FETCH OPTIONS CHAIN ───────────────────────────────────────────────────────
def get_options_chain(
    symbol: str,
    contract_type: str = "ALL",          # "CALL", "PUT", or "ALL"
    strike_count: int = 10,              # Number of strikes above/below ATM
    from_date: date = None,
    to_date: date = None,
    include_underlying_quote: bool = True,
) -> dict:
    """
    Fetch the full options chain for a given symbol.
    Returns the raw JSON response as a dict.
    """
    resp = client.get_option_chain(
        symbol,
        contract_type=Client.Options.ContractType(contract_type),
        strike_count=strike_count,
        include_underlying_quote=include_underlying_quote,
        from_date=from_date,
        to_date=to_date,
    )
    assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
    return resp.json()

# ── PARSE INTO DATAFRAME ──────────────────────────────────────────────────────
def parse_options_to_df(chain: dict) -> pd.DataFrame:
    """
    Flattens the nested options chain JSON into a clean Pandas DataFrame.
    Includes strike, expiry, bid, ask, last, volume, OI, IV, and Greeks.
    """
    rows = []

    for option_type in ["callExpDateMap", "putExpDateMap"]:
        exp_map = chain.get(option_type, {})
        side = "CALL" if option_type == "callExpDateMap" else "PUT"

        for expiry_key, strikes in exp_map.items():
            expiry_date = expiry_key.split(":")[0]   # e.g. "2025-04-18:30"

            for strike_price, contracts in strikes.items():
                for contract in contracts:
                    rows.append({
                        "symbol":           contract.get("symbol"),
                        "side":             side,
                        "expiry":           expiry_date,
                        "dte":              contract.get("daysToExpiration"),
                        "strike":           float(strike_price),
                        "bid":              contract.get("bid"),
                        "ask":              contract.get("ask"),
                        "last":             contract.get("last"),
                        "mark":             contract.get("mark"),
                        "volume":           contract.get("totalVolume"),
                        "open_interest":    contract.get("openInterest"),
                        "implied_vol":      contract.get("volatility"),
                        "delta":            contract.get("delta"),
                        "gamma":            contract.get("gamma"),
                        "theta":            contract.get("theta"),
                        "vega":             contract.get("vega"),
                        "rho":              contract.get("rho"),
                        "intrinsic_value":  contract.get("intrinsicValue"),
                        "in_the_money":     contract.get("inTheMoney"),
                    })

    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df.sort_values(["side", "expiry", "strike"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SYMBOL = "SPY"   # Change to any optionable ticker

    print(f"Fetching options chain for {SYMBOL}...")
    chain = get_options_chain(
        symbol=SYMBOL,
        contract_type="ALL",
        strike_count=10,
        from_date=date.today(),
    )

    # Underlying info
    underlying = chain.get("underlying", {})
    print(f"\nUnderlying: {SYMBOL}")
    print(f"  Last Price : ${underlying.get('last', 'N/A')}")
    print(f"  Mark       : ${underlying.get('mark', 'N/A')}")

    # Build DataFrame
    df = parse_options_to_df(chain)
    print(f"\nTotal contracts fetched: {len(df)}")
    print(df.head(10).to_string(index=False))

    # Optional: save to CSV
    out_file = f"{SYMBOL}_options_{date.today()}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved to {out_file}")
	