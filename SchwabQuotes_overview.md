# Overview of SchwabQuotes.py

The `SchwabQuotes.py` script is a Python application designed to programmatically access and process options chain data from the Charles Schwab API.

## Key Functionality:

1.  **API Configuration**: It initializes necessary API credentials such as `API_KEY`, `APP_SECRET`, `REDIRECT_URI`, and specifies a `TOKEN_PATH` for storing authentication tokens.
2.  **Authentication Flow**: The script handles OAuth authentication with the Schwab API using `schwab.auth.easy_client`. For the initial run, it facilitates user login via a browser to obtain an authorization token, which is then saved locally to `schwab_token.json` for seamless re-authentication in subsequent executions.
3.  **Options Chain Retrieval (`get_options_chain`)**: This function is responsible for querying the Schwab API for options chain data. It accepts parameters like the stock `symbol`, `contract_type` (CALL, PUT, or ALL), `strike_count` (number of strikes around the ATM price), and date ranges (`from_date`, `to_date`). It returns the raw JSON response from the API.
4.  **Data Parsing and Structuring (`parse_options_to_df`)**: The raw, nested JSON data received from the API is transformed into a clean and tabular Pandas DataFrame. This function extracts critical option contract details including:
    *   Symbol
    *   Option Type (CALL/PUT)
    *   Expiration Date (`expiry`)
    *   Days to Expiration (`dte`)
    *   Strike Price
    *   Bid, Ask, Last, and Mark prices
    *   Trading Volume and Open Interest
    *   Implied Volatility (`implied_vol`)
    *   Option Greeks (Delta, Gamma, Theta, Vega, Rho)
    *   Intrinsic Value
    *   In-the-money status
5.  **Execution and Output**: When run as a standalone script, it fetches the options chain for a predefined `SYMBOL` (defaulting to "SPY"). It then displays summary information about the underlying asset and the fetched options contracts. Finally, it saves the processed DataFrame to a CSV file named in the format `[SYMBOL]_options_[current_date].csv`, making the data readily available for further analysis or storage.

In summary, `SchwabQuotes.py` provides a robust solution for automating the collection, structuring, and storage of Schwab options market data.