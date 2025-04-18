# -*- coding: utf-8 -*-
"""data processing.ipynb

Handles fetching and processing market data with quality checks.
"""

import yfinance as yf
import pandas as pd
import numpy as np # Add numpy import
import logging # Import logging

# Configure logging for this module.
# Note: It's often better to configure logging centrally in your main app (app.py),
# but this basic config allows the module to log independently if needed.
# If configured in app.py, these lines might not be necessary here.
log = logging.getLogger(__name__) # Get logger for this module
if not log.handlers: # Avoid adding handlers multiple times if already configured
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler() # Log to console by default for this module
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

def extraer_datos(tickers, fecha_inicio):
    """
    Descarga los precios de cierre ajustados desde Yahoo Finance.
    Handles different structures returned by yf.download.
    """
    log.info(f"Attempting to fetch data for tickers: {tickers} from {fecha_inicio}")
    try:
        # Use auto_adjust=True (default) for adjusted prices
        # group_by='ticker' helps structure multi-ticker downloads
        data = yf.download(tickers, start=fecha_inicio, auto_adjust=True, group_by='ticker')

        # --- Structure Handling ---
        if isinstance(tickers, list) and len(tickers) > 1:
            if isinstance(data.columns, pd.MultiIndex):
                 # Try to select only the 'Close' column for each ticker
                 try:
                     data_close = data.loc[:, pd.IndexSlice[:, 'Close']]
                     # Rename columns to just the ticker name
                     data_close.columns = data_close.columns.levels[0] if data_close.columns.nlevels > 1 else data_close.columns
                     data = data_close # Assign back the processed data
                     log.info("Processed multi-index columns for multiple tickers.")
                 except KeyError:
                     log.warning(f"Could not find 'Close' column in multi-index structure for {tickers}. Processing may be affected.")
                     # Decide on fallback: return raw data, raise error, or try other columns? Here we continue with potentially unprocessed data.
            # Handle case where only some tickers returned data, resulting in single-level columns
            elif not data.empty:
                 fetched_tickers = [t for t in tickers if t in data.columns]
                 missing_tickers = [t for t in tickers if t not in data.columns]
                 if missing_tickers:
                     log.warning(f"Could not fetch data for tickers: {missing_tickers}")
                 if not fetched_tickers:
                      log.error(f"None of the requested tickers {tickers} returned data.")
                      return pd.DataFrame() # Return empty DataFrame
                 data = data[fetched_tickers] # Keep only fetched tickers
                 log.info(f"Successfully fetched single-level data for: {fetched_tickers}")

        elif isinstance(tickers, list) and len(tickers) == 1:
             # Single ticker in a list
             if 'Close' in data.columns:
                  data = data[['Close']] # Keep it as a DataFrame
                  data.columns = tickers # Rename column to ticker name
                  log.info(f"Processed single ticker in list: {tickers[0]}")
             else:
                 log.warning(f"Could not find 'Close' column for single ticker in list: {tickers[0]}")

        elif isinstance(tickers, str):
             # Single ticker as string
             if 'Close' in data.columns:
                 data = data[['Close']] # Select the 'Close' column, keep as DataFrame
                 data.columns = [tickers] # Rename the column
                 log.info(f"Processed single ticker string: {tickers}")
             else:
                 log.warning(f"Could not find 'Close' column for single ticker string: {tickers}")

        # --- Final Checks ---
        if data.empty:
            log.error(f"No data successfully extracted/processed for tickers: {tickers}")
            # It might be better to raise an exception here if no data is acceptable
            # raise ValueError(f"Failed to fetch or process any data for tickers: {tickers}")

        log.info(f"Data extraction finished. DataFrame shape: {data.shape}")
        return data

    except Exception as e:
        log.error(f"An error occurred during data extraction for {tickers}: {e}", exc_info=True)
        # Return empty DataFrame or re-raise exception depending on desired handling
        return pd.DataFrame()


def perform_data_quality_checks(df):
    """Performs basic quality checks on the price data DataFrame."""
    quality_issues = []
    if df.empty:
        log.warning("Data quality check skipped: DataFrame is empty.")
        quality_issues.append("Input DataFrame is empty.")
        return df, quality_issues

    log.info(f"Performing data quality checks on DataFrame with shape: {df.shape}")

    # 1. Check for excessive initial missing values (before dropping)
    initial_nas = df.isnull().sum()
    total_points = len(df)
    if total_points > 0:
        for ticker, na_count in initial_nas.items():
            na_percentage = (na_count / total_points) * 100
            if na_percentage > 50: # Threshold for excessive missing data (e.g., 50%)
                msg = f"Data Quality Warning: Ticker '{ticker}' has {na_percentage:.1f}% missing values before cleaning."
                quality_issues.append(msg)
                log.warning(msg)

    # 2. Check for large single-day percentage changes (potential outliers)
    returns = df.pct_change()
    threshold = 0.30 # e.g., +/- 30% change in one day
    large_changes = returns[(returns.abs() > threshold)] # Check absolute change > threshold
    if not large_changes.isnull().all().all():
        for col in large_changes.columns:
            changes_for_ticker = large_changes[col].dropna()
            if not changes_for_ticker.empty:
                 change_dates = changes_for_ticker.index.strftime('%Y-%m-%d').tolist()
                 msg = f"Data Quality Warning: Ticker '{col}' has large daily changes (> {threshold:.0%}) on dates: {change_dates[:5]}" # Show first 5
                 if len(change_dates) > 5: msg += "..."
                 quality_issues.append(msg)
                 log.warning(msg)

    # 3. Check for zero or negative prices
    if (df <= 0).any().any():
         neg_zero_cols = df.columns[(df <= 0).any()].tolist()
         msg = f"Data Quality Warning: Tickers {neg_zero_cols} have zero or negative prices."
         quality_issues.append(msg)
         log.warning(msg)
         # Consider corrective action? e.g., clipping values:
         # df = df.clip(lower=0.001) # Replace 0 or negative with a small positive number
         # log.warning(f"Applied lower clip to tickers {neg_zero_cols} due to non-positive prices.")


    if not quality_issues:
        log.info("Basic data quality checks passed without significant warnings.")
    else:
        log.warning(f"Data quality checks completed with {len(quality_issues)} potential issue(s).")

    # Return the (potentially modified, e.g., clipped) df and the list of found issues
    return df, quality_issues


def procesar_datos(data):
    """Performs quality checks and cleans data (e.g., drops NaNs)."""
    log.info("Starting data processing and quality checks...")

    if data is None or data.empty:
        log.warning("Skipping processing: Input data is empty or None.")
        return pd.DataFrame()

    # Perform quality checks
    data_checked, quality_issues = perform_data_quality_checks(data)
    # Note: quality_issues are logged within the function. You could pass them back
    # if the main app needs to react to them (e.g., display warnings to user).

    # Proceed with cleaning - dropping rows with *any* NaN value
    rows_before = len(data_checked)
    data_cleaned = data_checked.dropna(how='any') # Use how='any' explicitly
    rows_after = len(data_cleaned)

    if rows_before > 0:
        log.info(f"Removed {rows_before - rows_after} out of {rows_before} rows containing NaN values.")
    else:
        log.info("Input data had 0 rows initially.")


    if data_cleaned.empty and rows_before > 0:
         log.error("Data quality critical: DataFrame became empty after dropping NaN values. Check initial data quality or date range.")
         # Raise an exception to stop the process in app.py
         raise ValueError("No valid, complete data rows remaining after cleaning. Cannot proceed with optimization.")

    log.info(f"Data processing finished. Final DataFrame shape: {data_cleaned.shape}")
    return data_cleaned