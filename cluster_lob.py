# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from scipy.stats.mstats import winsorize
import seaborn as sns
import datetime
from joblib import Parallel, delayed
# import joblib
# import joblib.externals.loky
import plotly.graph_objects as go
from scipy.stats import norm,kurtosis,skew
#joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e10)
import arcticdb_api as aci
from collections import Counter
import time

# Functions
def tm_caculation():
    """
    Factory that returns a time-difference calculator function.

    This factory produces a callable that, on each invocation, computes the time
    elapsed since the last “significant” order (i.e. the last order whose `Diff` ≠ 0).
    On the very first call (or if `Diff == 0`), it returns 0.

    :returns:
        Callable[[order], float]: A function which accepts a `order` object and
        returns the elapsed time (float) since the last order with `Diff != 0`.
    """
    t = 0

    def _tm_caculation(order):
        """
        Calculate elapsed time since the last order with a non-zero `Diff`.

        :param order:
            An object with attributes:
            
            - **Time** (float): The current timestamp.
            - **Diff** (int): A flag indicating whether this order is “significant” (non-zero).

        :type order: Any object with `.Time` and `.Diff` attributes
        :returns:
            float: The difference between `order.Time` and the stored reference time.
                   Returns 0 if this is the first call or if `Diff == 0`.
        """
        nonlocal t
        # Raw difference
        t_diff = order.Time - t
        # If no prior timestamp, zero out the diff
        if not t:
            t_diff = 0
        # Only update reference time when Diff signals significance
        if order.Diff != 0:
            t = order.Time
        return t_diff
    return _tm_caculation

def t_elapse():
    """
    Factory that returns a price–time elapse calculator for orders by direction.

    This factory produces a callable which, on each invocation with a `order`
    object, computes two values:

      1. **dt**: Time elapsed since the last order at the same price and direction
         (i.e. depth change).
      2. **occur_t**: Time elapsed since the first occurrence of this price
         at the same direction.

    Internally, it tracks for each direction (indexed by int(order.Direction)):
      - `t[d]`: the most recent timestamp seen.
      - `depth_t[d]`: a dict mapping price → last timestamp for depth changes.
      - `price_occur[d]`: a dict mapping price → first timestamp observed.

    :returns:
        Callable[[order], Tuple[float, float]]: A function that accepts a `order`
        object and returns a `(dt, occur_t)` pair, both floats.
    """
    t = [0, 0, 0]
    depth_t = ({}, {}, {})
    price_occur = ({}, {}, {})
    def _t_elapse(order):
        """
        Calculate (dt, occur_t) for a given order.

        :param order:
            An object with attributes:
            
            - **Time** (float): The current timestamp.
            - **Direction** (int or convertible to int): -1: Sell limit order; 1: Buy limit order
            - **Price** (hashable): The price level of the order.

        :type order: Any object with `.Time`, `.Direction`, and `.Price`

        :returns:
            Tuple[float, float]:
            
            - **dt**:  
              Time since the last order at this same price and direction.
              If this is the first depth event at this price, returns 0.
            
            - **occur_t**:  
              Time since the first time this price was observed for this direction.
        """
        nonlocal t, depth_t, price_occur
        # Convert direction to index
        d = int(order.Direction)
        # Time since first occurrence of this price (or 0 if new)
        occur_t = order.Time - price_occur[d].get(order.Price, order.Time)
        # Time since last depth change for this price
        if order.Price in depth_t[d]:
            dt = order.Time - depth_t[d][order.Price]
        else:
            dt = 0
        # Record first occurrence if not seen before
        price_occur[d].setdefault(order.Price, order.Time)
        # Update last depth-change timestamp
        depth_t[d][order.Price] = order.Time
        # Update most recent time seen for this direction
        t[d] = order.Time
        return dt, occur_t
    return _t_elapse

def convert_seconds_to_time(nanoseconds_since_midnight):
    """
    Convert a count of nanoseconds since midnight into a `datetime.time` object.

    :param nanoseconds_since_midnight:
        Total number of nanoseconds elapsed since 00:00:00.
    :type nanoseconds_since_midnight: int or float

    :returns:
        A `datetime.time` instance corresponding to the given nanoseconds past midnight.
    :rtype: datetime.time
    """
    # Convert nanoseconds to seconds
    seconds_since_midnight = nanoseconds_since_midnight / 1e9
    # Build a datetime from the minimum date plus the elapsed seconds, then extract the time
    time_of_day = (
        datetime.datetime.min
        + datetime.timedelta(seconds=seconds_since_midnight)
    ).time()
    return time_of_day

def rapid_daily_data_processing(ticker, event_type, start_time, end_time):
    """
    Load and filter raw LOBSTER order book data for a given ticker and time range,
    selecting only specified event types.

    :param ticker:
        The security symbol to load data for.
    :type ticker: str
    :param event_type:
        Which message events to include; must be one of:
        
        - **"A"**: All events (EventType 1–5).
        - **"L"**: Add-only events (EventType == 1).
        - **"D"**: Delete-only events (EventType == 2 or 3).
        - **"M"**: Trade-only events (EventType == 4 or 5).
    :type event_type: str
    :param start_time:
        Start of the date/time range to load (inclusive).
    :type start_time: datetime.datetime
    :param end_time:
        End of the date/time range to load (inclusive).
    :type end_time: datetime.datetime
    :returns:
        pandas.DataFrame containing, for each order in the window:
        
        - **Name**: Ticker symbol.
        - **DateTime**, **Time**, **EventType**, **OrderID**, **Size**, **Price**, **Direction**:
          Raw message fields.
        - **AskPrice1–10**, **AskSize1–10**, **BidPrice1–10**, **BidSize1–10**:
          Top 10 levels of the order book.
    :rtype: pandas.DataFrame
    """
    # Specify the Arctic dataset name for LOBSTER data
    str_data_set = 'lobster'
    # Connect to the Arctic database instance
    adb = aci.fn_adb_instance(str_data_set)
    # Define the specific Arctic library to read from
    library = "lobster-mbp-10"
    arctic_library = adb[library]
    # Create a tuple representing the query time window
    date_range = (start_time, end_time)
    # Read the message+LOB data for the given ticker and date range
    features = arctic_library.read(symbol=ticker, date_range=date_range, ).data
    # Flatten the index so that DateTime becomes a column
    features = features.reset_index()
    # Columns for the raw message fields
    msg_col_name = ['DateTime', 'Time', 'EventType', 'OrderID', 'Size', 'Price', 'Direction']
    # Base names for the top-of-book and depth levels
    lob_col_base = ['AskPrice', 'AskSize', 'BidPrice', 'BidSize']
    lob_col_name = []
    # Generate column names for 10 levels of the order book
    for i in range(1, 11):
        lob_col_name = lob_col_name + [col + str(i) for col in lob_col_base]
    # Combine message columns with LOB depth columns
    col_name = msg_col_name + lob_col_name
    features.columns = col_name
    # Remove any entries with no buy/sell direction
    features = features[features["Direction"] != 0]
    # Insert a column at position 0 to tag each row with the ticker name
    features.insert(0, "Name", ticker)
    # Filter by the specified event_type:
    # "A" = all events, "L" = adds only, "D" = deletes only, "M" = trades only
    if event_type == "A":
        features = features[features["EventType"].isin([1,2,3,4,5])]
    elif event_type == "L":
        features = features[features["EventType"].isin([1])]
    elif event_type == "D":
        features = features[features["EventType"].isin([2,3])]
    elif event_type == "M":
        features = features[features["EventType"].isin([4,5])]
    # Return the filtered DataFrame
    return features

def feature_processing(features, moving_size, time_intervals, j):
    """
    Process raw LOBSTER features into engineered features and compute interval returns.

    This function takes a DataFrame of raw limit order book and message data and:
      - Computes the mid‐price and sign of each event.
      - Derives BestSize, SignSize, and depth metrics V, SBS, OBS.
      - Calculates temporal features TM, TPre, and T1 using provided helper functions.
      - Standardizes each feature over a rolling window of size `moving_size`.
      - Assigns each order to a bucket based on its RealTime and the provided `time_intervals`.
      - Constructs a summary DataFrame of standardized features (`sub_features`).
      - Builds a second DataFrame of interval returns (`returns`) with start/end mid-prices.

    :param pandas.DataFrame features:
        Raw DataFrame containing at least the following columns:
        AskPrice1–10, BidPrice1–10, AskSize1–10, BidSize1–10,
        Price, Direction, Size, EventType, RealTime, Time.
    :param int moving_size:
        Window size for rolling mean and standard deviation when standardizing features.
    :param list of tuple time_intervals:
        List of (start_datetime, end_datetime) pairs defining each interval.
    :param int j:
        Integer offset used in bucket numbering: each bucket index is
        `len(time_intervals) * j + i + 1` for interval i in this batch.
    :returns:
        tuple:
          - sub_features (pandas.DataFrame):
            Contains columns ["Name", "RealTime", "SignSize", "Bucket", "EventType"]
            plus all standardized feature columns StdV, StdSBS, StdOBS, StdTM, StdT1, StdTPre.
          - returns (pandas.DataFrame):
            Each row has keys `StartInterval`, `EndInterval`,
            `StartMidPrice`, `EndMidPrice` for each interval.
    :rtype: tuple of pandas.DataFrame
    :raises KeyError:
        If required columns are missing from `features`.
    """
    # Midprice
    MidPrice = pd.DataFrame({"MidPrice": features[['AskPrice1', 'BidPrice1']].mean(axis=1)})
    features.insert(1, "MidPrice", MidPrice)
    features['SignEvent'] = np.where(features['EventType'].isin([1, 4, 5]), 1,
                            np.where(features['EventType'].isin([2, 3]), -1, np.nan))               
    # Calculate diff for best‐price trades
    diff = (features['BidPrice1'] - features['Price']) * (features['Direction'] == 1) + \
           (features['AskPrice1'] - features['Price']) * (features['Direction'] == -1)
    features['BestSize'] = features['Size'] * (diff == 0)
    features['SignSize'] = features['Direction'] * features['BestSize'] * features['SignEvent']
    # Depth
    size_titles = [f'BidSize{i}' for i in range(1, 11)] + [f'AskSize{i}' for i in range(1, 11)]
    price_titles = [f'BidPrice{i}' for i in range(1, 11)] + [f'AskPrice{i}' for i in range(1, 11)]
    sizes = features.loc[:, size_titles]
    prices = features.loc[:, price_titles]
    depth_mat = (prices.sub(features.Price, axis=0) == 0).astype(int)
    depth_mat.columns = sizes.columns
    features['V'] = (sizes * depth_mat).sum(axis=1)
    # Bid and ask subsets
    bidfeatures = features[features["Direction"] == 1]
    asksizes = features[features["Direction"] == -1].loc[:, [f'AskSize{i}' for i in range(1, 11)]]
    bidsizes = bidfeatures.loc[:, [f'BidSize{i}' for i in range(1, 11)]]
    bidprices = bidfeatures.loc[:, [f'BidPrice{i}' for i in range(1, 11)]]
    askprices = features[features["Direction"] == -1].loc[:, [f'AskPrice{i}' for i in range(1, 11)]]
    biddepth_mat = (bidprices.sub(features.Price, axis=0) >= 0).astype(int)
    biddepth_mat.columns = bidsizes.columns
    obiddepth_mat = biddepth_mat.copy(); obiddepth_mat.columns = asksizes.columns
    askdepth_mat = (askprices.sub(features.Price, axis=0) <= 0).astype(int)
    askdepth_mat.columns = asksizes.columns
    oaskdepth_mat = askdepth_mat.copy(); oaskdepth_mat.columns = bidsizes.columns
    # Compute SBS and OBS
    SBS_mat = pd.concat([biddepth_mat, askdepth_mat], axis=1)
    features['SBS'] = (sizes * SBS_mat).sum(axis=1)
    OBS_mat = pd.concat([oaskdepth_mat, obiddepth_mat], axis=1)
    features['OBS'] = (sizes * OBS_mat).sum(axis=1)
    # Temporal features
    features["Diff"] = features.MidPrice.diff(-1).fillna(0)
    features["TM"] = features.loc[:, ["Time", "Diff"]].apply(tm_caculation(), axis=1)
    cols = features.loc[:, ["Time", "Price", "Direction", "V"]].apply(t_elapse(), axis=1)
    cols = pd.DataFrame(cols.tolist(), columns=["TPre", "T1"])
    features = pd.concat([features, cols], axis=1)
    # Standardization
    for col in ["V", "SBS", "OBS", "TM", "T1", "TPre"]:
        moving_avg_col = features[col].rolling(window=moving_size).mean()
        moving_std_col = features[col].rolling(window=moving_size).std()
        features[f'Std{col}'] = (features[col] - moving_avg_col) / moving_std_col
    features = features.dropna()
    # Bucket assignment
    features['Bucket'] = features['RealTime'].apply(
        lambda x: next(
            (len(time_intervals) * j + (i + 1)
             for i, interval in enumerate(time_intervals)
             if interval[0] <= x < interval[1]),
            None
        )
    )
    # Select sub‐features
    std_feature_col = ["StdV","StdSBS","StdOBS","StdTM","StdT1","StdTPre"]
    sub_features = features[
        ["Name", "RealTime", "SignSize", "Bucket", "EventType"] + std_feature_col
    ]
    # Compute interval returns
    returns = pd.DataFrame([
        {
            'StartInterval': interval[0],
            'EndInterval':   interval[1],
            'StartMidPrice': interval_data['MidPrice'].iloc[0] if not interval_data.empty else None,
            'EndMidPrice':   interval_data['MidPrice'].iloc[-1] if not interval_data.empty else None
        }
        for interval in time_intervals
        for interval_data in [features[(features['RealTime'].apply(lambda x: interval[0] <= x < interval[1]))]]
    ])
    return sub_features, returns

def calc_imbalance(winsorized_features):
    """
    Calculate order flow imbalance size and count by time bucket and label.

    :param pandas.DataFrame winsorized_features:
        DataFrame containing at least the following columns:
        - **SignSize**: signed order size (+ for buys, – for sells)
        - **Bucket**: time bucket identifier
        - **Label**: cluster label (e.g., $\phi_{1}$, \phi_{2}, \phi_{3})

    :returns:
        tuple:
          - **imbalance_size** (pandas.DataFrame):
            Pivot table indexed by `Bucket` with columns
            `$OFI^{S}(\phi_{1})$`, `$OFI^{S}(\phi_{2})$`, `$OFI^{S}(\phi_{3})$`,
            representing net signed size imbalance per label.
          - **imbalance_count** (pandas.DataFrame):
            Pivot table indexed by `Bucket` with columns
            `$OFI^{C}(\phi_{1})$`, `$OFI^{C}(\phi_{2})$`, `$OFI^{C}(\phi_{3})$`,
            representing net signed order count imbalance per label.
    :rtype: tuple of pandas.DataFrame
    """
    # Sum of positive order sizes by bucket & label
    total_positive_order_size = \
        winsorized_features[np.sign(winsorized_features['SignSize']) == 1] \
        .groupby(['Bucket', 'Label'])['SignSize'].sum()
    # Sum of negative order sizes by bucket & label
    total_negative_order_size = \
        winsorized_features[np.sign(winsorized_features['SignSize']) == -1] \
        .groupby(['Bucket', 'Label'])['SignSize'].sum()
    # Net signed size imbalance
    imbalance_size = total_positive_order_size + total_negative_order_size
    imbalance_size = (
        imbalance_size
        .reset_index(name='Imbalance')
        .pivot_table(index='Bucket', columns='Label', values='Imbalance')
        .add_prefix(r'$OFI^{S}$')
    )
    imbalance_size_cols = [r'$OFI^{S}(\phi_{1})$', r'$OFI^{S}(\phi_{2})$', r'$OFI^{S}(\phi_{3})$']
    imbalance_size.columns = imbalance_size_cols
    # Count of positive orders by bucket & label
    total_positive_order_count = winsorized_features[np.sign(winsorized_features['SignSize']) == 1] \
        .groupby(['Bucket', 'Label']) \
        .size()
    # Count of negative orders by bucket & label
    total_negative_order_count = winsorized_features[np.sign(winsorized_features['SignSize']) == -1] \
        .groupby(['Bucket', 'Label']) \
        .size()
    # Net signed count imbalance
    imbalance_count = total_positive_order_count - total_negative_order_count
    imbalance_count = (
        imbalance_count
        .reset_index(name='Imbalance')
        .pivot_table(index='Bucket', columns='Label', values='Imbalance')
        .add_prefix(r'$OFI^{C}$')
    )
    imbalance_count_cols = [r'$OFI^{C}(\phi_{1})$', r'$OFI^{C}(\phi_{2})$', r'$OFI^{C}(\phi_{3})$']
    imbalance_count.columns = imbalance_count_cols
    return imbalance_size, imbalance_count

          
def calc_benchmark_imbalance(winsorized_features):
    """
    Calculate benchmark order flow imbalance (size and count) aggregated by time bucket.

    :param winsorized_features:
        DataFrame containing at least:
          - **SignSize**: signed order size (+ for buys, – for sells)
          - **Bucket**: time bucket identifier
    :type winsorized_features: pandas.DataFrame

    :returns:
        tuple:
          - **imbalance_size** (pandas.DataFrame): Indexed by `Bucket` with column `$OFI^{S}(\phi_{*})$` for size imbalance.
          - **imbalance_count** (pandas.DataFrame): Indexed by `Bucket` with column `$OFI^{C}(\phi__{*})$` for count imbalance.
    :rtype: tuple of pandas.DataFrame
    """
    # Buy size sum by bucket
    total_positive_order_size = (
        winsorized_features[np.sign(winsorized_features['SignSize']) == 1]
        .groupby('Bucket')['SignSize']
        .sum()
    )
    # Sell size sum by bucket
    total_negative_order_size = (
        winsorized_features[np.sign(winsorized_features['SignSize']) == -1]
        .groupby('Bucket')['SignSize']
        .sum()
    )
    # Net size imbalance
    imbalance_size = total_positive_order_size + total_negative_order_size
    # DataFrame with size imbalance label
    imbalance_size = pd.DataFrame(imbalance_size).rename(
        columns={"SignSize": r"$OFI^{S}(\phi_{*})$"}
    )
    # Buy count by bucket
    total_positive_order_count = (
        winsorized_features[np.sign(winsorized_features['SignSize']) == 1]
        .groupby('Bucket')
        .size()
    )
    # Sell count by bucket
    total_negative_order_count = (
        winsorized_features[np.sign(winsorized_features['SignSize']) == -1]
        .groupby('Bucket')
        .size()
    )
    # Net count imbalance
    imbalance_count = total_positive_order_count - total_negative_order_count
    # DataFrame with count imbalance label
    imbalance_count = pd.DataFrame(
        imbalance_count,
        columns=[r'$OFI^{C}(\phi_{*})$']
    )
    # Return size and count imbalance
    return imbalance_size, imbalance_count
    
def calc_correlation(result, k):
    """
    Compute correlations between order flow imbalance variables and returns.

    :param result:
        pandas.DataFrame containing order flow imbalance variables in its first (k+1)*2 columns
        and returns columns 'CONR', 'FRNB', and 'FREB'.
    :param int k:
        Number of order flow imbalance; determines how many order flow imbalance columns (2*(k+1)).
    :returns:
        pandas.DataFrame where each row is an order flow imbalance variable and each column is the
        Pearson correlation with one of the returns ('CONR', 'FRNB', 'FREB').
    :rtype: pandas.DataFrame
    """
    # Select columns corresponding to imbalance variables
    cols = result.columns[: (k + 1) * 2]
    # Compute correlations of imbalance variables with each target
    correlation = pd.DataFrame({
        "CONR": result[cols].corrwith(result['CONR']),  # with CONR
        "FRNB": result[cols].corrwith(result['FRNB']),  # with FRNB
        "FREB": result[cols].corrwith(result['FREB'])   # with FREB
    })
    return correlation

def calc_correlation_heatmap(correlation, event_type, folder, ticker, data_type):
    """
    Generate and save a heatmap of correlations between order flow imbalance variables and returns.

    :param correlation:
        pandas.DataFrame of correlation values, indexed by imbalance variable,
        with columns for each target series.
    :param event_type:
        Identifier string for the event category (e.g., 'A', 'L', 'M').
    :param folder:
        Path to the directory where the heatmap PDF will be saved.
    :param ticker:
        Security symbol used in the output filename.
    :param data_type:
        Dataset type; must be either 'train' or 'test'.
    :returns:
        None. Saves a PDF heatmap to the specified folder.
    """
    # Replace negative zeros with positive zero for clarity
    correlation[correlation == -0] = 0
    # Set up figure size based on matrix dimensions
    plt.figure(figsize=(len(correlation.columns), 0.5 * len(correlation)))
    # Plot heatmap with annotations
    ax = sns.heatmap(
        correlation.round(2),
        cmap='YlOrRd',
        annot=True,
        fmt='g',
        cbar=True
    )
    # Keep y-axis labels horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # Save the figure as a high-resolution PDF
    plt.savefig(
        f'{folder}/{data_type}_{ticker}_corr_heatmap_{event_type}.pdf',
        dpi=1000,
        bbox_inches='tight'
    )

def calc_PnL(result, FR, k):
    """
    Calculate daily Profit and Loss based on order flow imbalance variables.

    :param result:
        pandas.DataFrame containing order flow imbalance in its first 2*(k+1) columns
        and actual future returns in column named by `FR`.
    :param FR:
        String name of the column in `result` holding the target return.
    :param int k:
        Number of order flow imbalance; determines how many order flow imbalance columns (2*(k+1)).
    :returns:
        pandas.DataFrame indexed by 'Dates' with columns for each feature's PnL.
        Each value is sign(prediction)*actual_return minus transaction cost.
    :rtype: pandas.DataFrame
    """
    # Set transaction cost (e.g., 0.0005) per trade
    tc = 0  # adjust if needed
    # Compute PnL for each feature column: sign of prediction * actual return minus cost
    PnL = pd.DataFrame({
        col: np.sign(result[col]) * result[FR] - tc
        for col in result.columns[:2 * (k + 1)]
    })
    # Preserve date index
    PnL['Dates'] = result['Dates']
    # Aggregate PnL by date
    PnL = PnL.groupby('Dates').sum()
    return PnL
    
def calc_scaled_PnL(PnL):
    """
    Scale PnL to target annualized volatility.

    :param PnL:
        pandas.Series or DataFrame of daily PnL values indexed by date.
    :type PnL: pandas.Series or pandas.DataFrame
    :returns:
        pandas.Series or DataFrame of scaled PnL such that annualized volatility
        equals the target volatility (default 15%).
    :rtype: same type as `PnL`
    """
    # Desired target annualized volatility
    target_vol = 0.15
    # Compute scaling factor: target_vol / (observed_vol * sqrt(252))
    vol_scaling = target_vol / (np.std(PnL) * np.sqrt(252))
    # Apply scaling
    scaled_PnL = PnL * vol_scaling
    return scaled_PnL

def calc_metric(scaled_PnL, time_intervals, FR, event_type, folder, data_type):
    """
    Compute performance metrics from scaled PnL and save to CSV.

    :param scaled_PnL:
        pandas.Series or DataFrame of scaled daily PnL values indexed by date.
    :param list time_intervals:
        List of (start, end) time intervals for bucketing within the day.
    :param FR:
        String name of the target return column used in filename.
    :param event_type:
        Event category identifier (e.g., 'add', 'cancel', 'trade').
    :param folder:
        Path to directory where CSV will be saved.
    :param data_type:
        Dataset type; must be either 'train' or 'test'.
    :returns:
        pandas.DataFrame of performance metrics for each strategy (columns of scaled_PnL).
    :rtype: pandas.DataFrame
    """
    # Ensure datetime index
    scaled_PnL.index = pd.to_datetime(scaled_PnL.index)
    # Initialize DataFrame for metrics
    metrics = pd.DataFrame(index=[
        'E[Return]', 'Volatility', 'Downside deviation', 'Maximum drawdown',
        'Sortino ratio', 'Calmar ratio', 'Hit rate', 'Avg. profit / avg. loss',
        'PnL per trade', 'Sharpe ratio', 'P-value'
    ])
    # Number of periods for PnL per trade (time intervals assumed global)
    periods = len(time_intervals) - 1
    # Compute metrics column by column
    for name, pnl in scaled_PnL.iteritems():
        T = len(pnl)
        # Annualized mean return
        mean_ann = float(np.mean(pnl) * 252)
        # Annualized volatility
        vol_ann = float(np.std(pnl) * np.sqrt(252))
        # Sharpe ratio
        sharpe = round(mean_ann / vol_ann, 3)
        # PnL per trade
        ppt = round((np.mean(pnl) * 10000) / periods, 3)
        # P-value for Sharpe
        sr_nonann = np.mean(pnl) / np.std(pnl)
        z = sr_nonann / np.sqrt((1 - skew(pnl) * sr_nonann + (kurtosis(pnl) - 1) * sr_nonann**2 / 4) / (T - 1))
        pval = round(min(norm.cdf(z), 1 - norm.cdf(z)) * 2, 3)
        # Downside deviation
        neg = pnl[pnl < 0]
        dsd = float(neg.std() * np.sqrt(252))
        # Sortino ratio
        sortino = round(mean_ann / dsd, 3)
        # Max drawdown
        comp = (pnl + 1).cumprod()
        dd = round((comp / comp.cummax() - 1).min(), 3)
        # Calmar ratio
        calmar = round(mean_ann / abs(dd), 3)
        # Hit rate
        hit = round((pnl > 0).mean(), 3)
        # Avg profit / avg loss
        apl = round(abs(pnl[pnl > 0].mean() / pnl[pnl < 0].mean()), 3)
        # Assign metrics
        metrics[name] = [
            round(mean_ann, 3), vol_ann, dsd, dd, sortino, calmar,
            hit, apl, ppt, sharpe, pval
        ]
    # Save results
    metrics.to_csv(f"{folder}/{data_type}_{FR}_{event_type}.csv")
    return metrics

def calc_top_cumPnL_plot(scaled_PnL, time_intervals, FR, event_type, folder, data_type):
    """
    Plot and save cumulative scaled PnL curves for top strategies.

    :param scaled_PnL:
        pandas.DataFrame of scaled daily PnL indexed by date for each strategy.
    :param list time_intervals:
        List of (start, end) time intervals for bucketing within the day.
    :param FR:
        String name of the target return used in axis labels and filename.
    :param event_type:
        Identifier for the event category (e.g., 'add', 'cancel', 'trade').
    :param folder:
        Path to directory where the PDF will be saved.
    :param data_type:
        Dataset type; must be either 'train' or 'test'.
    :returns:
        None. Saves a PDF figure of cumulative PnL curves to the specified folder.
    :rtype: None
    """
    # Ensure datetime index for plotting
    scaled_PnL.index = pd.to_datetime(scaled_PnL.index)
    # Annualized Sharpe ratio for annotation
    SR = pd.DataFrame(
        scaled_PnL.apply(lambda col: round(np.mean(col)/np.std(col)*np.sqrt(252), 2))
    ).T
    # PnL per trade for annotation
    PPT = pd.DataFrame(
        scaled_PnL.apply(lambda col: round(np.mean(col)*10000/(len(time_intervals)-1), 2))
    ).T

    # Cumulative returns in percentage
    cumPnL = 100 * scaled_PnL.cumsum()

    # Plot styling: label, group, color, line style
    trace_dict = {
        r'$OFI^{S}(\phi_{1})$': ['Cluster 1', 'Directional cluster', 'blue', 'solid'],
        r'$OFI^{S}(\phi_{2})$': ['Cluster 2', 'Opportunistic cluster', 'red', 'solid'],
        r'$OFI^{S}(\phi_{3})$': ['Cluster 3', 'Market-making cluster', 'darkviolet', 'solid'],
        r'$OFI^{S}(\phi_{*})$': ['No cluster', 'No cluster', 'green', 'solid'],
        r'$OFI^{C}(\phi_{1})$': ['Cluster 1', 'Directional cluster', 'blue', 'dashdot'],
        r'$OFI^{C}(\phi_{2})$': ['Cluster 2', 'Opportunistic cluster', 'red', 'dashdot'],
        r'$OFI^{C}(\phi_{3})$': ['Cluster 3', 'Market-making cluster', 'darkviolet', 'dashdot'],
        r'$OFI^{C}(\phi_{*})$': ['No cluster', 'No cluster', 'green', 'dashdot'],
    }
    fig = go.Figure()
    # Add a trace per strategy
    for col in list(scaled_PnL):
        fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL[col], mode='lines', legendgroup=trace_dict[col][0], 
                                legendgrouptitle_text=trace_dict[col][1], 
                                name= r''+col[:-1]+': SR = '+str(SR[col][0])+'\hspace{1mm} PPT = '+ str(PPT[col][0])+'$', 
                                line=dict(color=trace_dict[col][2], dash=trace_dict[col][3])))
    # Configure layout aesthetics
    fig.update_layout(
        xaxis_title="Dates in 2021",
        yaxis_title=f"Cumulative {FR} (%)",
        xaxis=dict(tickfont=dict(size=25), tickangle=0),
        yaxis=dict(tickfont=dict(size=25)),
        xaxis_title_font=dict(size=32),
        yaxis_title_font=dict(size=32),
        legend=dict(orientation='h', y=1, x=1.28,
                    xanchor='center', yanchor='top', font=dict(size=12)),
        legend_grouptitlefont_size=20,
        margin=dict(t=0, l=0, r=0, b=0, pad=0),
        width=1050,
        height=450
    )
    # Use every 20th date for x-axis ticks
    tick_vals = list(cumPnL.index[::20])
    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_vals,
        tickformat='%m-%d',
        tickangle=0,
        tickfont=dict(size=25)
    )
    # Save figure to PDF
    fig.write_image(f"{folder}/{data_type}_{FR}_{event_type}.pdf", scale=3, format='pdf')
    time.sleep(10)
    fig.write_image(f"{folder}/{data_type}_{FR}_{event_type}.pdf", scale=3, format='pdf')
    
def calc_cumPnL_plot(scaled_PnL, time_intervals, FR, event_type, folder, data_type):
    """
    Plot and save cumulative scaled PnL curves for each strategy separately.

    :param scaled_PnL:
        pandas.DataFrame of scaled daily PnL indexed by date, one column per strategy.
    :param list time_intervals:
        List of (start, end) time intervals for bucketing within the day.
    :param FR:
        Name of the target return, used in y-axis label and filename.
    :param event_type:
        Event category identifier (e.g., 'add', 'cancel', 'trade').
    :param folder:
        Directory path where the output PDF will be saved.
    :param data_type:
        Dataset type; must be either 'train' or 'test'.
    :returns:
        None. Saves a PDF plot of cumulative PnL for each strategy.
    :rtype: None
    """
    # Ensure the index is datetime for plotting
    scaled_PnL.index = pd.to_datetime(scaled_PnL.index)
    # Compute annualized Sharpe ratios for annotations
    SR = pd.DataFrame(scaled_PnL.apply(lambda col: round(np.mean(col) / np.std(col) * np.sqrt(252),2))).T.iloc[0, :].tolist()
    # Compute PnL per trade for annotations
    PPT = pd.DataFrame(scaled_PnL.apply(lambda col: round(np.mean(col)*10000/(len(time_intervals)-1),2))).T.iloc[0, :].tolist()
    # Calculate cumulative returns in percentage
    cumPnL = 100*scaled_PnL.cumsum()
    # Initialize Plotly figure
    fig = go.Figure()
    # Plot lines for column 1 and column 5
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 0], mode='lines', legendgroup="Cluster 1", legendgrouptitle_text= "Directional cluster", name= r'$OFI^{S}(\phi_{1}): SR = '+str(SR[0])+'\hspace{1mm} PPT = '+ str(PPT[0])+'$' , line=dict(color='blue', dash='solid')))
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 4], mode='lines', legendgroup="Cluster 1", name= r'$OFI^{C}(\phi_{1}): SR = '+str(SR[4])+'\hspace{1mm} PPT = '+ str(PPT[4])+'$', line=dict(color='blue', dash='dashdot')))
    # Plot lines for column 2 and column 6
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 1], mode='lines', legendgroup="Cluster 2", legendgrouptitle_text="Opportunistic cluster", name= r'$OFI^{S}(\phi_{2}): SR = '+str(SR[1])+'\hspace{1mm} PPT = '+ str(PPT[1])+'$', line=dict(color='red', dash='solid')))
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 5], mode='lines', legendgroup="Cluster 2", name= r'$OFI^{C}(\phi_{2}): SR = '+str(SR[5])+'\hspace{1mm} PPT = '+ str(PPT[5])+'$', line=dict(color='red', dash='dashdot')))
    # Plot lines for column 3 and column 7
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 2], mode='lines', legendgroup="Cluster 3", legendgrouptitle_text="Market-making cluster", name= r'$OFI^{S}(\phi_{3}): SR = '+str(SR[2])+'\hspace{1mm} PPT = '+ str(PPT[2])+'$', line=dict(color='darkviolet', dash='solid')))
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 6], mode='lines', legendgroup="Cluster 3", name= r'$OFI^{C}(\phi_{3}): SR = '+str(SR[6])+'\hspace{1mm} PPT = '+ str(PPT[6])+'$', line=dict(color='darkviolet', dash='dashdot')))
    # Plot lines for column 4 and column 8
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 3], mode='lines', legendgroup="No cluster", legendgrouptitle_text="No cluster", name= r'$OFI^{S}(\phi_{*}): SR = '+str(SR[3])+'\hspace{1mm} PPT = '+ str(PPT[3])+'$', line=dict(color='green', dash='solid')))
    fig.add_trace(go.Scatter(x=cumPnL.index, y=cumPnL.iloc[:, 7], mode='lines', legendgroup="No cluster", name= r'$OFI^{C}(\phi_{*}): SR = '+str(SR[7])+'\hspace{1mm} PPT = '+ str(PPT[7])+'$', line=dict(color='green', dash='dashdot')))
    # Configure layout: titles, fonts, legend
    fig.update_layout(
        xaxis_title= "Dates in 2021",
        yaxis_title= "Cumulative "+ FR +" (%)",
        xaxis_title_font=dict(size=32),
        yaxis_title_font=dict(size=32),
        xaxis=dict(tickfont=dict(size=25)),
        yaxis=dict(tickfont=dict(size=25)),
        xaxis_tickangle=0,
        legend=dict(orientation="h", y=1.8, x=0.5, xanchor='center', yanchor='top', font=dict(size= 12)),
        legend_grouptitlefont_size = 15,
        margin=dict(t=0, l=0, r=0, b=0, pad=0),
        width=800, 
        height=450
    )
    # Generate tick values by taking every 20th date from the index (cumPnL.index[0] is automatically included)
    tick_vals = list(cumPnL.index[::20])
    # Update the x-axis with these tick values, formatted to only show the date
    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_vals,
        tickformat='%m-%d',  # Display only the date
        tickangle=0,
        tickfont=dict(size=25)
    )
    # Save figure as high-resolution PDF
    fig.write_image(f'{folder}/{data_type}_{FR}_{event_type}.pdf', scale=3, format='pdf')
    time.sleep(10)
    fig.write_image(f'{folder}/{data_type}_{FR}_{event_type}.pdf', scale=3, format='pdf')

def calc_mean_median(winsorized_features, method):
    """
    Aggregate standardized features by label and compute an overall benchmark row.

    :param pandas.DataFrame winsorized_features:
        DataFrame containing winsorized standardized features and a 'Label' column.
    :param str method:
        Aggregation method: either "mean" to compute label means or "median" to compute label medians.
    :returns:
        pandas.DataFrame with one row per label ($\phi_{1}$, $\phi_{2}$, $\phi_{3}$) plus a benchmark row ($\phi_{*}$),
        columns renamed to ["V", "SBS", "OBS", "$T^{m}$", "$T^{1}$", "$T^{'}$"], and values rounded to 2 decimals.
    :rtype: pandas.DataFrame
    """
    std_feature_col = ["StdV","StdSBS","StdOBS","StdTM","StdT1","StdTPre"]
    # If computing means per label
    if method == "mean":
        # Group by Label, take mean of std_feature_col, round to 2 decimals
        winsorized_features = (
            winsorized_features
            .groupby('Label')
            .mean()
            .reset_index()
            .round(2)[std_feature_col]
        )
        # Compute overall benchmark as the mean across labels
        benchmark = winsorized_features.mean().round(2)
        # Convert series to single-row DataFrame with the same columns
        benchmark = pd.DataFrame(benchmark.values.reshape(1, -1), columns=benchmark.index)[std_feature_col]
        # Append benchmark row
        winsorized_features = pd.concat([winsorized_features, benchmark])
    # If computing medians per label
    if method == "median":
        # Group by Label, take median of std_feature_col, round to 2 decimals
        winsorized_features = (
            winsorized_features
            .groupby('Label')
            .median()
            .reset_index()
            .round(2)[std_feature_col]
        )
        # Compute overall benchmark as the median across labels
        benchmark = winsorized_features.median().round(2)
        # Convert series to single-row DataFrame with the same columns
        benchmark = pd.DataFrame(benchmark.values.reshape(1, -1), columns=benchmark.index)[std_feature_col]
        # Append benchmark row
        winsorized_features = pd.concat([winsorized_features, benchmark])
    # Rename columns to descriptive feature names
    winsorized_features.columns = ["V", "SBS", "OBS", r"$T^{m}$", r"$T^{1}$", r"$T^{'}$"]
    # Set index to LaTeX labels for each row (φ₁, φ₂, φ₃, φ_*)
    winsorized_features.index = [r'$\phi_{1}$', r'$\phi_{2}$', r'$\phi_{3}$', r'$\phi_{*}$']
    return winsorized_features

def calc_mean_median_heatmap(winsorized_features, event_type, folder, ticker, method, data_type):
    """
    Generate and save a heatmap of aggregated feature statistics.

    :param winsorized_features:
        pandas.DataFrame of aggregated statistics indexed by label ($\phi_{1}$, $\phi_{2}$, $\phi_{3}$, $\phi_{*}$)
        with standardized feature columns.
    :param event_type:
        String identifier for the event category (e.g., 'A', 'L', 'M').
    :param folder:
        Directory path where the heatmap PDF will be saved.
    :param ticker:
        Security symbol used in the output filename.
    :param method:
        Aggregation method; either 'mean' or 'median'.
    :param data_type:
        Dataset type, either 'train' or 'test'.
    :returns:
        None. Saves a PDF heatmap to the specified folder.
    """
    # Set figure size based on number of columns (width) and rows (height)
    plt.figure(figsize=(len(winsorized_features.columns), len(winsorized_features)))
    # Draw heatmap with annotations
    ax = sns.heatmap(
        winsorized_features,
        cmap='YlOrRd',
        annot=True,
        fmt='g',
        cbar=True
    )
    # Keep y-axis labels horizontal for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # Save the heatmap as a high-resolution PDF
    plt.savefig(
        f'{folder}/{data_type}_{ticker}_{method}_heatmap_{event_type}.pdf',
        dpi=1000,
        bbox_inches="tight"
    )

def calc_stock_parallel(ticker, daily_date_pairs, event_type, moving_size, time_intervals, j):
    """
    Load, filter, and process LOBSTER data for a single stock ticker in parallel.

    :param str ticker:
        Security symbol to process.
    :param tuple daily_date_pairs:
        (start_time, end_time) pair defining the trading day window.
    :param str event_type:
        Event category identifier (e.g., 'A', 'L', 'D', 'M').
    :param int moving_size:
        Window size for rolling mean and standard deviation when standardizing features.
    :param list time_intervals:
        List of (start, end) time intervals for bucketing within the day.
    :param int j:
        Batch index used to offset bucket numbering.
    :returns:
        tuple:
          - features_extra (pandas.DataFrame): engineered features if successful, otherwise [].
          - returns (pandas.DataFrame): interval return data if successful, otherwise [].
    :rtype: tuple of pandas.DataFrame or ([], [])
    """
    # Unpack the daily date range
    (start_time, end_time) = daily_date_pairs
    # Load raw LOBSTER features for this ticker and date range
    features = rapid_daily_data_processing(ticker, event_type, start_time, end_time)
    # Reset index so that DateTime is a column again
    features = features.reset_index(drop=True)
    # Proceed only if we have data
    if not features.empty:
        # Convert nanosecond timestamps into Python time and insert as RealTime column
        RealTime = features['Time'].apply(convert_seconds_to_time)
        features.insert(0, "RealTime", RealTime)
        # Define regular trading hours window (09:30–16:00)
        start_trading = datetime.datetime.strptime('09:30:00', '%H:%M:%S').time()
        end_trading   = datetime.datetime.strptime('16:00:00', '%H:%M:%S').time()
        features = features[
            features['RealTime'].apply(lambda x: start_trading <= x <= end_trading)
        ]
        # Further filter to include only ticks before 10:00
        cutoff = datetime.datetime.strptime('10:00:00', '%H:%M:%S').time()
        filtered_features = features[features['RealTime'] < cutoff]
        if not filtered_features.empty:
            # Compute engineered features and interval returns
            features_extra, returns = feature_processing(features, moving_size, time_intervals, j)
            # Check for NaNs and complete bucket coverage
            complete_buckets = len(features_extra["Bucket"].unique()) == len(time_intervals)
            if not (features.isna().any().any() or
                    features_extra.isna().any().any() or
                    not complete_buckets):

                print(j)          # Log current batch index
                j += 1            # Increment batch index
                # Drop the final bucket from features_extra
                features_extra = features_extra[
                    features_extra['Bucket'] != j * len(time_intervals)
                ]
                # Compute log returns for each interval
                returns["CONR"] = np.log(returns["EndMidPrice"] / returns["StartMidPrice"])
                returns['FRNB'] = np.log(returns["EndMidPrice"].shift(-1) / returns["EndMidPrice"])
                returns['FREB'] = np.log(returns["EndMidPrice"].iloc[-1] / returns["EndMidPrice"])
                # Drop the last interval
                returns = returns.drop(returns.index[-1])
                # Tag each return with the trading date
                returns["Dates"] = features["DateTime"][0].date()
                # Clean up raw features to free memory
                del features
                return features_extra, returns 
            else:
                return [], []
        else:
            return [], []
    else:
        return [], []

def calc_result(winsorized_features, returns_result, ticker):
    """
    Combine imbalance metrics and interval returns into a single result table.

    :param pandas.DataFrame winsorized_features:
        Winsorized standardized features with a 'Bucket' and 'Label' column, used to compute imbalances.
    :param pandas.DataFrame returns_result:
        DataFrame of interval returns with columns ['StartInterval', 'EndInterval', 'StartMidPrice',
        'EndMidPrice', 'CONR', 'FRNB', 'FREB', 'Dates'].
    :param str ticker:
        Security symbol to tag the result table.
    :returns:
        pandas.DataFrame with one row per time bucket, containing:
          - Size‐based OFI metrics for each label and benchmark (`$OFI^{S}(\phi_i)$` and `$OFI^{S}(\phi_*)$`)
          - Count‐based OFI metrics for each label and benchmark (`$OFI^{C}(\phi_i)$` and `$OFI^{C}(\phi_*)$`)
          - Interval return columns from `returns_result`
          - A final `Name` column equal to `ticker`
    :rtype: pandas.DataFrame
    """
    # Compute size and count imbalances by label
    imbalance_size, imbalance_count = calc_imbalance(winsorized_features)
    # Compute benchmark (overall) size and count imbalances
    benchmark_imbalance_size, benchmark_imbalance_count = calc_benchmark_imbalance(winsorized_features)
    # Concatenate label‐level and benchmark metrics with the returns DataFrame
    result = pd.concat([
        imbalance_size.reset_index(drop=True),
        benchmark_imbalance_size.reset_index(drop=True),
        imbalance_count.reset_index(drop=True),
        benchmark_imbalance_count.reset_index(drop=True),
        returns_result
    ], axis=1)
    # Tag each row with the ticker symbol
    result["Name"] = ticker
    return result

def calc_train_base_result(train_features_extra_result, k, train_returns_result, ticker):
    """
    Train base KMeans clustering on winsorized features and assemble training results.

    :param pandas.DataFrame train_features_extra_result:
        DataFrame containing engineered features including standardized columns (std_feature_col),
        'SignSize', and 'Bucket' for each observation.
    :param int k:
        Number of clusters for KMeans.
    :param pandas.DataFrame train_returns_result:
        DataFrame of interval returns with matching index to `train_features_extra_result`.
    :param str ticker:
        Security symbol to tag the resulting DataFrame.
    :returns:
        tuple:
          - train_base_kmeans (KMeans): Fitted KMeans clustering model.
          - train_result (pandas.DataFrame): Combined imbalance and return results via `calc_result`.
          - train_mean_winsorized_features (pandas.DataFrame): Label means + benchmark row via `calc_mean_median`.
          - train_median_winsorized_features (pandas.DataFrame): Label medians + benchmark row via `calc_mean_median`.
    :rtype: tuple
    """
    # Select only standardized feature columns
    std_feature_col = ["StdV","StdSBS","StdOBS","StdTM","StdT1","StdTPre"]
    train_features = train_features_extra_result[std_feature_col]
    # Winsorize each feature to limit outliers at 1st and 99th percentiles
    train_winsorized_features = train_features.apply(lambda x: winsorize(x, limits=[0.01, 0.01]))
    # Fit KMeans on the winsorized features
    train_base_kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42) \
        .fit(train_winsorized_features)
    # Predict cluster labels and shift to 1-based indexing
    train_labels = train_base_kmeans.predict(train_winsorized_features) + 1
    train_labels_col = pd.DataFrame(train_labels, columns=['Label'])
    # Combine winsorized features with SignSize, Bucket, and label columns
    train_winsorized_features = pd.concat([
        train_winsorized_features.reset_index(drop=True),
        train_features_extra_result[["SignSize", "Bucket"]].reset_index(drop=True),
        train_labels_col.reset_index(drop=True)
    ], axis=1)
    # Compute combined imbalance & return results
    train_result = calc_result(train_winsorized_features, train_returns_result, ticker)
    # Compute label means + benchmark row
    train_mean_winsorized_features = calc_mean_median(train_winsorized_features, "mean")
    # Compute label medians + benchmark row
    train_median_winsorized_features = calc_mean_median(train_winsorized_features, "median")
    return train_base_kmeans, train_result, train_mean_winsorized_features, train_median_winsorized_features

def calc_train_result(train_features_extra_result, train_base_kmeans, k, train_returns_result, ticker):
    """
    Refine clustering on training data using base centroids and assemble updated results.

    :param pandas.DataFrame train_features_extra_result:
        DataFrame containing engineered features including standardized columns (std_feature_col),
        'SignSize', and 'Bucket' for each observation.
    :param KMeans train_base_kmeans:
        Fitted KMeans model from the base training run, used to extract initial centroids.
    :param int k:
        Number of clusters to use in the refined clustering.
    :param pandas.DataFrame train_returns_result:
        DataFrame of interval returns with matching index to `train_features_extra_result`.
    :param str ticker:
        Security symbol to tag the resulting DataFrame.
    :returns:
        tuple:
          - train_kmeans (KMeans): Re-fitted KMeans model initialized at base centroids.
          - train_result (pandas.DataFrame): Combined imbalance and return results via `calc_result`.
          - train_mean_winsorized_features (pandas.DataFrame): Label means + benchmark row via `calc_mean_median`.
          - train_median_winsorized_features (pandas.DataFrame): Label medians + benchmark row via `calc_mean_median`.
    :rtype: tuple
    """
    # Extract only the standardized feature columns
    std_feature_col = ["StdV","StdSBS","StdOBS","StdTM","StdT1","StdTPre"]
    train_features = train_features_extra_result[std_feature_col]
    # Winsorize each feature to cap outliers at the 1st and 99th percentiles
    train_winsorized_features = train_features.apply(lambda x: winsorize(x, limits=[0.01, 0.01]))
    # Retrieve centroids from the base model for initialization
    centroids = train_base_kmeans.cluster_centers_
    # Re-fit KMeans using the base centroids to refine clusters
    train_kmeans = KMeans(n_clusters=k, init=centroids, random_state=42).fit(train_winsorized_features)

    # Predict new cluster labels and shift to 1-based indexing
    train_labels = train_kmeans.predict(train_winsorized_features) + 1
    train_labels_col = pd.DataFrame(train_labels, columns=['Label'])
    # Combine winsorized features with SignSize, Bucket, and new labels
    train_winsorized_features = pd.concat([
        train_winsorized_features.reset_index(drop=True),
        train_features_extra_result[["SignSize", "Bucket"]].reset_index(drop=True),
        train_labels_col.reset_index(drop=True)
    ], axis=1)
    # Compute combined imbalance metrics and returns
    train_result = calc_result(train_winsorized_features, train_returns_result, ticker)
    # Compute label means + benchmark row
    train_mean_winsorized_features = calc_mean_median(train_winsorized_features, "mean")
    # Compute label medians + benchmark row
    train_median_winsorized_features = calc_mean_median(train_winsorized_features, "median")
    # Return the refined model and result tables
    return train_kmeans, train_result, train_mean_winsorized_features, train_median_winsorized_features

def calc_test_result(test_features_extra_result, test_returns_result, train_kmeans, mapping_dict, event_type, folder, ticker):
    """
    Apply trained clusters to test data, compute imbalance & return results, and generate heatmaps.

    :param pandas.DataFrame test_features_extra_result:
        Engineered features including standardized columns (std_feature_col),
        'SignSize', and 'Bucket' for each test observation.
    :param pandas.DataFrame test_returns_result:
        DataFrame of interval returns for the test set, indexed to match features.
    :param KMeans train_kmeans:
        Fitted KMeans model from training, used to predict test labels.
    :param dict mapping_dict:
        Mapping from raw cluster labels to desired label numbers.
    :param str event_type:
        Identifier for the event category (e.g., 'A', 'L', 'M').
    :param str folder:
        Directory path where output heatmaps will be saved.
    :param str ticker:
        Security symbol used in output filenames.
    :returns:
        pandas.DataFrame:
        Combined imbalance and return results for the test set, with a `Name` column.
    :rtype: pandas.DataFrame
    """
    # Select only standardized feature columns from test data
    std_feature_col = ["StdV","StdSBS","StdOBS","StdTM","StdT1","StdTPre"]
    test_features = test_features_extra_result[std_feature_col]
    # Winsorize to limit extreme values
    test_winsorized_features = test_features.apply(lambda x: winsorize(x, limits=[0.01, 0.01]))
    # Predict cluster labels using the trained model, then shift to 1-based
    test_labels = train_kmeans.predict(test_winsorized_features) + 1
    test_labels_col = pd.DataFrame(test_labels, columns=['Label'])
    # Remap raw labels to consistent numbering
    test_labels_col['Label'] = test_labels_col['Label'].map(mapping_dict)
    # Combine standardized features, SignSize, Bucket, and mapped labels
    test_winsorized_features = pd.concat([
        test_winsorized_features.reset_index(drop=True),
        test_features_extra_result[["SignSize", "Bucket"]].reset_index(drop=True),
        test_labels_col.reset_index(drop=True)
    ], axis=1)
    # Compute combined imbalance metrics and returns
    test_result = calc_result(test_winsorized_features, test_returns_result, ticker)
    # Compute label-wise mean & benchmark, then plot heatmap
    test_mean_winsorized_features = calc_mean_median(test_winsorized_features, "mean")
    calc_mean_median_heatmap(
        test_mean_winsorized_features,
        event_type, folder, ticker,
        method="mean",
        data_type="test"
    )
    # Compute label-wise median & benchmark, then plot heatmap
    test_median_winsorized_features = calc_mean_median(test_winsorized_features, "median")
    calc_mean_median_heatmap(
        test_median_winsorized_features,
        event_type, folder, ticker,
        method="median",
        data_type="test"
    )
    return test_result

def identify_clusters(matrix):
    """
    Determine cluster assignments for directional, opportunistic, and market‐making clusters.

    This function analyzes a metric matrix with 3 rows (one per cluster) and 3 columns:
      - Column 0 contains directional metrics.
      - Column 2 contains opportunistic metrics.
    It selects:
      - **directional**: the row with the maximum value in column 0.
      - **opportunistic**: the row with the maximum value in column 2.
      - **market‐making**: the remaining row.

    In case the same row maximizes both columns 0 and 2, the conflict is resolved by:
      1. Comparing the two metrics at the conflicting row.
      2. Assigning the row to whichever role has the larger metric.
      3. Selecting the other role from the remaining rows based on the highest value in the corresponding column.
      4. Assigning the last row as market‐making.

    :param numpy.ndarray matrix:
        Array of shape (3, n) where `matrix[i, j]` is the metric for cluster `i` on feature `j`.
    :returns: dict[str,int]
        Mapping of cluster roles to 1-based cluster indices:
          - `"directional"`: cluster index (1–3)
          - `"opportunistic"`: cluster index (1–3)
          - `"market‐making"`: cluster index (1–3)
    """
    # Find row index with max directional metric (column 0)
    dir_idx = np.argmax(matrix[:, 0])
    # Find row index with max opportunistic metric (column 2)
    opp_idx = np.argmax(matrix[:, 2])
    
    if dir_idx != opp_idx:
        # No conflict: assign directly
        directional_idx = dir_idx
        opportunistic_idx = opp_idx
        # The leftover row is market‐making
        market_making_idx = list(set(range(3)) - {dir_idx, opp_idx})[0]
    else:
        # Conflict: same row for both roles
        conflict_idx = dir_idx
        remaining = list(set(range(3)) - {conflict_idx})
        if matrix[conflict_idx, 0] > matrix[conflict_idx, 2]:
            # If the directional metric is higher, assign conflict row to directional
            directional_idx = conflict_idx
            # Among remaining, pick the highest opportunistic metric for opportunistic
            opportunistic_idx = remaining[np.argmax(matrix[remaining, 2])]
            # Last one is market‐making
            market_making_idx = list(set(remaining) - {opportunistic_idx})[0]
        else:
            # Otherwise assign conflict row to opportunistic
            opportunistic_idx = conflict_idx
            # Among remaining, pick the highest directional metric for directional
            directional_idx = remaining[np.argmax(matrix[remaining, 0])]
            # Last one is market‐making
            market_making_idx = list(set(remaining) - {directional_idx})[0]
    # Return as 1-based indices for readability
    return {
        "directional": directional_idx + 1,
        "opportunistic": opportunistic_idx + 1,
        "market-making": market_making_idx + 1
    }

def calc_daily_date_pairs(monthly_pairs):
    """
    Generate a list of consecutive daily date pairs from a start to an end date.

    :param tuple monthly_pairs:
        A two‐element tuple (start_time, end_time), where both are `datetime.date` or
        `datetime.datetime` objects defining the range.
    :returns:
        List of `(date1, date2)` tuples for each day in the range,
        where `date1` is the previous day and `date2` is the next day.
    :rtype: list of tuple
    """
    # Unpack the start and end of the monthly range
    (start_time, end_time) = monthly_pairs
    date_pairs = []
    # Loop from 1 to number of days between start and end (exclusive of end)
    for i in range(1, (end_time - start_time).days):
        # Compute the pair: yesterday and today
        date1 = start_time + datetime.timedelta(days=i-1)
        date2 = start_time + datetime.timedelta(days=i)
        date_pairs.append((date1, date2))
    return date_pairs

def calc_monthly_date_pairs(start_time, end_time):
    """
    Generate sequential start/end pairs for each month in a date range.

    :param start_time:
        The beginning of the overall period (datetime.date or datetime.datetime).
    :type start_time: datetime.date or datetime.datetime
    :param end_time:
        The end of the overall period (datetime.date or datetime.datetime).
    :type end_time: datetime.date or datetime.datetime
    :returns:
        List of tuples `(month_start, month_end)` for each month between
        `start_time` (inclusive) and `end_time` (exclusive of `month_end` unless it equals `end_time`).
    :rtype: list of tuple
    """
    # Prepare the list to hold each month's date pair
    month_date_pair_list = []
    # Begin at the first day of the starting month
    current_month = start_time
    # Continue until we reach or pass the end_time
    while current_month < end_time:
        # Move to the first day of the next month by advancing 32 days and resetting to day=1
        next_month = current_month.replace(day=1) + datetime.timedelta(days=32)
        next_month = next_month.replace(day=1)
        # If overshooting, cap at end_time
        if next_month > end_time:
            next_month = end_time
        # Record the pair for this month
        month_date_pair_list.append((current_month, next_month))
        # Advance to process the next interval
        current_month = next_month
    return month_date_pair_list

def process_stock_data(ticker, monthly_date_pairs, event_type, moving_size, time_intervals):
    """
    Load and process LOBSTER data for a ticker across multiple monthly periods.

    This function:
      1. Splits each monthly range into daily pairs.
      2. Parallel‐processes each trading day to compute engineered features and interval returns.
      3. Aggregates daily results into two DataFrames: 
         - `features_extra_result` with extra features per order.
         - `returns_result` with interval returns per day.

    :param str ticker:
        Security symbol to process.
    :param list of tuple monthly_date_pairs:
        List of (start_time, end_time) pairs defining each monthly window.
    :param str event_type:
        Identifier for the event category (e.g. 'A', 'L', 'M').
    :param int moving_size:
        Window size for rolling mean and standard deviation when standardizing features.
    :param list time_intervals:
        Intraday intervals used for bucketing ticks.
    :returns:
        tuple:
          - features_extra_result (pandas.DataFrame): concatenated extra features across all days.
          - returns_result (pandas.DataFrame): concatenated interval returns with columns 
            ['CONR', 'FRNB', 'FREB', 'Dates'].
    :rtype: tuple of pandas.DataFrame
    """
    # Prepare empty DataFrames to accumulate daily results
    features_extra_result = pd.DataFrame()
    returns_result = pd.DataFrame()
    i = 0  # Counter for month index
    # Loop over each month in the input list
    for monthly_pairs in monthly_date_pairs:
        # Break the month range into individual trading days
        daily_date_pairs = calc_daily_date_pairs(monthly_pairs)
        # Parallel‐execute calc_stock_parallel for each trading day
        r = Parallel(n_jobs=-1)(
            delayed(calc_stock_parallel)(
                ticker, daily, event_type, moving_size, time_intervals, j + i * 31
            ) for j, daily in enumerate(daily_date_pairs)
        )
        # Unzip the returned list of tuples into separate lists
        features_extra, returns = zip(*r)
        # Filter out any days with no data
        features_extra = [f for f in features_extra if len(f) > 0]
        returns = [ret for ret in returns if len(ret) > 0]             
        # Concatenate all daily extra‐feature DataFrames for this month
        for features_extra_daily in features_extra:
            features_extra_result = pd.concat([features_extra_result, features_extra_daily])
        features_extra_result = features_extra_result.reset_index(drop=True)
        # Concatenate all daily return DataFrames for this month
        for returns_daily in returns:
            returns_result = pd.concat([returns_result, returns_daily])
        returns_result = returns_result.reset_index(drop=True)       
        # Clean up temporary variables and advance month counter
        del r, features_extra, returns
        i += 1
    # Keep only the final return columns of interest
    returns_result = returns_result[["CONR", "FRNB", "FREB", "Dates"]]
    return features_extra_result, returns_result

def calc_mode_dict(all_dict):
    """
    Select a unique “mode” value for each key from lists of candidates.

    For each key in `all_dict`, this function:
      1. Computes the frequency of each candidate value in the list.
      2. Chooses the most frequent candidate that hasn’t already been assigned to another key.
      3. If all top candidates are already used, it falls back to the overall most frequent.

    :param dict all_dict:
        Mapping from each key to a list of candidate values (possibly empty).
    :returns:
        dict: A `mode_dict` mapping each key to its selected value, or `None`
        if the original list was empty. Ensures no two keys share the same value
        unless unavoidable.
    :rtype: dict
    """
    # Initialize the output dict and track values already assigned
    mode_dict = {}
    used_values = set()
    # Process each key and its list of candidate values
    for key, values in all_dict.items():
        # If there are no candidates, assign None
        if not values:
            mode_dict[key] = None
            continue
        # Count how often each candidate appears
        counter = Counter(values)
        # Sort candidates by descending frequency
        candidates = counter.most_common()
        chosen = None
        # Pick the most frequent candidate that hasn't been used yet
        for candidate, freq in candidates:
            if candidate not in used_values:
                chosen = candidate
                break
        # If all frequent candidates are already used, fallback to the top one
        if chosen is None:
            chosen = candidates[0][0]
        # Record the chosen mode and mark it as used
        mode_dict[key] = chosen
        used_values.add(chosen)
    return mode_dict

def calc_mapping_index(df, map_dict, index):
    """
    Reorder and reindex a DataFrame based on a mapping dictionary.

    :param pandas.DataFrame df:
        Input DataFrame to be reordered.
    :param dict map_dict:
        Mapping from the original integer positions (after reset) to a sort key.
    :param list index:
        New index labels to assign after sorting.
    :returns:
        pandas.DataFrame: The DataFrame sorted according to `map_dict` and
        reindexed with `index`.
    :rtype: pandas.DataFrame
    """
    # Reset the DataFrame index to simple integer positions
    df = df.reset_index(drop=True)
    # Map each row position to a sort key and store temporarily
    df['map'] = df.index.map(map_dict)
    # Sort rows by the mapped key, then remove the temporary column
    df = df.sort_values('map').drop('map', axis=1)
    # Assign the provided index labels
    df.index = index
    return df