# Import cluster_lob
import cluster_lob
import pandas as pd
import datetime
import time
# Small tick stocks
small_tick_stocks = ["CHTR","GOOG","GS","IBM","MCD","NVDA"]
ticker_list = small_tick_stocks
folder = "small_tick_stocks"
# Medium tick stocks
medium_tick_stocks = ["AAPL","ABBV","PM"]
ticker_list = medium_tick_stocks
folder = "medium_tick_stocks"
# Large tick stocks
large_tick_stocks = ["CMCSA", "CSCO", "INTC", "MSFT","KO","VZ"]
ticker_list = large_tick_stocks
folder = "large_tick_stocks"
# Setup
k = 3
moving_size = 100
phi_index = [r'$\phi_{1}$', r'$\phi_{2}$', r'$\phi_{3}$', r'$\phi_{*}$']
ofi_cluster_index = [r'$OFI^{S}(\phi_{1})$', r'$OFI^{S}(\phi_{2})$', r'$OFI^{S}(\phi_{3})$',
                     r'$OFI^{C}(\phi_{1})$', r'$OFI^{C}(\phi_{2})$', r'$OFI^{C}(\phi_{3})$']
ofi_no_cluster_index = [r"$OFI^{S}(\phi_{*})$", r'$OFI^{C}(\phi_{*})$']
ofi_all_index = [r'$OFI^{S}(\phi_{1})$', r'$OFI^{S}(\phi_{2})$', r'$OFI^{S}(\phi_{3})$', r"$OFI^{S}(\phi_{*})$",
                 r'$OFI^{C}(\phi_{1})$', r'$OFI^{C}(\phi_{2})$', r'$OFI^{C}(\phi_{3})$', r"$OFI^{C}(\phi_{*})$"]
default_dict = {"directional": 1, "opportunistic": 2, "market-making": 3}
# Train period
train_start_time = datetime.datetime(2021, 1, 1)
train_end_time = datetime.datetime(2021, 7, 1)
# Test period
test_start_time = datetime.datetime(2021, 7, 1)
test_end_time = datetime.datetime(2021, 12, 31)
# Buckets
time_intervals = [(datetime.time(9, 30), datetime.time(10, 0)), (datetime.time(10, 0), datetime.time(10, 30)), 
                  (datetime.time(10, 30), datetime.time(11, 0)),(datetime.time(11, 0), datetime.time(11, 30)), 
                  (datetime.time(11, 30), datetime.time(12, 0)), (datetime.time(12, 0), datetime.time(12, 30)),
                  (datetime.time(12, 30), datetime.time(13, 0)), (datetime.time(13, 0), datetime.time(13, 30)), 
                  (datetime.time(13, 30), datetime.time(14, 0)),(datetime.time(14, 0), datetime.time(14, 30)), 
                  (datetime.time(14, 30), datetime.time(15, 0)), (datetime.time(15, 0), datetime.time(15, 30)),
                  (datetime.time(15, 30), datetime.time(16, 0))]
# Main function
def main():
    """
    Main pipeline to perform LOBSTER clustering, evaluation, and visualization across train and test sets.

    Executes the following steps for each event_type in ["A","L","D","M"]:
      1. Train clustering models on training data for all tickers.
      2. Identify cluster roles and determine mapping to canonical clusters.
      3. Generate heatmaps of feature statistics and correlation heatmaps.
      4. Compute PnL metrics, generate cumulative PnL plots for both all and top clusters.
      5. Apply trained clusters to test data, produce analogous results and visualizations.

    Relies on global configuration variables:
    ticker_list, train_start_time, train_end_time, test_start_time,
    time_intervals, k, default_dict, phi_index, ofi_all_index,
    ofi_cluster_index, ofi_no_cluster_index, folder.
    """
    # Loop over each event category
    for event_type in ["A", "L", "D", "M"]:
        print(event_type)
        # Collect cluster selections across all tickers
        all_dict = {"directional": [], "opportunistic": [], "market-making": []}
        print("Train:")
        # Prepare dictionaries and lists for training outputs
        train_kmeans_dict = {}
        train_mean_winsorized_features_dict = {}
        train_median_winsorized_features_dict = {}
        train_correlation_dict = {}
        train_FRNB_PnL_list = []
        train_FREB_PnL_list = []
        # Process each ticker for training
        for ticker in ticker_list:
            print(ticker)
            # Generate monthly date windows for training period
            train_monthly_date_pairs = cluster_lob.calc_monthly_date_pairs(train_start_time, train_end_time)
            # Compute daily features and returns in parallel
            train_features_extra_result, train_returns_result = cluster_lob.process_stock_data(
                ticker, train_monthly_date_pairs, event_type, moving_size, time_intervals
            )
            # Initialize or refine the clustering model
            if ticker == ticker_list[0]:
                # For first ticker, train base KMeans
                train_base_kmeans, train_result, train_mean_winsorized_features, train_median_winsorized_features = \
                    cluster_lob.calc_train_base_result(train_features_extra_result, k, train_returns_result, ticker)
                train_kmeans = train_base_kmeans
            else:
                # For subsequent tickers, refine clustering using base centroids
                train_kmeans, train_result, train_mean_winsorized_features, train_median_winsorized_features = \
                    cluster_lob.calc_train_result(train_features_extra_result, train_base_kmeans, k, train_returns_result, ticker)
            # Store per-ticker clustering objects and feature stats
            train_kmeans_dict[ticker] = train_kmeans
            train_mean_winsorized_features_dict[ticker] = train_mean_winsorized_features
            train_median_winsorized_features_dict[ticker] = train_median_winsorized_features
            # Free memory
            del train_features_extra_result
            # Compute correlations between OFI features and target returns
            train_correlation = cluster_lob.calc_correlation(train_result, k)
            train_correlation_dict[ticker] = train_correlation
            # Identify cluster roles from correlation matrices
            dict1 = cluster_lob.identify_clusters(train_correlation.iloc[0:3].to_numpy())
            dict2 = cluster_lob.identify_clusters(train_correlation.iloc[4:7].to_numpy())
            merged_dict = {key: [dict1[key], dict2[key]] for key in dict1.keys()}
            # Accumulate choices across tickers
            for key, value_list in merged_dict.items():
                all_dict[key].extend(value_list)
            # Compute PnL series for FRNB and scale to target vol
            train_FRNB_PnL = cluster_lob.calc_PnL(train_result, "FRNB", k)
            train_FRNB_scaled_PnL = cluster_lob.calc_scaled_PnL(train_FRNB_PnL)
            train_FRNB_PnL_list.append(train_FRNB_scaled_PnL)
            # Compute PnL series for FREB and scale to target vol
            train_FREB_PnL = cluster_lob.calc_PnL(train_result, "FREB", k)
            train_FREB_scaled_PnL = cluster_lob.calc_scaled_PnL(train_FREB_PnL)
            train_FREB_PnL_list.append(train_FREB_scaled_PnL)
        # Determine modal cluster assignments
        print("All:", all_dict)
        mode_dict = cluster_lob.calc_mode_dict(all_dict)
        print("Mode:", mode_dict)
        # Map modes to default cluster indices
        mapping_dict = {mode_dict[key]: default_dict[key] for key in mode_dict}
        print("Map:", mapping_dict)
        # Build mapping for standardized
        new_mapping_dict = {key - 1: value - 1 for key, value in mapping_dict.items()}
        new_mapping_dict[k] = k
        extended_mapping_dict = new_mapping_dict.copy()
        for key, value in new_mapping_dict.items():
            extended_mapping_dict[key + 4] = value + 4
        print("Extended Map:", extended_mapping_dict)
        # Generate heatmaps for training feature statistics and OFI correlations
        for ticker in ticker_list:
            # Mean-based feature heatmap
            train_mean_winsorized_features = cluster_lob.calc_mapping_index(
                train_mean_winsorized_features_dict[ticker],
                new_mapping_dict,
                phi_index
            )
            cluster_lob.calc_mean_median_heatmap(
                train_mean_winsorized_features,
                event_type, folder, ticker, "mean", "train"
            )
            # Median-based feature heatmap
            train_median_winsorized_features = cluster_lob.calc_mapping_index(
                train_median_winsorized_features_dict[ticker],
                new_mapping_dict,
                phi_index
            )
            cluster_lob.calc_mean_median_heatmap(
                train_median_winsorized_features,
                event_type, folder, ticker, "median", "train"
            )
            # Correlation heatmap for OFI vs. targets
            train_correlation = cluster_lob.calc_mapping_index(
                train_correlation_dict[ticker],
                extended_mapping_dict,
                ofi_all_index
            )
            cluster_lob.calc_correlation_heatmap(
                train_correlation,
                event_type, folder, ticker, "train"
            )
        # Aggregate and plot PnL metrics for FRNB
        train_FRNB_PnL_mean = pd.concat(train_FRNB_PnL_list).groupby(level=0).mean()
        train_FRNB_scaled_PnL_mean = cluster_lob.calc_scaled_PnL(train_FRNB_PnL_mean)
        train_FRNB_scaled_PnL_mean = cluster_lob.calc_mapping_index(
            train_FRNB_scaled_PnL_mean.T,
            extended_mapping_dict,
            ofi_all_index
        ).T
        train_FRNB_metric = cluster_lob.calc_metric(
            train_FRNB_scaled_PnL_mean, time_intervals, 
            "FRNB", event_type, folder, "train_all"
        )
        train_FRNB_top_sr = train_FRNB_metric.loc['Sharpe ratio', ofi_cluster_index].nlargest(1).index.tolist()
        train_FRNB_top_sr = sorted(train_FRNB_top_sr, key=lambda x: ofi_cluster_index.index(x))
        # Cumulative PnL plots: all clusters
        cluster_lob.calc_cumPnL_plot(
            train_FRNB_scaled_PnL_mean, time_intervals, 
            "FRNB", event_type, folder, "train_all"
        )
        # Top cluster PnL and plot
        _ = cluster_lob.calc_metric(
            train_FRNB_scaled_PnL_mean[train_FRNB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FRNB", event_type, folder, "train_top"
        )
        cluster_lob.calc_top_cumPnL_plot(
            train_FRNB_scaled_PnL_mean[train_FRNB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FRNB", event_type, folder, "train_top"
        )
        # Aggregate and plot PnL metrics for FREB
        train_FREB_PnL_mean = pd.concat(train_FREB_PnL_list).groupby(level=0).mean()
        train_FREB_scaled_PnL_mean = cluster_lob.calc_scaled_PnL(train_FREB_PnL_mean)
        train_FREB_scaled_PnL_mean = cluster_lob.calc_mapping_index(
            train_FREB_scaled_PnL_mean.T,
            extended_mapping_dict,
            ofi_all_index
        ).T
        train_FREB_metric = cluster_lob.calc_metric(
            train_FREB_scaled_PnL_mean, time_intervals, 
            "FREB", event_type, folder, "train_all"
        )
        train_FREB_top_sr = train_FREB_metric.loc['Sharpe ratio', ofi_cluster_index].nlargest(1).index.tolist()
        train_FREB_top_sr = sorted(train_FREB_top_sr, key=lambda x: ofi_cluster_index.index(x))
        cluster_lob.calc_cumPnL_plot(
            train_FREB_scaled_PnL_mean, time_intervals, 
            "FREB", event_type, folder, "train_all"
        )
        _ = cluster_lob.calc_metric(
            train_FREB_scaled_PnL_mean[train_FREB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FREB", event_type, folder, "train_top"
        )
        cluster_lob.calc_top_cumPnL_plot(
            train_FREB_scaled_PnL_mean[train_FREB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FREB", event_type, folder, "train_top"
        )
        print("Test:")
        # Prepare lists for test dataset PnL
        test_FRNB_PnL_list = []
        test_FREB_PnL_list = []
        for ticker in ticker_list:
            print(ticker)
            # Generate monthly date windows for test period
            test_monthly_date_pairs = cluster_lob.calc_monthly_date_pairs(test_start_time, test_end_time)
            # Compute daily features and returns for test set
            test_features_extra_result, test_returns_result = cluster_lob.process_stock_data(
                ticker, test_monthly_date_pairs, event_type, moving_size, time_intervals
            )
            # Apply trained clustering to test features
            train_kmeans = train_kmeans_dict[ticker]
            test_result = cluster_lob.calc_test_result(
                test_features_extra_result,
                test_returns_result,
                train_kmeans,
                mapping_dict,
                event_type,
                folder,
                ticker
            )
            del test_features_extra_result, test_returns_result
            # Correlation heatmap for test set
            test_correlation = cluster_lob.calc_correlation(test_result, k)
            cluster_lob.calc_correlation_heatmap(
                test_correlation,
                event_type, folder, ticker, "test"
            )
            # Compute and collect test PnL series for FRNB
            test_FRNB_PnL = cluster_lob.calc_PnL(test_result, "FRNB", k)
            test_FRNB_scaled_PnL = cluster_lob.calc_scaled_PnL(test_FRNB_PnL)
            test_FRNB_PnL_list.append(test_FRNB_scaled_PnL)
            # Compute and collect test PnL series for FREB
            test_FREB_PnL = cluster_lob.calc_PnL(test_result, "FREB", k)
            test_FREB_scaled_PnL = cluster_lob.calc_scaled_PnL(test_FREB_PnL)
            test_FREB_PnL_list.append(test_FREB_scaled_PnL)
        # Aggregate and evaluate test PnL metrics for FRNB
        test_FRNB_PnL_mean = pd.concat(test_FRNB_PnL_list).groupby(level=0).mean()
        test_FRNB_scaled_PnL_mean = cluster_lob.calc_scaled_PnL(test_FRNB_PnL_mean)
        _ = cluster_lob.calc_metric(
            test_FRNB_scaled_PnL_mean, time_intervals, 
            "FRNB", event_type, folder, "test_all"
        )
        _ = cluster_lob.calc_metric(
            test_FRNB_scaled_PnL_mean[train_FRNB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FRNB", event_type, folder, "test_top"
        )
        cluster_lob.calc_cumPnL_plot(
            test_FRNB_scaled_PnL_mean, time_intervals, 
            "FRNB", event_type, folder, "test_all"
        )
        cluster_lob.calc_top_cumPnL_plot(
            test_FRNB_scaled_PnL_mean[train_FRNB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FRNB", event_type, folder, "test_top"
        )
        # Aggregate and evaluate test PnL metrics for FREB
        test_FREB_PnL_mean = pd.concat(test_FREB_PnL_list).groupby(level=0).mean()
        test_FREB_scaled_PnL_mean = cluster_lob.calc_scaled_PnL(test_FREB_PnL_mean)
        _ = cluster_lob.calc_metric(
            test_FREB_scaled_PnL_mean, time_intervals, 
            "FREB", event_type, folder, "test_all"
        )
        _ = cluster_lob.calc_metric(
            test_FREB_scaled_PnL_mean[train_FREB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FREB", event_type, folder, "test_top"
        )
        cluster_lob.calc_cumPnL_plot(
            test_FREB_scaled_PnL_mean, time_intervals, 
            "FREB", event_type, folder, "test_all"
        )
        cluster_lob.calc_top_cumPnL_plot(
            test_FREB_scaled_PnL_mean[train_FREB_top_sr + ofi_no_cluster_index], time_intervals, 
            "FREB", event_type, folder, "test_top"
        )

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        end_time = time.time()
        running_time = end_time - start_time
        with open(folder+"/running_time.txt", "w") as file:
            file.write("Main function running time: {:.2f} seconds\n".format(running_time))
        print("Main function running time: {:.2f} seconds".format(running_time))
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping...")
