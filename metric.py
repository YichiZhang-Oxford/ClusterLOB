import pandas as pd

for event_type in ["A","L","D","M"]:
    for FR in ["FRNB","FREB"]:
        df_small = pd.read_csv(f"small_tick_stocks/test_top_{FR}_{event_type}.csv",index_col=0)
        df_medium = pd.read_csv(f"medium_tick_stocks/test_top_{FR}_{event_type}.csv",index_col=0)
        df_large = pd.read_csv(f"large_tick_stocks/test_top_{FR}_{event_type}.csv",index_col=0)

        merged_df = pd.concat([df_small, df_medium, df_large], axis=1)

        # Optionally, save the merged DataFrame
        merged_df.to_csv(f"metric/test_top_{FR}_{event_type}.csv", index=True)