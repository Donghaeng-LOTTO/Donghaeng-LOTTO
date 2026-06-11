import pandas as pd

quality = pd.read_csv("data/processed/dataset_quality_report.csv")
print(quality.head())
print(quality[[
    "game_id",
    "has_record_raw",
    "has_relay_raw",
    "n_relay_events",
    "n_plate_appearances",
    "pa_count_plausible",
    "state_warn_ratio",
]].head(30))