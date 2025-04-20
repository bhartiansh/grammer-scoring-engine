import pandas as pd
from scripts.model import train_model, predict_scores

def run_pipeline(input_csv, test_list, train_csv, output_csv):
    train_model(input_csv, train_csv)

    all_transcripts = pd.read_csv(input_csv)
    test_df = pd.read_csv(test_list)

    merged = pd.merge(test_df, all_transcripts, on="filename", how="left")
    merged["transcript"] = merged["transcript"].fillna("")

    merged["label"] = predict_scores(merged["transcript"])
    scored_df = merged[["filename", "label"]]

    if scored_df.shape[0] != test_df.shape[0]:
        raise ValueError(f"Expected {test_df.shape[0]} rows, got {scored_df.shape[0]}.")

    scored_df.to_csv(output_csv, index=False)
