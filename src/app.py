from pathlib import Path

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

# Model pinned to a specific Hub commit so results don't change when "latest" moves
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
BATCH_SIZE = 32


def stars_to_band(stars: int) -> str:
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def load_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME, revision=MODEL_REVISION)


def predict_sentiment(df: pd.DataFrame, classifier) -> pd.DataFrame:
    outputs = classifier(df["review_text"].tolist(), batch_size=BATCH_SIZE, truncation=True)
    enriched = df.copy()
    enriched["pred_stars"] = [int(o["label"].split()[0]) for o in outputs]  # labels look like "4 stars"
    enriched["pred_confidence"] = [round(o["score"], 4) for o in outputs]
    enriched["sentiment_band"] = enriched["pred_stars"].map(stars_to_band)
    return enriched


def main():
    reviews = pd.read_csv(RAW_PATH)
    classifier = load_model()  # loaded once, reused for every review
    enriched = predict_sentiment(reviews, classifier)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(enriched)} reviews to {OUTPUT_PATH}")
    print(enriched["sentiment_band"].value_counts(normalize=True).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
