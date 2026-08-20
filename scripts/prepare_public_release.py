"""Create the minimum data files required by the public app.

This intentionally excludes each review's raw text, cleaned text, and watch date.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reviews = pd.read_csv(root / "data" / "reviews.csv", dtype={"review_id": str})
    mapping = pd.read_csv(root / "model" / "mapping.csv", dtype={"review_id": str})
    reviews[["review_id", "title", "rating"]].to_csv(
        root / "data" / "public_catalog.csv", index=False, encoding="utf-8-sig"
    )
    mapping[["review_id", "title", "rating"]].to_csv(
        root / "model" / "public_mapping.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Prepared {len(reviews)} public catalog records and {len(mapping)} public mapping records.")


if __name__ == "__main__":
    main()
