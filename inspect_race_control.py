
import requests
import pandas as pd

BASE_URL = "https://api.openf1.org/v1"

# Bahrain 2024 race session, already used in your analysis.
SESSION_KEY = 9472


def fetch_race_control(session_key: int) -> pd.DataFrame:
    url = f"{BASE_URL}/race_control"
    response = requests.get(
        url,
        params={"session_key": session_key},
        timeout=30,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def main():
    df = fetch_race_control(SESSION_KEY)

    if df.empty:
        print("No race-control messages returned.")
        return

    print(f"Race-control messages: {len(df)}")
    print("\nAvailable columns:")
    print(df.columns.tolist())

    useful_columns = [
        column
        for column in [
            "date",
            "category",
            "flag",
            "scope",
            "sector",
            "message",
            "driver_number",
            "lap_number",
        ]
        if column in df.columns
    ]

    print("\nPotentially relevant messages:")
    pattern = r"safety car|virtual safety car|\bVSC\b|yellow|red flag|green flag|clear"

    mask = (
        df[useful_columns]
        .astype(str)
        .apply(
            lambda column: column.str.contains(
                pattern, case=False, regex=True, na=False
            )
        )
        .any(axis=1)
    )

    print(
        df.loc[mask, useful_columns]
        .sort_values("date")
        .to_string(index=False)
    )

    output_path = "data/openf1/bahrain_2024_race_control.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved full messages to {output_path}")


if __name__ == "__main__":
    main()
