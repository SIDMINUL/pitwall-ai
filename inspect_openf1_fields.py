
import requests

url = "https://api.openf1.org/v1/laps"
response = requests.get(
    url,
    params={"session_key": 9472, "driver_number": 1},
    timeout=30,
)
response.raise_for_status()

rows = response.json()

if not rows:
    print("No lap records returned.")
else:
    print("Available fields:")
    for field in sorted(rows[0].keys()):
        print("-", field)

    print("\nFirst three records:")
    for row in rows[:3]:
        print(row)
