import json
import requests

def debug():
    try:
        r = requests.get("https://api.jolpi.ca/ergast/f1/2026/drivers/max_verstappen/results.json")
        data = r.json()
        with open('debug_jolpica.json', 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(e)
debug()
