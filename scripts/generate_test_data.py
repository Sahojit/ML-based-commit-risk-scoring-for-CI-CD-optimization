"""Generate test prediction data"""
import requests
import random
import time

API_URL = "http://localhost:8000/predict"

# Generate 20 test predictions
for i in range(20):
    data = {
        "commit_hash": f"test_{i}",
        "lines_added": random.randint(10, 500),
        "lines_deleted": random.randint(5, 200),
        "files_changed": random.randint(1, 15),
        "touches_core": random.choice([0, 1]),
        "total_commits": random.randint(50, 500),
        "buggy_commits": random.randint(5, 50),
        "recent_frequency": random.randint(1, 20)
    }
    
    response = requests.post(API_URL, json=data)
    print(f"Prediction {i+1}: {response.json()['risk_level']}")
    time.sleep(0.5)

print("\n✅ Generated 20 test predictions!")