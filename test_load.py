"""
Test script for verifying predictive maintenance backend and models.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import requests

BASE_URL = "http://127.0.0.1:8000"

def main():
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE BACKEND TEST SUITE")
    print("=" * 60)

    # 1. Health check
    print("\n1. Testing GET /health...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {r.status_code}")
        print("Response:", json.dumps(r.json(), indent=2))
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    except Exception as e:
        print(f"Health check failed (is server running?): {e}")
        sys.exit(1)

    # 2. Schema check
    print("\n2. Testing GET /schema...")
    r = requests.get(f"{BASE_URL}/schema", timeout=5)
    print(f"Status: {r.status_code}")
    schema = r.json()
    print(f"Features count: {len(schema.get('features', []))}")
    print(f"Machine types count: {len(schema.get('machine_types', []))}")
    assert r.status_code == 200

    # 3. Metrics check
    print("\n3. Testing GET /metrics...")
    r = requests.get(f"{BASE_URL}/metrics", timeout=5)
    print(f"Status: {r.status_code}")
    metrics = r.json()
    print("Metrics:", json.dumps(metrics, indent=2))
    assert r.status_code == 200

    # 4. Single prediction with sample from demo_machines.csv
    csv_path = Path(__file__).resolve().parent / "data" / "demo_machines.csv"
    if not csv_path.is_file():
        csv_path = Path("data/demo_machines.csv")

    df = pd.read_csv(csv_path)

    # Test normal machine (non-failing)
    print("\n4. Testing POST /predict (Normal Machine)...")
    sample_normal = df.iloc[1].to_dict()
    sample_normal = {k: (None if pd.isna(v) else v) for k, v in sample_normal.items()}
    r = requests.post(f"{BASE_URL}/predict", json=sample_normal, timeout=5)
    print(f"Status: {r.status_code}")
    print(f"Machine Type: {sample_normal.get('Machine_Type')}")
    print("Prediction:", json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    # Test failing machine
    failing_rows = df[df["Failure_Within_7_Days"] == 1]
    if len(failing_rows) > 0:
        print("\n5. Testing POST /predict (High-Risk Machine)...")
        sample_fail = failing_rows.iloc[0].to_dict()
        sample_fail = {k: (None if pd.isna(v) else v) for k, v in sample_fail.items()}
        r = requests.post(f"{BASE_URL}/predict", json=sample_fail, timeout=5)
        print(f"Status: {r.status_code}")
        print(f"Machine Type: {sample_fail.get('Machine_Type')}")
        print("Prediction:", json.dumps(r.json(), indent=2))
        assert r.status_code == 200

    # 6. Test server-side Machine_Age calculation
    print("\n6. Testing POST /predict with omitted Machine_Age...")
    sample_calc_age = dict(sample_normal)
    sample_calc_age.pop("Machine_Age", None)
    sample_calc_age["Installation_Year"] = 2021.0
    r = requests.post(f"{BASE_URL}/predict", json=sample_calc_age, timeout=5)
    print(f"Status: {r.status_code}")
    print("Prediction with server-side age:", json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    # 7. Batch prediction
    print("\n7. Testing POST /predict-batch (5 machines)...")
    batch = []
    for i in range(5):
        row = df.iloc[i].to_dict()
        row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        batch.append(row)
    r = requests.post(f"{BASE_URL}/predict-batch", json=batch, timeout=10)
    print(f"Status: {r.status_code}")
    print(f"Batch returned {len(r.json())} predictions:")
    for idx, item in enumerate(r.json()):
        print(f"  Machine {idx}: prob={item['failure_probability']}%, "
              f"prediction={item['failure_prediction']}, "
              f"risk={item['risk_category']}, "
              f"RUL={item['remaining_useful_life_days']} days")
    assert r.status_code == 200
    assert len(r.json()) == 5

    # 8. 422 Validation Error Test
    print("\n8. Testing 422 validation error handling...")
    r = requests.post(f"{BASE_URL}/predict", json={"Temperature_C": "invalid_value"}, timeout=5)
    print(f"Status: {r.status_code} (Expected 422)")
    print("Response:", json.dumps(r.json(), indent=2))
    assert r.status_code == 422

    print("\n" + "=" * 60)
    print("SUCCESS: ALL ENDPOINTS TESTED AND VALIDATED PROPERLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
