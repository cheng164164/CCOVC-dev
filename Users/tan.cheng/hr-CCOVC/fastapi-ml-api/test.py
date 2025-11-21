import requests
import json

# Base URL
BASE_URL = "http://127.0.0.1:8000"

# ---------- 1. Single row test ----------
def test_single():
    payload = {
        "Legal_Entity_Label": "Modular Mining Systems Pty Ltd",
        "Business_Unit_Label": "Mining Technology Solutions",
        "Division_Label": "Global Operations",
        "Employment_Type_Label": "Salaried",
        "Job_Classification_Label": "Advisor, One Komatsu Integration"
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("\n--- Single Row Test ---")
    print("Status:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=4))


# ---------- 2. Batch test (list of rows) ----------
def test_batch():
    payload = [
        {
            "Legal_Entity_Label": "Modular Mining Systems Pty Ltd",
            "Business_Unit_Label": "Mining Technology Solutions",
            "Division_Label": "Global Operations",
            "Employment_Type_Label": "Salaried",
            "Job_Classification_Label": "Advisor, One Komatsu Integration"
        },
        {
            "Legal_Entity_Label": "Tramac Canada",
            "Business_Unit_Label": "Tramac",
            "Division_Label": "General & Administrative",
            "Employment_Type_Label": "Hourly",
            "Job_Classification_Label": "Tramac Jobs"
        },
        {
            "Legal_Entity_Label": "Joy Global Underground",
            "Business_Unit_Label": "Manufacturing",
            "Division_Label": "Manufacturing",
            "Employment_Type_Label": "Hourly",
            "Job_Classification_Label": "Shaft Cell Operator"
        },
    ]

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("\n--- Batch Test ---")
    print("Status:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=4))


# ---------- 3. CSV file upload test ----------
def test_csv():
    files = {"file": open("test_batch.csv", "rb")}
    response = requests.post(f"{BASE_URL}/predict_file", files=files)

    print("\n--- CSV Upload Test ---")
    print("Status:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=4))


if __name__ == "__main__":
    test_single()
    test_batch()
    # test_csv()  # Uncomment after you have test_batch.csv
