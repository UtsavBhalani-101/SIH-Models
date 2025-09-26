# use_api.py (Final Version 1.4, Correct)

import requests
import json

BASE_URL = "https://of8766175--weather-safety-api-fastapi-app-dev.modal.run"

def run_test(name, method, url, data=None):
    """Helper function to run a test and print results."""
    print(f"=== {name} ===")
    try:
        if method.upper() == 'GET':
            response = requests.get(url)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data)
        else:
            raise ValueError("Unsupported HTTP method")
        
        print(f"Status: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        print(json.dumps(result, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
    finally:
        print("-" * 20)

def main():
    print("Weather Safety API Testing")
    print("=" * 50 + "\n")

    run_test("Health Check", "GET", f"{BASE_URL}/health")
    run_test("Model Info", "GET", f"{BASE_URL}/model/info")

    pleasant_weather = {
        "temperature": 22.0, "apparent_temperature": 25.0, "humidity": 60,
        "wind_speed": 8.0, "wind_bearing": 180.0, "visibility": 15.0,
        "cloud_cover": 0.3, "pressure": 1013.0, "summary_clear": 1
    }
    run_test("Basic Prediction (Pleasant)", "POST", f"{BASE_URL}/predict", pleasant_weather)

    extreme_weather = {
        "temperature": -10.0, "apparent_temperature": -15.0, "humidity": 95,
        "wind_speed": 80.0, "wind_bearing": 270.0, "visibility": 0.5,
        "cloud_cover": 1.0, "pressure": 975.0, "precip_snow": 1, "summary_foggy": 1,
        "summary_wind": 1, "summary_snow": 1
    }
    run_test("Extreme Weather Prediction", "POST", f"{BASE_URL}/predict", extreme_weather)

    nyc_location = {"latitude": 40.7128, "longitude": -74.0060, "days": 1}
    run_test("Open-Meteo Integration (NYC)", "POST", f"{BASE_URL}/predict/openmeteo", nyc_location)

    weather_batch = [pleasant_weather, extreme_weather]
    run_test("Batch Prediction", "POST", f"{BASE_URL}/predict/batch", weather_batch)

    invalid_data = {"temperature": 200.0, "humidity": 50, "wind_speed": 10, "visibility": 10, "pressure": 1000}
    run_test("Input Validation Test", "POST", f"{BASE_URL}/predict", invalid_data)

    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()