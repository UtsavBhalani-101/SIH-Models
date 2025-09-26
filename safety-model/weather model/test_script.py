# test_regression_api.py - Test Dynamic vs Static Scores

import requests
import json

BASE_URL = "https://of8766175--weather-safety-api-fastapi-app-dev.modal.run"

def run_test(name, method, url, data=None):
    """Helper function to run a test and print results."""
    print(f"\n{'='*50}")
    print(f"🧪 {name}")
    print('='*50)
    try:
        if method.upper() == 'GET':
            response = requests.get(url)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data)
        
        print(f"Status: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        
        # Pretty print with focus on key differences
        if 'safety_score' in result:
            print(f"🎯 Safety Score: {result['safety_score']}")
            print(f"📊 Category: {result['safety_category']}")
            print(f"🔍 Model Type: {result.get('model_type', 'unknown')}")
            print(f"🎲 Confidence: {result['confidence']}")
        else:
            print(json.dumps(result, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")

def compare_models():
    """Compare different weather scenarios to show dynamic vs static scoring."""
    
    print("🌤️  WEATHER SAFETY API - DYNAMIC SCORING TEST")
    print("="*60)
    
    # Test model info
    run_test("Model Information", "GET", f"{BASE_URL}/model/info")
    
    # Test various weather scenarios
    scenarios = [
        {
            "name": "Perfect Summer Day",
            "data": {
                "temperature": 24.0, "apparent_temperature": 26.0, "humidity": 45,
                "wind_speed": 7.0, "visibility": 20.0, "cloud_cover": 0.1,
                "pressure": 1015.0, "summary_clear": 1
            }
        },
        {
            "name": "Slightly Windy Day", 
            "data": {
                "temperature": 22.0, "apparent_temperature": 24.0, "humidity": 55,
                "wind_speed": 25.0, "visibility": 15.0, "cloud_cover": 0.3,
                "pressure": 1012.0, "summary_partly_cloudy": 1
            }
        },
        {
            "name": "Cool and Breezy",
            "data": {
                "temperature": 16.0, "apparent_temperature": 14.0, "humidity": 70,
                "wind_speed": 35.0, "visibility": 12.0, "cloud_cover": 0.6,
                "pressure": 1008.0, "summary_breezy": 1
            }
        },
        {
            "name": "Foggy Morning",
            "data": {
                "temperature": 18.0, "apparent_temperature": 19.0, "humidity": 95,
                "wind_speed": 5.0, "visibility": 1.5, "cloud_cover": 0.9,
                "pressure": 1010.0, "summary_foggy": 1
            }
        },
        {
            "name": "Rainy Day",
            "data": {
                "temperature": 12.0, "apparent_temperature": 10.0, "humidity": 85,
                "wind_speed": 20.0, "visibility": 3.0, "cloud_cover": 1.0,
                "pressure": 995.0, "precip_rain": 1, "summary_rain": 1
            }
        },
        {
            "name": "Winter Storm",
            "data": {
                "temperature": -8.0, "apparent_temperature": -15.0, "humidity": 90,
                "wind_speed": 55.0, "visibility": 0.8, "cloud_cover": 1.0,
                "pressure": 980.0, "precip_snow": 1, "summary_snow": 1
            }
        },
        {
            "name": "Extreme Heat",
            "data": {
                "temperature": 38.0, "apparent_temperature": 45.0, "humidity": 30,
                "wind_speed": 15.0, "visibility": 8.0, "cloud_cover": 0.2,
                "pressure": 1005.0, "summary_clear": 1
            }
        }
    ]
    
    print("\n🔬 TESTING DIFFERENT WEATHER SCENARIOS")
    print("Notice how regression gives precise scores vs classification's fixed buckets")
    
    for scenario in scenarios:
        run_test(f"📈 {scenario['name']}", "POST", f"{BASE_URL}/predict", scenario["data"])
    
    # Test sample predictions endpoint
    run_test("📊 Sample Predictions Comparison", "GET", f"{BASE_URL}/model/sample-predictions")
    
    print("\n" + "="*60)
    print("🎉 DYNAMIC SCORING TEST COMPLETED!")
    print("\n💡 Key Differences:")
    print("   🔸 Regression Model: Gives precise scores like 67.3, 23.8, 89.1")
    print("   🔸 Classification Model: Only gives 0, 25, 50, 75, or 100")
    print("   🔸 Dynamic scoring is more realistic and useful for decision making!")
    print("="*60)

if __name__ == "__main__":
    compare_models()