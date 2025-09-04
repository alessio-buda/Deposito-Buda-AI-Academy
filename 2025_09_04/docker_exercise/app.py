import requests
import os

# Configure SSL certificate for requests
CERT_PATH = os.environ.get('REQUESTS_CA_BUNDLE', '/usr/local/share/ca-certificates/cacert.crt')

def main():
    
    city = input("Enter city name: ")
    days_range = input("Enter number of days for forecast (1, 3, 7): ")
    include_details = input("Include detailed weather info? (yes/no): ").strip().lower() == 'yes'
    
    try:
        # call wheater API with explicit certificate verification
        coords = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=10&language=en&format=json", 
                            verify=CERT_PATH)
        if coords.status_code != 200 or 'results' not in coords.json() or len(coords.json()['results']) == 0:
            print(f"Could not find coordinates for city: {city}")
            return
        lat, lon = coords.json()['results'][0]['latitude'], coords.json()['results'][0]['longitude']

        # Call weather API with explicit certificate verification
        weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,{'precipitation_sum,wind_speed_10m_max' if include_details else ''}&timezone=auto&forecast_days={days_range}&current_weather=true",
                             verify=CERT_PATH)
        if weather.status_code != 200:
            print("Failed to retrieve weather data")
            return
        print("======== Weather forecast ========")
        forecast = weather.json()
        daily = forecast['daily']
        days = daily['time']
        max_temps = daily['temperature_2m_max']
        min_temps = daily['temperature_2m_min']

        for i in range(len(days)):
            print(f"Date: {days[i]}")
            print(f"  Max Temp: {max_temps[i]}°C")
            print(f"  Min Temp: {min_temps[i]}°C")
            if include_details:
                precip = daily.get('precipitation_sum', [None]*len(days))[i]
                wind = daily.get('wind_speed_10m_max', [None]*len(days))[i]
                print(f"  Precipitation: {precip} mm")
                print(f"  Max Wind Speed: {wind} km/h")
            print("-" * 30)
    
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        print("This might be due to corporate network restrictions.")
    except Exception as e:
        print(f"An error occurred: {e}")

main()