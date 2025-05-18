import requests

url = 'http://127.0.0.1:5000/predict'
file_path = r'C:\Users\Sania\Documents\GitHub\voice-mood-detector\data\archive\Actor_01\03-01-01-01-01-01-01.wav'

with open(file_path, 'rb') as f:
    files = {'audio': f}
    response = requests.post(url, files=files)

print("Response:", response.status_code)
print("Result:", response.json())