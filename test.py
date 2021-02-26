import requests
request = requests.get('http://api.open-notify.org/iss-now.json')
data = request.json()
print(data['iss_position'])