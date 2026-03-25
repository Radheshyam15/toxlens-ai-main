import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "smiles1": "CCO",
    "smiles2": "CCN"
}

response = requests.post(url, json=data)
print(response.json())