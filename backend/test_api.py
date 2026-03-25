import requests

url = "http://[IP_ADDRESS]/predict"

data = {
    "smiles1": "CCO",
    "smiles2": "CCN"
}

response = requests.post(url, json=data)
print(response.json())