import requests

url = 'https://en.wikipedia.org/w/rest.php/v1/search/page'
headers = {
    'User-Agent': 'MediaWiki REST API docs examples/0.1 (https://www.mediawiki.org/wiki/API_talk:REST_API)'
}
params = {
    'q': 'fastapi',
    'limit': '20'
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

print(data)