import requests
from bs4 import BeautifulSoup

def scrape_data(url):
    ''' '''
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
