from bs4 import BeautifulSoup
import requests

def fetch_books_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    response = requests.get(url,headers=headers)
    data = BeautifulSoup(response.content, 'html.parser')
    # img_tags = data.find_all("img", class_="lazyload img-responsive center-block")
    image_thumb = data.find_all("a",class_="image_thumb")
    return image_thumb
def fetch_books_description(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    response = requests.get(url,headers=headers)
    data = BeautifulSoup(response.content, 'html.parser')
    description = data.find("meta", {"itemprop": "description"}).get("content")
    return description