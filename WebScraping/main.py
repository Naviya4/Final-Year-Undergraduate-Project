from bs4 import BeautifulSoup
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def create_session(retries=3, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def filter_english_chars(text):
    """Keep only English characters and standard punctuation."""
    return ''.join(char for char in text if ord(char) < 128)


def scrape_house_links(page_url):
    """Scrape house links from a single page."""
    try:
        response = session.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        # List all the different classes that contain the ad listings
        ad_classes = ['top-ads-container--1Jeoq gtm-top-ad', 'normal--2QYVk gtm-normal-ad first-add--1u5Mw', 'normal--2QYVk gtm-normal-ad']
        product_links = []

        for ad_class in ad_classes:
            product_list = soup.find_all('li', class_=ad_class)
            for item in product_list:
                link = item.find('a', href=True)
                if link and link['href'] != '/en/promotions':
                    full_link = f"https://ikman.lk{link['href']}"
                    product_links.append(full_link)

        return product_links
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []


def scrape_house_details(link):
    """Scrape details of a single house."""
    print(f"Scraping: {link}")  # Print the link being scraped
    try:
        response = session.get(link)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')

        # This is a general map that you may need to adjust based on the actual features.
        feature_map = {
            'Bedrooms:': None,
            'Bathrooms:': None,
            'House size:': None,
            'Land size:': None,
            # Add more features here as needed.
        }

        features = soup.find_all('div', class_='word-break--2nyVq value--1lKHt')
        for feature in features:
            # Assuming the title of the feature is the previous sibling
            feature_title = feature.previous_sibling.get_text(strip=True)
            if feature_title in feature_map:
                feature_map[feature_title] = feature.get_text(strip=True)

        location = soup.find_all('a', class_='subtitle-location-link--1q5zA')

        description_container = soup.find('div', class_='description--1nRbz')
        # Check if the container was found before trying to access its attributes
        if description_container:
            description_lines = description_container.stripped_strings
            description = ' '.join(line + ' ' for line in description_lines)
        else:
            description = ''  # Set a default empty description if the container is not found

        price = soup.find('div', class_='amount--3NTpl').text

        return {
            'sub_location': location[0].text if len(location) > 0 else None,
            'parent_location': location[1].text if len(location) > 1 else None,
            'bed_rooms': feature_map.get('Bedrooms:'),
            'bath_rooms': feature_map.get('Bathrooms:'),
            'house_size': feature_map.get('House size:'),
            'land_size': feature_map.get('Land size:'),
            'description': filter_english_chars(description.strip()),
            'price': price
        }
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    return None


session = create_session()
headers = {'User-Agent': 'YourUserAgentStringHere'}
session.headers.update(headers)

product_links = []
x = 1
max_pages = 944  # Update this to the number of pages you want to scrape
while x <= max_pages:
    page_url = f"https://ikman.lk/en/ads/sri-lanka/houses-for-sale?sort=date&order=desc&buy_now=0&urgent=0&page={x}"
    links = scrape_house_links(page_url)
    if not links:
        break
    product_links.extend(links)
    x += 1

house_list = [scrape_house_details(link) for link in product_links if link is not None]

df = pd.DataFrame([house for house in house_list if house])
df.to_csv("Sri-Lanka_Home_Details.csv", index=False)