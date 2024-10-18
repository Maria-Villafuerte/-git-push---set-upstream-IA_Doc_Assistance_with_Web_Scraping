import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def sanitize_filename(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()


def download_page(url, output_dir):
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Create a file path for the HTML file
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.lstrip('/').split('/')
        sanitized_parts = [sanitize_filename(part) for part in path_parts]
        file_path = os.path.join(output_dir, *sanitized_parts)

        # If the path doesn't end with .html, append index.html
        if not file_path.endswith('.html'):
            file_path = os.path.join(file_path, 'index.html')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the HTML content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Downloaded: {url}")

        # Find all links in the page
        links = soup.find_all('a', href=True)
        return [(urljoin(url, link['href'])) for link in links]
    else:
        print(f"Failed to download: {url}")
        return []


def crawl(start_url, output_dir, domain):
    visited = set()
    to_visit = [start_url]

    while to_visit:
        current_url = to_visit.pop(0)
        if current_url not in visited and domain in current_url:
            visited.add(current_url)
            new_links = download_page(current_url, output_dir)
            to_visit.extend([link for link in new_links if link not in visited])


if __name__ == "__main__":
    start_url = "https://developer.android.com/reference/kotlin/androidx/classes"
    output_dir = "android_docs"
    domain = "developer.android.com"

    crawl(start_url, output_dir, domain)
    print("Crawling completed.")