import requests
from bs4 import BeautifulSoup
import time
import os
import logging
from urllib.parse import urljoin, urlparse

class WikiScraper:
    def __init__(self, base_url="https://wiki.sugarlabs.org"):
        self.base_url = base_url
        self.visited = set()
        self.rate_limit = 1
        self.session = requests.Session()
        # Add User-Agent
        self.session.headers.update({
            'User-Agent': 'SugarLabs Documentation Bot (https://github.com/sugarlabs/sugar-docs)'
        })
        self.max_retries = 3
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
           
    def should_scrape_url(self, url):
        """Filter out unwanted URLs"""
        skip_patterns = [
            '/File:', '/Image:', '/Special:', 
            '/Category:', '/Help:', '/Template:'
        ]
        return not any(pattern in url for pattern in skip_patterns)


    def clean_content(self, soup):
        for element in soup.select('.printfooter, .catlinks, #mw-navigation, #footer'):
            element.decompose()
        
        # Get main content
        content = soup.select_one('#mw-content-text')
        if content:
            return content.get_text(separator='\n', strip=True)
        return ""

    def get_internal_links(self, soup):
        links = set()
        for link in soup.select('#mw-content-text a[href^="/"]'):
            href = link.get('href', '')
            if href.startswith('/go/'):
                links.add(urljoin(self.base_url, href))
        return links

    def scrape_page(self, url, retry_count=0):
        if url in self.visited or not self.should_scrape_url(url):
            return None, set()
        
        self.visited.add(url)
        time.sleep(self.rate_limit)
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.select_one('#firstHeading')
            if not title:
                return None, set()
                
            title = title.get_text()
            content = self.clean_content(soup)
            links = self.get_internal_links(soup)
            
            return {
                'title': title,
                'content': content,
                'url': url
            }, links
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            if retry_count < self.max_retries:
                time.sleep(self.rate_limit * 2)
                return self.scrape_page(url, retry_count + 1)
            return None, set()
            
        except Exception as e:
            self.logger.error(f"Unexpected error scraping {url}: {str(e)}")
            return None, set()

    def scrape_wiki(self, start_url):
        queue = [start_url]
        scraped_content = {}
        total_pages = 0
        failed_pages = []
        
        while queue:
            current_url = queue.pop(0)
            self.logger.info(f"Scraping [{len(scraped_content)}/{total_pages}]: {current_url}")
            
            page_data, new_links = self.scrape_page(current_url)
            if page_data:
                scraped_content[current_url] = page_data
                new_urls = [link for link in new_links 
                          if link not in self.visited 
                          and self.should_scrape_url(link)]
                queue.extend(new_urls)
                total_pages = len(scraped_content) + len(queue)
            else:
                failed_pages.append(current_url)
                
        self.logger.info(f"Scraping complete. Success: {len(scraped_content)}, Failed: {len(failed_pages)}")
        return scraped_content

    def save_content(self, content, output_dir="parsed_data/wiki"):
        """Save scraped content to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for url, data in content.items():
            # Create filename from title
            filename = data['title'].lower().replace(' ', '_')
            filename = ''.join(c for c in filename if c.isalnum() or c in '_-')
            filepath = os.path.join(output_dir, f"{filename}.txt")
            
            # Save content with metadata
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {data['title']}\n")
                f.write(f"URL: {url}\n")
                f.write("-" * 80 + "\n")
                f.write(data['content'])
            
            self.logger.info(f"Saved: {filepath}")

def main():
    scraper = WikiScraper()
    start_url = "https://wiki.sugarlabs.org/go/Welcome_to_the_Sugar_Labs_wiki"
    
    try:
        content = scraper.scrape_wiki(start_url)
        scraper.save_content(content)
        print(f"Successfully scraped {len(content)} pages")
    except Exception as e:
        print(f"Error during scraping: {str(e)}")

if __name__ == "__main__":
    main()