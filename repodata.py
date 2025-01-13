import requests
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_repositories():
    """Fetch Sugar Labs repositories from GitHub API"""
    url = "https://api.github.com/orgs/sugarlabs/repos?per_page=100"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    all_repos = []
    
    try:
        while url:
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to fetch repos: {response.status_code}")
                break
                
            repos = response.json()
            all_repos.extend(repos)
            
            # Get next page URL from Link header
            url = response.links.get('next', {}).get('url')
            
            # Check rate limits
            if int(response.headers.get('X-RateLimit-Remaining', 0)) < 1:
                logger.warning("GitHub API rate limit reached")
                break
                
        return all_repos
    except Exception as e:
        logger.error(f"Error fetching repositories: {e}")
        return []

def format_repo_data(repos):
    """Format repository data as text"""
    formatted_data = []
    
    for repo in repos:
        try:
            text = f"""
Repository: {repo['name']}
Description: {repo['description'] or 'No description available'}
URL: {repo['html_url']}
Primary Language: {repo['language'] or 'Not specified'}
Stars: {repo['stargazers_count']}
Forks: {repo['forks_count']}
Topics: {', '.join(repo.get('topics', []))}
Last Updated: {repo.get('updated_at', 'Unknown')}
Default Branch: {repo.get('default_branch', 'master')}
License: {repo.get('license', {}).get('name', 'Not specified')}
---
"""
            formatted_data.append((repo['name'], text))
        except Exception as e:
            logger.error(f"Error formatting repo {repo.get('name', 'unknown')}: {e}")
            
    return formatted_data

def save_repo_data():
    """Fetch repositories and save as text files"""
    output_dir = "parsed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    repos = fetch_repositories()
    formatted_repos = format_repo_data(repos)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    for name, content in formatted_repos:
        filename = f"{output_dir}/repo_{name}_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved repository data: {filename}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

if __name__ == "__main__":
    logger.info("Starting repository data collection...")
    save_repo_data()
    logger.info("Repository data collection complete")