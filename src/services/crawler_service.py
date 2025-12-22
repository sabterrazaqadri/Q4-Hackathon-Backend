import requests
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from src.config.settings import settings
from src.utils.url_validator import is_valid_url, normalize_url, extract_domain
from src.utils.content_processing import extract_text_from_html, get_content_metadata
from src.models.content_chunk import CrawlJob, CrawlJobCreate
from src.services.base_service import BaseService
from src.utils.exceptions import CrawlError


class CrawlerService(BaseService):
    """
    Service for crawling Docusaurus book pages and extracting content
    """
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Backend-Crawler/1.0'
        })
        self.visited_urls: Set[str] = set()
        self.crawl_jobs: Dict[str, CrawlJob] = {}
        
    def _is_valid_docusaurus_url(self, url: str) -> bool:
        """
        Validate if URL is a valid target for crawling
        """
        return is_valid_url(url)
    
    def _get_links_from_page(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract all valid links from a page, with special handling for Docusaurus sites
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []

        # Find all links in the page
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Skip anchor links and external links (unless they're to the same base)
            if href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                continue

            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)

            # Normalize the URL
            normalized_url = normalize_url(absolute_url)

            # Additional check for Docusaurus patterns in the URL
            parsed = urlparse(normalized_url)
            path = parsed.path

            # Skip non-HTML resources (images, PDFs, etc.) unless they're part of navigation
            if any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx', '.zip', '.rar']):
                continue

            # Only add URLs from the same domain
            if extract_domain(normalized_url) == extract_domain(base_url):
                # Additional check to see if this looks like a documentation page
                if self._is_documentation_page(path):
                    links.append(normalized_url)

        return links

    def _is_documentation_page(self, path: str) -> bool:
        """
        Determine if a path likely leads to documentation content
        """
        # Common patterns for documentation pages
        doc_patterns = [
            '.html',  # HTML pages
            '/docs/',  # Common documentation path
            '/guide/',  # Guide pages
            '/tutorial/',  # Tutorial pages
            '/api/',  # API documentation
            '/reference/',  # Reference docs
            '/changelog',  # Changelog
            '/readme',  # Readme files
            '/faq',  # FAQ
            '/examples',  # Examples
        ]

        path_lower = path.lower()
        return any(pattern in path_lower for pattern in doc_patterns) or path.endswith('/')
    
    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a single URL with retry logic
        """
        async def fetch_attempt():
            try:
                response = self.session.get(
                    url,
                    timeout=settings.timeout,
                    allow_redirects=True
                )

                # Check if request was successful
                if response.status_code == 200:
                    return response.text
                else:
                    self.logger.warning(f"Failed to fetch {url}. Status code: {response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
                raise e  # Re-raise to trigger retry logic

        try:
            # Use the base service's retry mechanism
            return await self.retry_with_backoff(fetch_attempt, max_retries=settings.max_retries)
        except Exception as e:
            self.logger.error(f"All retry attempts failed for {url}: {str(e)}")
            return None
    
    async def crawl_single_page(self, url: str) -> Optional[Dict[str, str]]:
        """
        Crawl a single page and extract content and metadata
        """
        if not self._is_valid_docusaurus_url(url):
            raise CrawlError(f"Invalid URL: {url}")

        normalized_url = normalize_url(url)
        content_html = await self._fetch_page_content(normalized_url)

        if not content_html:
            return None

        # Extract text and metadata
        content_text = extract_text_from_html(content_html)
        metadata = get_content_metadata(content_html, normalized_url)

        return {
            "url": normalized_url,
            "content": content_text,
            "metadata": metadata
        }
    
    async def crawl_site(self, base_url: str, max_pages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Crawl an entire site starting from the base URL
        """
        if not self._is_valid_docusaurus_url(base_url):
            raise CrawlError(f"Invalid base URL: {base_url}")

        base_url = normalize_url(base_url)
        crawled_pages = []
        urls_to_visit = [base_url]
        crawled_count = 0

        while urls_to_visit and (max_pages is None or crawled_count < max_pages):
            current_url = urls_to_visit.pop(0)

            # Skip if already visited
            if current_url in self.visited_urls:
                continue

            self.logger.info(f"Crawling: {current_url}")

            # Fetch and process the page
            page_data = await self.crawl_single_page(current_url)
            if page_data:
                crawled_pages.append(page_data)
                self.visited_urls.add(current_url)
                crawled_count += 1

                # Add a small delay to be respectful to the server
                time.sleep(settings.crawl_delay)

                # Extract links from the page to continue crawling
                content_html = await self._fetch_page_content(current_url)
                if content_html:
                    new_links = self._get_links_from_page(content_html, base_url)

                    # Add new links to visit if they haven't been visited
                    for link in new_links:
                        if link not in self.visited_urls and link not in urls_to_visit:
                            urls_to_visit.append(link)
            else:
                self.logger.warning(f"Failed to crawl: {current_url}")

        return crawled_pages
    
    def initiate_crawl_job(self, crawl_job_create: CrawlJobCreate) -> CrawlJob:
        """
        Initiate a new crawl job
        """
        # Create a new crawl job with a unique ID
        from uuid import uuid4
        from datetime import datetime
        
        crawl_job = CrawlJob(
            id=uuid4(),
            source_urls=crawl_job_create.source_urls,
            status="pending",
            start_time=None,
            end_time=None,
            processed_count=0,
            failed_count=0,
            error_details=None
        )
        
        self.crawl_jobs[str(crawl_job.id)] = crawl_job
        return crawl_job
    
    def get_crawl_job(self, job_id: str) -> Optional[CrawlJob]:
        """
        Get the status of a crawl job
        """
        return self.crawl_jobs.get(job_id)
    
    async def execute_crawl_job(self, job_id: str) -> CrawlJob:
        """
        Execute a crawl job
        """
        crawl_job = self.crawl_jobs.get(job_id)
        if not crawl_job:
            raise CrawlError(f"Crawl job {job_id} not found")

        # Update job status
        crawl_job.status = "in_progress"
        crawl_job.start_time = crawl_job.start_time or datetime.now()

        # Reset visited URLs for this job
        self.visited_urls.clear()

        # Process each source URL
        total_processed = 0
        total_failed = 0

        for source_url in crawl_job.source_urls:
            try:
                # Crawl the site
                crawled_pages = await self.crawl_site(source_url)
                total_processed += len(crawled_pages)

                # Update job counts
                crawl_job.processed_count = total_processed
            except Exception as e:
                self.logger.error(f"Error crawling {source_url}: {str(e)}")
                total_failed += 1
                crawl_job.failed_count = total_failed
                crawl_job.error_details = str(e)

        # Update final job status
        crawl_job.status = "completed" if total_failed == 0 else "failed"
        crawl_job.end_time = datetime.now()

        return crawl_job