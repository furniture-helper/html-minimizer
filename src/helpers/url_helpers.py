from urllib.parse import urlparse


def get_domain_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    return parsed_url.netloc