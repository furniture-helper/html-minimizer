from helpers.url_helpers import get_domain_from_url


class MinimizerEvent:

    def __init__(self, url: str, original_size_bytes: int, minimized_size_bytes: int):
        self.url = url
        self.domain = get_domain_from_url(url)
        self.original_size_bytes = original_size_bytes
        self.minimized_size_bytes = minimized_size_bytes

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "domain": self.domain,
            "original_size_bytes": self.original_size_bytes,
            "minimized_size_bytes": self.minimized_size_bytes,
        }

