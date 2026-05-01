import os
import shelve
import time
from collections import defaultdict, deque
from threading import RLock
from urllib.parse import urlparse

from utils import get_logger, get_urlhash, normalize
from scraper import is_valid


class Frontier:
    def __init__(self, config, restart):
        self.logger = get_logger("FRONTIER")
        self.config = config
        self.lock = RLock()
        # Maps hostname -> deque of pending URLs for that domain.
        self.domain_queues = defaultdict(deque)
        # Maps hostname -> timestamp of last fetch from that domain.
        self.domain_last_access = {}
        # Number of URLs currently being downloaded (fetched but not yet complete).
        # Prevents threads from declaring the frontier empty while scraped links
        # are still being added.
        self.in_progress = 0

        if not os.path.exists(self.config.save_file) and not restart:
            self.logger.info(
                f"Did not find save file {self.config.save_file}, "
                f"starting from seed.")
        elif os.path.exists(self.config.save_file) and restart:
            self.logger.info(
                f"Found save file {self.config.save_file}, deleting it.")
            os.remove(self.config.save_file)

        self.save = shelve.open(self.config.save_file)
        if restart:
            for url in self.config.seed_urls:
                self.add_url(url)
        else:
            self._parse_save_file()
            if not self.save:
                for url in self.config.seed_urls:
                    self.add_url(url)

    def _get_domain(self, url):
        return urlparse(url).hostname or ""

    def _parse_save_file(self):
        total_count = len(self.save)
        tbd_count = 0
        for url, completed in self.save.values():
            if not completed and is_valid(url):
                self.domain_queues[self._get_domain(url)].append(url)
                tbd_count += 1
        self.logger.info(
            f"Found {tbd_count} urls to be downloaded from {total_count} "
            f"total urls discovered.")

    def get_tbd_url(self):
        """Return the next URL to crawl, respecting per-domain politeness delay.

        Blocks until a URL from a ready domain is available, or returns None
        when all queues are empty and no downloads are in progress.
        """
        while True:
            with self.lock:
                now = time.time()
                for domain, queue in self.domain_queues.items():
                    if not queue:
                        continue
                    last = self.domain_last_access.get(domain, 0)
                    if now - last >= self.config.time_delay:
                        url = queue.popleft()
                        self.domain_last_access[domain] = now
                        self.in_progress += 1
                        return url
                # Done when no pending URLs remain and no downloads are active.
                if self.in_progress == 0 and all(
                        len(q) == 0 for q in self.domain_queues.values()):
                    return None
            # No domain is ready yet — yield CPU briefly and retry.
            time.sleep(0.05)

    def add_url(self, url):
        url = normalize(url)
        urlhash = get_urlhash(url)
        with self.lock:
            if urlhash not in self.save:
                self.save[urlhash] = (url, False)
                self.save.sync()
                self.domain_queues[self._get_domain(url)].append(url)

    def mark_url_complete(self, url):
        urlhash = get_urlhash(url)
        with self.lock:
            if urlhash not in self.save:
                self.logger.error(
                    f"Completed url {url}, but have not seen it before.")
            self.save[urlhash] = (url, True)
            self.save.sync()
            self.in_progress -= 1
