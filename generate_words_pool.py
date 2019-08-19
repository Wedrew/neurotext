from string import ascii_letters, whitespace
from urllib.request import urlopen
from urllib.error import URLError
from httplib2 import Http
from concurrent import futures
from multiprocessing import Pool, cpu_count

class GenerateWords():
    def __init__(self, total_books = 10, allowed_characters = set(ascii_letters + ' ' + '\n' + '\t')):
        self.total_books = total_books
        self.allowed_characters = allowed_characters
        self.url_prefix = "https://www.gutenberg.org/cache/epub/"
        self.unformatted_path = "{}/pg{}.txt"
        self.workers = cpu_count()
        self.sanitized_words = []
        self.working_urls = []

    def set_total_books(self, total_books):
        self.total_books = total_books

    def generate(self):
        all_urls = [self.url_prefix + self.unformatted_path.format(i, i) for i in range(1, self.total_books + 1)]
        self.all_urls = self.test_urls(all_urls)

        with Pool(processes=self.workers) as pool:
            for url, words in zip(self.all_urls, pool.map(self.sanitize, self.all_urls)):
                self.sanitized_words.extend(words)

    def sanitize(self, url):
        book_string = urlopen(url).read().decode('utf-8', 'ignore')
        return ''.join([letter for letter in book_string if letter in self.allowed_characters]).split()

    def test_urls(self, urls):
        for url in urls:
            try:
                urlopen(url)
            except URLError:
                urls.remove(url)

        return urls

if __name__ == '__main__':
    words = GenerateWords()
    words.set_total_books(100)
    words.generate()
