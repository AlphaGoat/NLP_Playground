import googlesearch


def perform_google_search(query):
    results = googlesearch.search(query, tld="co.in", num=10, stop=1, pause=2)
