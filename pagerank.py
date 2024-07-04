import copy
import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dictionary = {}
    if len(corpus[page]):
        for linkedPage in corpus:
            dictionary[linkedPage] = (1 - damping_factor) / len(corpus)
            if linkedPage in corpus[page]:
                dictionary[linkedPage] += (damping_factor / len(corpus[page]))
    else:
        if len(corpus):
            for linkedPage in corpus:
                dictionary[linkedPage] = 1 / len(corpus)
    return dictionary


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dictionary = {}
    page = random.choice(list(corpus.keys()))
    dictionary[page] = 1
    for i in range(0, n):
        linkedDictionary = transition_model(corpus, page, damping_factor)
        page = random.choices(list(linkedDictionary.keys()), list(
            linkedDictionary.values()), k=1)[0]
        if page not in dictionary:
            dictionary[page] = 1
        else:
            dictionary[page] += 1
    for linkedPage in dictionary:
        dictionary[linkedPage] = dictionary[linkedPage] / n
    return dictionary


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dictionary = {}
    clonedDictionary = {}
    if len(corpus):
        for linkedPage in corpus:
            dictionary[linkedPage] = 1 / len(corpus)
        while True:
            clonedDictionary = copy.deepcopy(dictionary)
            for linkedPage in corpus:
                dictionary[linkedPage] = (1 - damping_factor) / len(corpus)
                keys = list()
                for page in corpus:
                    if linkedPage in corpus[page]:
                        keys.append(page)
                for i in keys:
                    if len(corpus[i]) and linkedPage in corpus[i]:
                        dictionary[linkedPage] += (clonedDictionary[i] /
                                                   len(corpus[i])) * damping_factor
                    else:
                        dictionary[linkedPage] += (clonedDictionary[i] /
                                                   len(corpus)) * damping_factor
            changed = False
            for key in clonedDictionary:
                if dictionary[key] - clonedDictionary[key] > 0.001 or dictionary[key] - clonedDictionary[key] < -0.001:
                    changed = True
            if not changed:
                return dictionary


if __name__ == "__main__":
    main()
