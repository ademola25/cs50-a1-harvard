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
    # Initialize probability distribution with equal probability for all pages
    # This represents the (1 - damping_factor) portion where we randomly jump to any page
    total_pages = len(corpus)
    prob_distribution = {}
    
    # Start with base probability for random jumps (1 - damping_factor) / N
    base_prob = (1 - damping_factor) / total_pages
    for p in corpus:
        prob_distribution[p] = base_prob
    
    # Get the links from the current page
    links = corpus[page]
    
    # If the page has no outgoing links, treat it as linking to all pages
    if not links:
        # Equal probability for all pages (including current page)
        equal_prob = 1.0 / total_pages
        return {p: equal_prob for p in corpus}
    
    # Add the damping_factor portion: probability of following a link
    link_prob = damping_factor / len(links)
    for link in links:
        prob_distribution[link] += link_prob
    
    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize page visit counts
    page_counts = {page: 0 for page in corpus}
    
    # Start with a random page
    current_page = random.choice(list(corpus.keys()))
    page_counts[current_page] += 1
    
    # Generate n-1 more samples (we already have 1 from the initial random choice)
    for _ in range(n - 1):
        # Get transition probabilities from current page
        probabilities = transition_model(corpus, current_page, damping_factor)
        
        # Choose next page based on the probability distribution
        pages = list(probabilities.keys())
        weights = [probabilities[page] for page in pages]
        current_page = random.choices(pages, weights=weights)[0]
        
        # Increment count for the chosen page
        page_counts[current_page] += 1
    
    # Convert counts to probabilities (normalize by total samples)
    pagerank = {page: count / n for page, count in page_counts.items()}
    
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)
    
    # Initialize each page with equal PageRank (1/N)
    pagerank = {page: 1 / total_pages for page in corpus}
    
    # Iterate until convergence
    while True:
        new_pagerank = {}
        
        for page in corpus:
            # Start with the random jump probability: (1-d)/N
            rank = (1 - damping_factor) / total_pages
            
            # Add contributions from all pages in the corpus
            for linking_page in corpus:
                links_from_linking_page = corpus[linking_page]
                
                # If linking page has no outgoing links, treat as linking to all pages
                if not links_from_linking_page:
                    # Page with no links contributes to ALL pages equally
                    rank += damping_factor * pagerank[linking_page] / total_pages
                else:
                    # Only add contribution if this linking_page actually links to current page
                    if page in links_from_linking_page:
                        rank += damping_factor * pagerank[linking_page] / len(links_from_linking_page)
            
            new_pagerank[page] = rank
        
        # Check for convergence (no value changes by more than 0.001)
        converged = True
        for page in corpus:
            if abs(new_pagerank[page] - pagerank[page]) > 0.001:
                converged = False
                break
        
        pagerank = new_pagerank
        
        if converged:
            break
    
    return pagerank


if __name__ == "__main__":
    main()

