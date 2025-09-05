import nltk
import sys
import string

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
S -> NP VP Conj S
S -> NP VP Conj VP

NP -> N
NP -> Det N
NP -> Det Adj N
NP -> Det Adj Adj N
NP -> Det Adj Adj Adj N
NP -> Adj N
NP -> NP PP

VP -> V
VP -> V NP
VP -> V PP
VP -> V NP PP
VP -> V Adv
VP -> Adv V
VP -> V NP Adv
VP -> VP Conj VP

PP -> P NP

AdvP -> Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Tokenize the sentence using NLTK
    words = nltk.word_tokenize(sentence)
    
    # Filter words to include only those with at least one alphabetic character
    # and convert to lowercase
    result = []
    for word in words:
        # Check if word contains at least one alphabetic character
        if any(char.isalpha() for char in word):
            result.append(word.lower())
    
    return result


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    
    def find_np_chunks(subtree):
        # If this subtree is labeled NP
        if subtree.label() == "NP":
            # Check if it contains any NP subtrees
            contains_np = False
            for child in subtree.subtrees():
                if child != subtree and child.label() == "NP":
                    contains_np = True
                    break
            
            # If it doesn't contain other NPs, it's a chunk
            if not contains_np:
                chunks.append(subtree)
            else:
                # If it contains other NPs, recursively search its children
                for child in subtree:
                    if hasattr(child, 'label'):
                        find_np_chunks(child)
        else:
            # If not an NP, search its children
            for child in subtree:
                if hasattr(child, 'label'):
                    find_np_chunks(child)
    
    find_np_chunks(tree)
    return chunks


if __name__ == "__main__":
    main()