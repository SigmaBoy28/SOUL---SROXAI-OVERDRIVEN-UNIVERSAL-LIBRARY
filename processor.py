import re
import math
from collections import Counter


# Keep this list reasonably focused.
# You can expand it later.
STOP_WORDS = {
    # Articles
    "a", "an", "the",

    # Conjunctions
    "and", "or", "but", "if", "then",
    "because", "although", "while",
    "whereas", "also", "however",
    "therefore", "instead",

    # Prepositions
    "of", "to", "in", "on", "at", "by",
    "for", "from", "with", "about",
    "into", "through", "during",
    "before", "after", "above",
    "below", "between", "under",
    "over", "without", "within",

    # Pronouns
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",

    # Demonstratives
    "this", "that", "these", "those",
    "here", "there",

    # Verbs
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",

    # Modal verbs
    "can", "could",
    "will", "would",
    "shall", "should",
    "may", "might",
    "must",

    # Negation
    "not", "no", "yes",

    # Quantifiers
    "very", "more", "most",
    "some", "any", "all",
    "each", "every",
    "both", "either", "neither",
    "many", "much", "few",
    "several",

    # Question words
    "who", "whom", "whose",
    "which", "what", "where",
    "when", "why", "how",

    # Numbers
    "one", "two", "three",
    "first", "second", "third",

    # Common adverbs
    "just", "only",
    "even", "still",
    "already", "again",
    "often", "always",
    "never", "sometimes",

    # Generic nouns
    "people", "person",
    "man", "woman",
    "child", "children",
    "thing", "things",
    "way", "ways",
    "time", "times",
    "day", "days",
    "year", "years",
    "world", "life",
    "part", "parts",
    "place", "places",

    # Generic verbs
    "make", "made", "making",
    "get", "got", "getting",
    "go", "went", "going",
    "come", "came", "coming",
    "take", "took", "taking",
    "give", "gave", "giving",
    "use", "used", "using",
    "know", "knew", "knowing",
    "think", "thought", "thinking",
    "see", "saw", "seeing",
    "look", "looked", "looking",
    "find", "found", "finding",
    "want", "wanted",
    "need", "needed",
    "like", "liked",
    "work", "worked",
    "seem", "seemed",

    # Generic adjectives
    "good", "bad",
    "big", "small",
    "large", "little",
    "long", "short",
    "high", "low",
    "new", "old",
    "young",
    "different", "same",
    "important", "possible",
    "easy", "hard",
    "great", "better", "best",
    "real", "true",
    "right", "wrong",

    # Generic academic words
    "example", "question",
    "answer", "problem",
    "result", "reason",
    "case", "kind", "type",
    "number", "name",
    "information", "idea",
    "fact", "point",
    "form",

    # Miscellaneous
    "remain", "remains",
    "thus", "etc",
}


def tokenize(text: str) -> list[str]:
    """
    Convert raw text into normalized word tokens.
    """

    text = text.lower()

    # Keep alphabetic words and simple apostrophes.
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    return words


def clean_words(words: list[str]) -> list[str]:
    """
    Remove stop words and very short words.
    """

    cleaned = []

    for word in words:

        if len(word) < 2:
            continue

        if word in STOP_WORDS:
            continue

        cleaned.append(word)

    return cleaned


def process_text(text: str) -> list[str]:
    """
    Raw text -> cleaned word list.
    """

    words = tokenize(text)

    return clean_words(words)


def word_frequency(words: list[str]) -> Counter:
    """
    Count how frequently each word occurs.
    """

    return Counter(words)


def build_keyword_index(documents: dict[str, list[str]]) -> dict[str, set[str]]:
    """
    Build:

        keyword -> set of documents

    Example:

        {
            "neural": {
                "AI.pdf",
                "System.pdf"
            }
        }
    """

    index = {}

    for filename, words in documents.items():

        unique_words = set(words)

        for word in unique_words:

            if word not in index:
                index[word] = set()

            index[word].add(filename)

    return index


def calculate_idf(
    documents: dict[str, list[str]]
) -> dict[str, float]:

    """
    Calculate inverse document frequency.

    IDF makes common words less important
    and rare words more important.
    """

    total_documents = len(documents)

    document_frequency = Counter()

    for words in documents.values():

        for word in set(words):
            document_frequency[word] += 1

    idf = {}

    for word, frequency in document_frequency.items():

        idf[word] = math.log(
            (1 + total_documents) /
            (1 + frequency)
        ) + 1

    return idf


def build_tfidf_vectors(
    documents: dict[str, list[str]],
    idf: dict[str, float]
) -> dict[str, dict[str, float]]:

    """
    Create TF-IDF vectors for every document.
    """

    vectors = {}

    for filename, words in documents.items():

        counts = Counter(words)

        total_words = len(words)

        vector = {}

        if total_words == 0:
            vectors[filename] = vector
            continue

        for word, count in counts.items():

            tf = count / total_words

            vector[word] = tf * idf[word]

        vectors[filename] = vector

    return vectors


def cosine_similarity(
    vector_a: dict[str, float],
    vector_b: dict[str, float]
) -> float:

    """
    Calculate cosine similarity between two sparse vectors.
    """

    if not vector_a or not vector_b:
        return 0.0

    # Iterate through the smaller vector.
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a

    dot_product = 0.0

    for word, value in vector_a.items():

        if word in vector_b:
            dot_product += value * vector_b[word]

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a.values())
    )

    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b.values())
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)