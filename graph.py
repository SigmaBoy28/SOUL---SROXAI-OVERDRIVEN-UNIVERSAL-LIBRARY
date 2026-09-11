from itertools import combinations

from processor import cosine_similarity


def build_relationships(
    documents,
    tfidf_vectors,
    threshold
):

    relationships = []

    filenames = list(documents.keys())

    for file_a, file_b in combinations(filenames, 2):

        vector_a = tfidf_vectors[file_a]
        vector_b = tfidf_vectors[file_b]

        similarity = cosine_similarity(
            vector_a,
            vector_b
        )

        if similarity >= threshold:

            relationships.append({
                "document_a": file_a,
                "document_b": file_b,
                "similarity": similarity
            })

    return relationships


def build_graph(
    documents,
    relationships
):

    """
    Convert the relationship list into
    a graph-friendly structure.
    """

    nodes = []

    for filename in documents:

        nodes.append({
            "id": filename,
            "type": "document"
        })

    edges = []

    for relationship in relationships:

        edges.append({
            "source": relationship["document_a"],
            "target": relationship["document_b"],
            "weight": relationship["similarity"]
        })

    return {
        "nodes": nodes,
        "edges": edges
    }