import json
from pathlib import Path

import pymupdf

from config import (
    DOCUMENTS_DIR,
    DATABASE_PATH,
    DATA_DIR,
    SIMILARITY_THRESHOLD,
)

from processor import (
    process_text,
    word_frequency,
    build_keyword_index,
    calculate_idf,
    build_tfidf_vectors,
)

from database import Database

from graph import (
    build_relationships,
    build_graph,
)

from search import (
    search,
    show_related_documents,
)


def extract_pdf(filepath: Path) -> str:

    pdf = pymupdf.open(filepath)

    pages = []

    try:

        for page in pdf:

            text = page.get_text().strip()

            if text:
                pages.append(text)

    finally:

        pdf.close()

    return "\n\n".join(pages)


def load_documents():

    documents = {}

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print(
            f"No PDF files found in: "
            f"{DOCUMENTS_DIR}"
        )

        return documents

    for filepath in pdf_files:

        print(
            f"Reading: {filepath.name}"
        )

        try:

            text = extract_pdf(filepath)

            words = process_text(text)

            documents[filepath.name] = words

            print(
                f"  {len(words):,} processed words"
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

    return documents


def build_database(
    documents,
    relationships
):

    database = Database(
        DATABASE_PATH
    )

    # Rebuild from scratch.
    # This guarantees that deleted/changed PDFs
    # don't leave stale relationships.
    database.clear()

    document_ids = {}

    for filename, words in documents.items():

        path = DOCUMENTS_DIR / filename

        document_id = database.add_document(
            filename=filename,
            path=str(path),
            word_count=len(words)
        )

        document_ids[filename] = document_id

        frequencies = word_frequency(words)

        for word, frequency in frequencies.items():

            keyword_id = database.add_keyword(
                word
            )

            database.connect_keyword(
                document_id,
                keyword_id,
                frequency
            )

    for relationship in relationships:

        file_a = relationship["document_a"]
        file_b = relationship["document_b"]

        database.add_relationship(
            document_ids[file_a],
            document_ids[file_b],
            relationship["similarity"]
        )

    database.commit()

    return database


def save_graph(graph):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    graph_path = DATA_DIR / "graph.json"

    with open(
        graph_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            graph,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nGraph saved to: {graph_path}"
    )


def print_relationships(
    relationships
):

    print("\n" + "=" * 60)
    print("DOCUMENT RELATIONSHIPS")
    print("=" * 60)

    if not relationships:

        print(
            "No relationships found."
        )

        return

    for relationship in relationships:

        print(
            f"\n{relationship['document_a']}"
        )

        print(
            f"    ↕ "
            f"{relationship['similarity']:.3f}"
        )

        print(
            f"{relationship['document_b']}"
        )


def interactive_search(database):

    print("\n")
    print("=" * 60)
    print("KNOWLEDGE WEB SEARCH")
    print("=" * 60)

    print(
        "Type a keyword to find documents."
    )

    print(
        "Type 'related <filename>' to find "
        "related documents."
    )

    print(
        "Type 'graph' to show relationships."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        command = input(
            "\n> "
        ).strip()

        if not command:
            continue

        if command.lower() == "exit":
            break

        if command.lower() == "graph":

            relationships = (
                database.get_all_relationships()
            )

            for (
                file_a,
                file_b,
                similarity
            ) in relationships:

                print(
                    f"{file_a} "
                    f"<-> "
                    f"{file_b} "
                    f"({similarity:.3f})"
                )

            continue

        if command.lower().startswith(
            "related "
        ):

            filename = command[9:].strip()

            results = show_related_documents(
                database,
                filename
            )

            if not results:

                print(
                    "No related documents found."
                )

            else:

                for (
                    related,
                    similarity
                ) in results:

                    print(
                        f"{related} "
                        f"({similarity:.3f})"
                    )

            continue

        results = search(
            database,
            command
        )

        if not results:

            print(
                "No documents contain that keyword."
            )

            continue

        print(
            f"\nDocuments containing "
            f"'{command}':"
        )

        for filename, frequency in results:

            print(
                f"  {filename} "
                f"— {frequency} occurrence(s)"
            )


def main():

    print("=" * 60)
    print("SOUL")
    print("SROXAI OVERDRIVEN UNIVERSAL LIBRARY")
    print("=" * 60)

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. LOAD PDFs
    # --------------------------------------------------

    documents = load_documents()

    if not documents:

        return

    # --------------------------------------------------
    # 2. BUILD KEYWORD INDEX
    # --------------------------------------------------

    print("\nBuilding keyword index...")

    keyword_index = build_keyword_index(
        documents
    )

    print(
        f"Unique keywords: "
        f"{len(keyword_index):,}"
    )

    # --------------------------------------------------
    # 3. TF-IDF
    # --------------------------------------------------

    print(
        "\nCalculating document importance..."
    )

    idf = calculate_idf(
        documents
    )

    tfidf_vectors = build_tfidf_vectors(
        documents,
        idf
    )

    # --------------------------------------------------
    # 4. BUILD DOCUMENT RELATIONSHIPS
    # --------------------------------------------------

    print(
        "\nBuilding document relationships..."
    )

    relationships = build_relationships(
        documents,
        tfidf_vectors,
        SIMILARITY_THRESHOLD
    )

    print(
        f"Relationships found: "
        f"{len(relationships):,}"
    )

    print_relationships(
        relationships
    )

    # --------------------------------------------------
    # 5. BUILD GRAPH
    # --------------------------------------------------

    graph = build_graph(
        documents,
        relationships
    )

    save_graph(
        graph
    )

    # --------------------------------------------------
    # 6. SAVE DATABASE
    # --------------------------------------------------

    print(
        "\nBuilding SQLite database..."
    )

    database = build_database(
        documents,
        relationships
    )

    print(
        f"Database saved to: "
        f"{DATABASE_PATH}"
    )

    # --------------------------------------------------
    # 7. INTERACTIVE SEARCH
    # --------------------------------------------------

    try:

        interactive_search(
            database
        )

    finally:

        database.close()


if __name__ == "__main__":
    main()