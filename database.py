import sqlite3
from pathlib import Path


class Database:

    def __init__(self, database_path: Path):

        self.database_path = database_path

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            self.database_path
        )

        self.connection.execute("PRAGMA foreign_keys = ON")

        self.create_tables()


    def create_tables(self):

        cursor = self.connection.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                word_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_keywords (
                document_id INTEGER NOT NULL,
                keyword_id INTEGER NOT NULL,
                frequency INTEGER NOT NULL,

                PRIMARY KEY (
                    document_id,
                    keyword_id
                ),

                FOREIGN KEY (
                    document_id
                )
                REFERENCES documents(id)
                ON DELETE CASCADE,

                FOREIGN KEY (
                    keyword_id
                )
                REFERENCES keywords(id)
                ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS relationships (
                document_a INTEGER NOT NULL,
                document_b INTEGER NOT NULL,
                similarity REAL NOT NULL,

                PRIMARY KEY (
                    document_a,
                    document_b
                ),

                FOREIGN KEY (
                    document_a
                )
                REFERENCES documents(id)
                ON DELETE CASCADE,

                FOREIGN KEY (
                    document_b
                )
                REFERENCES documents(id)
                ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS
            idx_keywords_word
            ON keywords(word);

            CREATE INDEX IF NOT EXISTS
            idx_document_keywords_keyword
            ON document_keywords(keyword_id);
        """)

        self.connection.commit()


    def clear(self):

        self.connection.execute(
            "DELETE FROM relationships"
        )

        self.connection.execute(
            "DELETE FROM document_keywords"
        )

        self.connection.execute(
            "DELETE FROM keywords"
        )

        self.connection.execute(
            "DELETE FROM documents"
        )

        self.connection.commit()


    def add_document(
        self,
        filename: str,
        path: str,
        word_count: int
    ) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO documents (
                filename,
                path,
                word_count
            )
            VALUES (?, ?, ?)
            """,
            (
                filename,
                path,
                word_count
            )
        )

        self.connection.commit()

        return cursor.lastrowid


    def add_keyword(
        self,
        word: str
    ) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT OR IGNORE INTO keywords(word)
            VALUES (?)
            """,
            (word,)
        )

        cursor.execute(
            """
            SELECT id
            FROM keywords
            WHERE word = ?
            """,
            (word,)
        )

        return cursor.fetchone()[0]


    def connect_keyword(
        self,
        document_id: int,
        keyword_id: int,
        frequency: int
    ):

        self.connection.execute(
            """
            INSERT INTO document_keywords (
                document_id,
                keyword_id,
                frequency
            )
            VALUES (?, ?, ?)
            """,
            (
                document_id,
                keyword_id,
                frequency
            )
        )


    def add_relationship(
        self,
        document_a: int,
        document_b: int,
        similarity: float
    ):

        # Always keep the smaller ID first.
        if document_a > document_b:
            document_a, document_b = (
                document_b,
                document_a
            )

        self.connection.execute(
            """
            INSERT OR REPLACE INTO relationships (
                document_a,
                document_b,
                similarity
            )
            VALUES (?, ?, ?)
            """,
            (
                document_a,
                document_b,
                similarity
            )
        )


    def commit(self):

        self.connection.commit()


    def get_document_id(
        self,
        filename: str
    ):

        cursor = self.connection.execute(
            """
            SELECT id
            FROM documents
            WHERE filename = ?
            """,
            (filename,)
        )

        result = cursor.fetchone()

        return result[0] if result else None


    def search_keyword(
        self,
        word: str
    ):

        cursor = self.connection.execute(
            """
            SELECT
                d.filename,
                dk.frequency
            FROM document_keywords dk

            JOIN documents d
                ON d.id = dk.document_id

            JOIN keywords k
                ON k.id = dk.keyword_id

            WHERE k.word = ?

            ORDER BY dk.frequency DESC
            """,
            (word.lower(),)
        )

        return cursor.fetchall()


    def related_documents(
        self,
        filename: str
    ):

        cursor = self.connection.execute(
            """
            SELECT
                CASE
                    WHEN d1.filename = ?
                    THEN d2.filename
                    ELSE d1.filename
                END AS related_document,

                r.similarity

            FROM relationships r

            JOIN documents d1
                ON d1.id = r.document_a

            JOIN documents d2
                ON d2.id = r.document_b

            WHERE d1.filename = ?
               OR d2.filename = ?

            ORDER BY r.similarity DESC
            """,
            (
                filename,
                filename,
                filename
            )
        )

        return cursor.fetchall()


    def get_all_relationships(self):

        cursor = self.connection.execute(
            """
            SELECT
                d1.filename,
                d2.filename,
                r.similarity

            FROM relationships r

            JOIN documents d1
                ON d1.id = r.document_a

            JOIN documents d2
                ON d2.id = r.document_b

            ORDER BY r.similarity DESC
            """
        )

        return cursor.fetchall()


    def close(self):

        self.connection.close()