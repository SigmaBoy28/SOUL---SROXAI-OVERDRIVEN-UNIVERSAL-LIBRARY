from database import Database


def search(
    database: Database,
    keyword: str
):

    keyword = keyword.strip().lower()

    if not keyword:
        return []

    return database.search_keyword(keyword)


def show_related_documents(
    database: Database,
    filename: str
):

    return database.related_documents(filename)