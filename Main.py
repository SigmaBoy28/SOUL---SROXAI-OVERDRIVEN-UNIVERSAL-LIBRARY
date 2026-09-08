import pymupdf 

pdf = pymupdf.open("example.pdf")  


paragraphs = []

for page_number, page in enumerate(pdf, start=1):
    text = page.get_text()
    
    page_paragraphs = text.split("\n\n")

    for paragraph in page_paragraphs:
        paragraph = paragraph.strip()

    if paragraph:
       paragraphs.append({
           "page": page_number,
           "text": paragraph
           })

pdf.close()

for paragraph in paragraphs:
    print("-----")
    print("page: ", paragraph["page"])
    print(paragraph["text"])