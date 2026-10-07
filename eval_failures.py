import json
from rag import load_pdf, chunk, build_index, search

pages = load_pdf("docs/paper.pdf")
questions = json.load(open("eval_set.json"))
chunks = chunk(pages, size=400, overlap=80)
index = build_index(chunks)

for item in questions:
    found = search(index, chunks, item["q"], k=6)
    if not any(item["keyword"].lower() in c["text"].lower() for c in found):
        print("MISSED:", item["q"])