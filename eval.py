import json
from rag import load_pdf, chunk, build_index, search

pages = load_pdf("docs/paper.pdf")
questions = json.load(open("eval_set.json"))

print("chunk_size | top_k | hit_rate")
for size in [400, 800, 1200]:
    chunks = chunk(pages, size=size, overlap=size // 5)
    index = build_index(chunks)
    for k in [2, 4, 6]:
        hits = 0
        for item in questions:
            found = search(index, chunks, item["q"], k=k)
            if any(item["keyword"].lower() in c["text"].lower() for c in found):
                hits += 1
        print(f"{size} | {k} | {hits / len(questions):.0%}")