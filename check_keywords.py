import json
from rag import load_pdf

pages = load_pdf("docs/paper.pdf")
text = " ".join(t for _, t in pages).lower()
for item in json.load(open("eval_set.json")):
    if item["keyword"].lower() not in text:
        print("NOT FOUND:", item["keyword"])
print("Check finished.")