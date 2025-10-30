import faiss
import json

index = faiss.read_index("index.faiss.new/vectors.index")
with open("index.faiss.new/index_meta.json") as f:
    meta = json.load(f)

print("FAISS index loaded successfully!")
print("Number of metadata entries:", len(meta))
print("Index info:", index)
