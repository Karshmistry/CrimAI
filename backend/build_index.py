import os, cv2, pickle, numpy as np
from insightface.app import FaceAnalysis
import faiss

# Load pretrained face model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640,640))

embeddings, ids = [], []

# Loop through gallery images
for file in os.listdir("gallery"):
    path = os.path.join("gallery", file)
    img = cv2.imread(path)
    faces = app.get(img)
    if faces:
        emb = faces[0].embedding.astype("float32")
        emb /= np.linalg.norm(emb) + 1e-9
        embeddings.append(emb)
        ids.append(file)
        print(f"✅ Processed: {file}")

# Convert to FAISS index
embeddings = np.vstack(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save index
pickle.dump({"index": index, "ids": ids}, open("faiss_gallery.pkl", "wb"))
print("🎉 Gallery index saved with", len(ids), "faces")
