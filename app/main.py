import os
import json
import datetime
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DIR = "app/input"
OUTPUT_DIR = "app/output"
PERSONA_FILE = os.path.join(INPUT_DIR, "persona.json")

def load_persona():
    with open(PERSONA_FILE, "r") as f:
        return json.load(f)

def extract_text_from_pdfs():
    pdf_texts = {}
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(INPUT_DIR, filename)
            with pdfplumber.open(filepath) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                pdf_texts[filename] = pages
    return pdf_texts

def get_embeddings(model, texts):
    return model.encode(texts, convert_to_tensor=True)

def split_into_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    return paragraphs

def main():
    # Step 1: Load persona and job
    persona_data = load_persona()
    persona_raw = persona_data["persona"]
    job_raw = persona_data["job_to_be_done"]

    # Handle both flat and nested formats
    if isinstance(persona_raw, dict):
        persona = persona_raw.get("role") or persona_raw.get("name") or ""
    else:
        persona = str(persona_raw)

    if isinstance(job_raw, dict):
        job = job_raw.get("task") or job_raw.get("description") or ""
    else:
        job = str(job_raw)

    persona_query = f"{persona} - {job}"


    # Load model (small + fast)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed persona+job
    persona_embedding = get_embeddings(model, [persona_query])

    # Step 2: Read PDFs page by page
    pdf_texts = extract_text_from_pdfs()
    extracted_sections = []

    for doc, pages in pdf_texts.items():
        for i, text in enumerate(pages):
            if not text.strip():
                continue
            page_embedding = get_embeddings(model, [text])
            similarity = cosine_similarity(persona_embedding, page_embedding)[0][0]

            extracted_sections.append({
                "document": doc,
                "page_number": i + 1,
                "section_title": f"Page {i + 1}",
                "similarity_score": float(similarity)
            })

    # Step 3: Sort by relevance
    extracted_sections.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Keep top 5
    for rank, item in enumerate(extracted_sections[:5], start=1):
        item["importance_rank"] = rank
        del item["similarity_score"]  # not needed in final output

    # Step 4: Extract refined sub-sections (from top 5 pages)
    subsection_analysis = []
    for item in extracted_sections[:5]:
        doc = item["document"]
        page_number = item["page_number"]
        full_text = pdf_texts[doc][page_number - 1]

        paragraphs = split_into_paragraphs(full_text)
        if not paragraphs:
            continue

        para_embeddings = get_embeddings(model, paragraphs)
        similarities = cosine_similarity(persona_embedding, para_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:2]  # top 2

        for idx in top_indices:
            subsection_analysis.append({
                "document": doc,
                "page_number": page_number,
                "refined_text": paragraphs[idx]
            })

    # Step 5: Build output JSON
    output = {
        "metadata": {
            "input_documents": list(pdf_texts.keys()),
            "persona": persona_data["persona"],
            "job_to_be_done": persona_data["job_to_be_done"],
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections[:5],
        "subsection_analysis": subsection_analysis
    }

    # Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "output.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("✅ Done! Output saved to output/output.json")

if __name__ == "__main__":
    main()


