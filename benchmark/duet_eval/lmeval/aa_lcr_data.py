"""AA-LCR (Artificial Analysis Long Context Reasoning, v1.1, Hub dataset ArtificialAnalysis/AA-LCR): 100 questions, each over a
set of real documents of ~100K tokens.  Builds the exam file aa_lcr_v1.1.jsonl from the two published files: one record per
question with the official prompt (the dataset card's template: the documents in the order of data_source_filenames, each as
"BEGIN DOCUMENT i: ... END DOCUMENT i", then the question) and the official answer as stored in the csv.

    python -m benchmark.duet_eval.lmeval.aa_lcr_data --src DIR (AA-LCR_Dataset.csv, AA-LCR_extracted-text.zip) --out DIR"""
import argparse
import csv
import json
import pathlib
import unicodedata
import zipfile

PROMPT = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


def zip_texts(path):
    """{NFC member path: text}; the archive stores UTF-8 names without the UTF-8 flag (zipfile reads them as cp437), some in NFD
    (macOS), while the csv names them in NFC."""
    texts = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename if info.flag_bits & 0x800 else info.filename.encode("cp437").decode("utf-8")
            texts[unicodedata.normalize("NFC", name)] = z.read(info).decode("utf-8")
    return texts


def build(src, out):
    src, out = pathlib.Path(src), pathlib.Path(out)
    texts = zip_texts(src / "AA-LCR_extracted-text.zip")
    records = []
    with open(src / "AA-LCR_Dataset.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names = row["data_source_filenames"].split(";")
            docs = [texts[unicodedata.normalize("NFC", f"lcr/{row['document_category']}/{row['document_set_id']}/{name}")] for name in names]
            documents_text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
            records.append({"question_id": int(row["question_id"]), "document_category": row["document_category"],
                            "document_set_id": row["document_set_id"], "question": row["question"], "answer": row["answer"],
                            "n_documents": len(docs), "input_tokens_cl100k": int(row["input_tokens"]),
                            "prompt": PROMPT.format(documents_text=documents_text, question=row["question"])})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "aa_lcr_v1.1.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} questions, {sum(r['n_documents'] for r in records)} documents -> {out / 'aa_lcr_v1.1.jsonl'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.src, a.out)


if __name__ == "__main__":
    main()
