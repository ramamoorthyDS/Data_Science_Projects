from flask import Flask, render_template, request
import os
import pdfplumber
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

app = Flask(__name__)

# Create folders
os.makedirs("pdf_files", exist_ok=True)
os.makedirs("text_files", exist_ok=True)
os.makedirs("clean_text", exist_ok=True)

# Download NLTK
nltk.download('punkt')
nltk.download('stopwords')

@app.route("/", methods=["GET", "POST"])

def home():

    results = []
    paper_keywords = {}

    # Upload PDFs
    if request.method == "POST":

        uploaded_files = request.files.getlist("pdf_files")

        # Save uploaded PDFs
        for uploaded_file in uploaded_files:

            if uploaded_file.filename != "":

                pdf_path = os.path.join(
                    "pdf_files",
                    uploaded_file.filename
                )

                uploaded_file.save(pdf_path)

        # PDF to Text
        for file in os.listdir("pdf_files"):

            if file.endswith(".pdf"):

                text = ""

                with pdfplumber.open(
                    os.path.join("pdf_files", file)
                ) as pdf:

                    for page in pdf.pages:

                        page_text = page.extract_text()

                        if page_text:
                            text += page_text

                with open(
                    os.path.join(
                        "text_files",
                        file.replace(".pdf", ".txt")
                    ),
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(text)

        # Preprocessing
        stop_words = set(stopwords.words("english"))

        for file in os.listdir("text_files"):

            if file.endswith(".txt"):

                with open(
                    os.path.join("text_files", file),
                    "r",
                    encoding="utf-8"
                ) as f:

                    text = f.read().lower()

                tokens = word_tokenize(text)

                tokens = [
                    word for word in tokens
                    if word.isalpha()
                ]

                tokens = [
                    word for word in tokens
                    if word not in stop_words
                ]

                clean_text = " ".join(tokens)

                with open(
                    os.path.join("clean_text", file),
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(clean_text)

        # Read cleaned documents
        documents = []
        filenames = []

        for file in os.listdir("clean_text"):

            if file.endswith(".txt"):

                with open(
                    os.path.join("clean_text", file),
                    "r",
                    encoding="utf-8"
                ) as f:

                    doc = f.read()

                    documents.append(doc)

                    filenames.append(file)

        # Keyword Extraction
        for i, doc in enumerate(documents):

            vectorizer = TfidfVectorizer(max_features=5)

            X = vectorizer.fit_transform([doc])

            keywords = vectorizer.get_feature_names_out()

            paper_keywords[filenames[i]] = keywords

        # BM25
        tokenized_docs = [
            doc.split()
            for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_docs)

        # Search Query
        query = request.form.get("query")

        if query:

            query_tokens = query.lower().split()

            scores = bm25.get_scores(query_tokens)

            ranked_docs = sorted(
                zip(scores, filenames),
                reverse=True
            )

            results = ranked_docs[:5]

    return render_template(
        "index.html",
        results=results,
        keywords=paper_keywords
    )

if __name__ == "__main__":

    app.run(debug=True)