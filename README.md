# Campus Lost and Found Information Retrieval System

A beginner-friendly college mini project built with Python, Streamlit, TF-IDF, and cosine similarity. It lets students report lost or found items, search and rank reports, and view a small Precision@K evaluation.

## What is included?

- Search across item name, category, description, and location using real TF-IDF vectors and cosine similarity.
- Optional category, location, and lost/found preferences. Matching category and location provide small ranking boosts.
- A similarity percentage for each result. This percentage is the cosine similarity multiplied by 100; the metadata boosts affect ordering, not the displayed score.
- A report form that appends new records to `data/items.csv`, so records remain after restarting the app.
- Sample records in `data/items.csv` and a separate relevance dataset with review notes for Precision@K.
- A browse view of all reports.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Internet access once to install the Python packages

No paid APIs, external AI services, or database server are used.

## Run it on Windows (beginner steps)

1. Install Python from [python.org](https://www.python.org/downloads/). During installation, select **Add Python to PATH**.
2. Open the project folder in File Explorer: `F:\Engg Projects\lost_and_found_IR`.
3. Click the address bar, type `powershell`, and press Enter. This opens PowerShell in the project folder.
4. Create a virtual environment (a private package area for this project):

   ```powershell
   python -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that same window, then try the activation command again.

6. Install the required packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

7. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

8. Streamlit will open the site in your browser. If it does not, copy the **Local URL** shown in PowerShell (usually `http://localhost:8501`) into your browser.
9. To stop the app, return to PowerShell and press **Ctrl+C**.

Next time, open PowerShell in the same folder, activate `.venv` as in step 5, and run the start command in step 7.

## Run it on macOS or Linux

From a terminal opened in the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## How information retrieval works

1. Text is converted to lowercase, accents and punctuation are removed, and common English stop words (such as “the” and “is”) are removed.
2. The TF-IDF vectorizer represents every report and the search query as weighted word and two-word phrase vectors. Common terms across records carry less weight than distinctive terms.
3. Cosine similarity compares the query vector with each report vector. A score closer to 100% means the text vectors point in a more similar direction. It is a ranking signal, not a probability that the item is a match.
4. Matching category and location preferences add small ranking boosts. If a status is selected, records with the other status are excluded.
5. The app sorts by this combined ranking and displays the text cosine similarity as a percentage.

## Precision@K evaluation

The **Evaluate the system** tab runs every query in `data/evaluation_queries.csv` through the same TF-IDF and cosine-similarity search used by the app. Expected results are not passed into or hardcoded in the search algorithm.

### Evaluation files

- `data/evaluation_queries.csv` contains query IDs, natural-language queries, and optional category and location preferences.
- `data/evaluation_ground_truth.csv` contains one reviewed query/item judgment per row. Each query has at least ten judged item records, including relevant and non-relevant examples. Judgments are `relevant`, `not_relevant`, or `ambiguous`.
- `data/evaluation_label_review.csv` documents how every positive label from the original five-query evaluation was reviewed against the item name, description, category, status, and location.
- `data/evaluation_legacy_backup.csv` preserves the original query definitions and positive item IDs before review.

### How relevance is judged

A record is **relevant** when the item identity and any stated distinguishing attributes (such as color) fit the query, its selected category and requested location are supported by the record, and any clearly requested Lost/Found status is consistent. A record is **not relevant** when a key item, attribute, category, status, or location condition clearly conflicts. Use **ambiguous** when the fields support more than one reasonable interpretation, for example a near-synonym campus location, a status conflict, a location mismatch alongside a strong item match, or a misspelled location. Ambiguous cases are flagged for a person to review; the app does not silently turn them into positive labels.

These judgments are a classroom ground truth and need human review before they are treated as authoritative. If item records change, review the related ground-truth rows rather than inferring new relevance labels from the search results.

To resolve an ambiguous or unjudged result, review its item fields and update the `judgment` value in `data/evaluation_ground_truth.csv` to `relevant` or `not_relevant`; add the reason to `notes`. Keep the legacy backup unchanged.

### Precision calculation

`Precision@K = relevant records in the first K result positions / K`

The app reports Precision@1, Precision@3, Precision@5, and Precision@10 separately. If a top-K result contains ambiguous or unjudged records, the app shows a lower-to-upper range: the lower bound counts only definite relevant records, and the upper bound assumes all unresolved records are eventually judged relevant. Unfilled result positions still count in K and contribute zero. This avoids silently deciding uncertain cases for the model.

## Project files

```text
lost_and_found_IR/
├── app.py              # Interface, text preprocessing, retrieval, and evaluation
├── data/
│   ├── items.csv       # Sample data and saved user reports
│   ├── evaluation_queries.csv          # Query set for evaluation
│   ├── evaluation_ground_truth.csv     # Reviewed query/item relevance judgments
│   ├── evaluation_label_review.csv     # Review of the original positive labels
│   └── evaluation_legacy_backup.csv    # Unmodified backup of original labels
├── requirements.txt    # Python package list
├── README.md           # Setup and project guide
└── .gitignore          # Local files excluded from Git
```

## Notes

- The sample contact details use fictional `@campus.edu` addresses. Replace these with an appropriate contact method before a real deployment.
- The app stores contact details in plain text in the CSV. Use fictional data for a classroom demonstration and do not publish private contact details.
- The sample dates are illustrative.
