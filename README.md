# Findora-Lost and Found Retrieval System

## Technologies Used

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-TF--IDF%20%26%20Cosine-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-CSV%20Data-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img alt="Information Retrieval" src="https://img.shields.io/badge/IR-Ranked%20Search-247B83?style=for-the-badge" />
</div>

A campus web application for reporting lost or found belongings and searching existing reports. Instead of relying on exact keyword matches, it uses Information Retrieval techniques to rank records that are textually similar to a student's description.

This project was created as a beginner-friendly college mini project for an Information Retrieval course. It demonstrates text preprocessing, TF-IDF weighting, cosine similarity, ranked search, CSV data storage, and Precision@K evaluation.

## Project Overview

Students can submit details about an item they lost or found, including its name, category, description, location, date, and contact information. Other students can describe an item in everyday language and see reports ranked by how closely they match.

For example, a search for **“wallet I left in canteen”** can match reports using terms such as *purse* and *cafeteria*. The application keeps its records in CSV files, so new reports are still available after the app is restarted.

## Main Features

- Search reports with TF-IDF and cosine similarity.
- Understand common word variations and synonyms, including *wallet/purse*, *canteen/cafeteria*, and *charger/adapter*.
- Use item name, description, category, location, and color clues to rank possible matches.
- Narrow or prioritize results by category, location, and Lost/Found status.
- See a similarity percentage beside each result. This is a text similarity score, not a probability that the item is a match.
- Submit lost and found reports through the app. Reports are saved in `data/items.csv`.
- Browse all records and review the app's Precision@K evaluation.
- Run locally without paid APIs or external AI services.

## How Search Works

1. The app reads item reports from `data/items.csv`.
2. It normalizes text by lowercasing, removing punctuation and common filler words, handling simple word variations, and mapping known synonyms to shared terms.
3. TF-IDF converts the search and report text into weighted word and two-word phrase vectors. Distinctive terms receive more weight than common terms.
4. Cosine similarity compares the search vector with report vectors. Additional field scores help item names, descriptions, and locations contribute to ranking.
5. Category, location, and color clues can adjust ordering. The app displays the overall text cosine similarity as a percentage.

## Evaluation and Relevance Judgments

Open the **Evaluate the system** tab to run the eight saved queries against the current item records. The search algorithm does not read or use the expected relevance labels; the labels are used only after retrieval to measure its results.

Evaluation files:

- `data/evaluation_queries.csv` contains query text and optional category and location preferences.
- `data/evaluation_ground_truth.csv` contains reviewed judgments for query and item pairs. Each query has ten judged records with notes.
- `data/evaluation_label_review.csv` records how the original positive labels were reviewed.
- `data/evaluation_legacy_backup.csv` preserves the original evaluation labels as a backup.

Judgments are `relevant`, `not_relevant`, or `ambiguous`. A record is relevant when its item type and requested details match the query and the selected category, location, and any clearly stated Lost/Found status are supported. A record is not relevant when an important part of the query conflicts with its details. If the record could reasonably match but needs a person to resolve a detail, it is marked ambiguous.

The app reports Precision@1, Precision@3, Precision@5, and Precision@10 separately:

```text
Precision@K = relevant records in the first K result positions / K
```

When ambiguous or unjudged records appear in the first K positions, the app displays a lower-to-upper range. The lower bound counts only definite relevant records; the upper bound assumes all unresolved records are later judged relevant. Empty result positions still count in K and contribute zero.

To resolve a judgment, inspect the item's name, description, category, status, and location. Then update the `judgment` and `notes` columns in `data/evaluation_ground_truth.csv`. Keep the legacy backup unchanged.

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- Internet access to install the packages the first time

## Run on Windows

1. Install Python from [python.org](https://www.python.org/downloads/). Select **Add Python to PATH** during installation.
2. Open the project folder in File Explorer: `F:\Engg Projects\lost_and_found_IR`.
3. Click the address bar, type `powershell`, and press Enter.
4. Create a virtual environment for the project:

   ```powershell
   python -m venv .venv
   ```

5. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window, then try again.

6. Install the packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

7. Start the application:

   ```powershell
   python -m streamlit run app.py
   ```

8. Open the local URL shown in PowerShell, usually `http://localhost:8501`.
9. To stop the app, return to PowerShell and press **Ctrl+C**.

For later sessions, open PowerShell in the project folder, activate `.venv` as in step 5, and run the command from step 7.

## Run on macOS or Linux

Open a terminal in the project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Project Files

```text
lost_and_found_IR/
├── app.py                              # Streamlit interface, retrieval, and evaluation display
├── data/
│   ├── items.csv                       # Sample reports and saved user reports
│   ├── evaluation_queries.csv          # Evaluation query set
│   ├── evaluation_ground_truth.csv     # Reviewed relevance judgments
│   ├── evaluation_label_review.csv     # Review of original positive labels
│   └── evaluation_legacy_backup.csv    # Backup of original evaluation labels
├── requirements.txt                    # Python package requirements
├── README.md                           # Project guide
└── .gitignore                          # Files Git should ignore
```

## Data and Privacy Note

Item contact information is stored as plain text in the CSV file. Use fictional contact details for demonstrations, and remove private details before sharing or publishing the project. The item reports are classroom sample data and may be edited for your campus.

## Author

**Adtiya Vikram**  
Email: [1dt24cy003@dsatm.edu.in](mailto:1dt24cy003@dsatm.edu.in)
