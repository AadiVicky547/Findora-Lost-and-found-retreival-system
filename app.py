"""A small, beginner-friendly lost and found search application."""

from pathlib import Path
import re
import unicodedata

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_FILE = DATA_DIR / "items.csv"
EVALUATION_QUERIES_FILE = DATA_DIR / "evaluation_queries.csv"
EVALUATION_GROUND_TRUTH_FILE = DATA_DIR / "evaluation_ground_truth.csv"
ITEM_COLUMNS = ["item_id", "status", "item_name", "category", "description", "location", "date", "contact"]

# Small hand-written synonym groups keep the project offline and easy to explain.
# Every term in a group is normalized to the first (canonical) term.
SYNONYM_GROUPS = {
    "wallet": {"wallet", "purse", "billfold"},
    "cafeteria": {"cafeteria", "canteen", "diningroom", "foodcourt"},
    "phone": {"phone", "mobile", "cellphone", "smartphone"},
    "glasses": {"glasses", "spectacles", "eyeglasses"},
    "backpack": {"backpack", "rucksack", "schoolbag"},
    "bottle": {"bottle", "flask", "waterbottle"},
    "charger": {"charger", "adapter", "poweradapter", "charge", "charging", "charged", "charg"},
    "notebook": {"notebook", "journal", "notepad"},
    "keychain": {"keychain", "keyring", "keyfob"},
    "idcard": {"idcard", "identitycard", "studentcard"},
    "engineer": {"engineer", "engineering", "engg"},
}
SYNONYM_LOOKUP = {word: canonical for canonical, words in SYNONYM_GROUPS.items() for word in words}
EXTRA_STOP_WORDS = {
    "left", "lost", "found", "looking", "look", "think", "maybe", "please",
    "somewhere", "something", "item", "thing", "belongings", "remember",
    "where", "when", "did", "does", "was", "were", "have", "has", "had",
    "the", "this", "that", "with", "near", "around", "inside", "outside",
}
CAMPUS_COLORS = {"black", "white", "red", "blue", "green", "yellow", "brown", "grey", "gray", "silver", "orange", "purple"}


def clean_text(text: str) -> str:
    """Normalize text, remove conversational filler, and fold simple word forms."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())
    # Handle common multiword forms before splitting into individual words.
    text = re.sub(r"\bdining\s+hall\b|\bfood\s+court\b", " cafeteria ", text)
    text = re.sub(r"\bengg\s+block\b|\bengineering\s+building\b|\bengg\s+building\b", " engineering block ", text)
    text = re.sub(r"\bwater\s+bottle\b", " waterbottle ", text)
    text = re.sub(r"\bpower\s+adapter\b", " poweradapter ", text)
    text = re.sub(r"\bkey\s+ring\b", " keyring ", text)
    text = re.sub(r"\bkey\s+fob\b", " keyfob ", text)
    text = re.sub(r"\bid\s+card\b|\bidentity\s+card\b|\bstudent\s+card\b", " idcard ", text)

    words = []
    for word in text.split():
        if word in ENGLISH_STOP_WORDS or word in EXTRA_STOP_WORDS:
            continue
        word = SYNONYM_LOOKUP.get(word, word)
        # A small suffix stemmer handles ordinary plurals and common endings
        # without an extra package or a model download.
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("sses") and len(word) > 5:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
            word = word[:-1]
        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        words.append(SYNONYM_LOOKUP.get(word, word))
    return " ".join(words)


def load_items() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        pd.DataFrame(columns=ITEM_COLUMNS).to_csv(DATA_FILE, index=False)
    data = pd.read_csv(DATA_FILE, dtype=str).fillna("")
    for column in ITEM_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    return data[ITEM_COLUMNS]


def searchable_text(row: pd.Series) -> str:
    # Repetition is a simple field-weighting technique: title and location
    # terms matter more than incidental words in a longer description.
    return clean_text(" ".join(
        [row["item_name"]] * 4
        + [row["location"]] * 3
        + [row["category"]] * 2
        + [row["description"]]
    ))


def search_items(query: str, category: str = "Any category", location: str = "", status: str = "Any status", limit: int = 10) -> pd.DataFrame:
    """Rank records with TF-IDF cosine scores for text, names, descriptions, and places."""
    records = load_items()
    if records.empty or not query.strip():
        return records.iloc[0:0].assign(similarity=pd.Series(dtype=float))

    documents = records.apply(searchable_text, axis=1).tolist()
    names = records["item_name"].apply(clean_text).tolist()
    places = records["location"].apply(clean_text).tolist()
    descriptions = records["description"].apply(clean_text).tolist()
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return records.iloc[0:0].assign(similarity=pd.Series(dtype=float))
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(documents + [cleaned_query])
        name_matrix = vectorizer.transform(names)
        location_matrix = vectorizer.transform(places)
        description_matrix = vectorizer.transform(descriptions)
    except ValueError:  # Handles an empty vocabulary in an unusually sparse dataset.
        return records.iloc[0:0].assign(similarity=pd.Series(dtype=float))

    similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    name_similarity = cosine_similarity(matrix[-1], name_matrix).ravel()
    description_similarity = cosine_similarity(matrix[-1], description_matrix).ravel()

    # Use the explicit location field when provided. Otherwise, find a known
    # campus place mentioned in the natural-language query (for example,
    # "near the engineering block") and compare that place on its own.
    location_query = clean_text(location)
    if not location_query:
        known_places = sorted(set(places), key=len, reverse=True)
        location_query = next((place for place in known_places if place and place in cleaned_query), "")
    query_location_vector = vectorizer.transform([location_query]) if location_query else None
    if query_location_vector is not None:
        location_similarity = cosine_similarity(query_location_vector, location_matrix).ravel()
    else:
        # Reuse the NumPy score vector as a numeric zero vector with one entry
        # per record. A Python list cannot be multiplied by location_weight.
        location_similarity = similarities.copy()
        location_similarity[:] = 0.0

    # Keep the combined TF-IDF score central, while rewarding item-name,
    # description, and location evidence separately. Description weight helps
    # match attributes such as a black finish even when the title is brief.
    location_weight = 0.10 if location.strip() else 0.04
    scores = (
        0.25 * similarities
        + 0.20 * name_similarity
        + 0.45 * description_similarity
        + location_weight * location_similarity
    )
    preferred_location_terms = set(location_query.split())
    query_colors = set(cleaned_query.split()) & CAMPUS_COLORS
    for index, (_, item) in enumerate(records.iterrows()):
        # A selected category strongly favors same-category matches and pushes
        # other categories down without replacing the text similarity score.
        if category != "Any category":
            if item["category"].casefold() == category.casefold():
                scores[index] += 0.32
            else:
                scores[index] -= 0.20
        item_location_terms = set(clean_text(item["location"]).split())
        if preferred_location_terms and preferred_location_terms.intersection(item_location_terms):
            scores[index] += 0.12 if location.strip() else 0.04
        elif location_query:
            scores[index] -= 0.03 if location.strip() else 0.01

        # TF-IDF rewards a shared color word but cosine alone does not penalize
        # a contradictory color. Apply a small attribute-aware reranking step.
        item_colors = set(clean_text(f"{item['item_name']} {item['description']}").split()) & CAMPUS_COLORS
        if query_colors and query_colors.intersection(item_colors):
            scores[index] += 0.05
        elif query_colors and item_colors:
            scores[index] -= 0.14
        if status != "Any status" and item["status"].casefold() != status.casefold():
            scores[index] = -1

    result = records.copy()
    # Similarity percentage remains the actual cosine score; metadata is used for ranking.
    result["similarity"] = similarities * 100
    result["_rank"] = scores
    result = result[result["_rank"] > 0].sort_values(["_rank", "similarity"], ascending=False)
    return result.drop(columns="_rank").head(limit).reset_index(drop=True)


def add_item(status: str, item_name: str, category: str, description: str, location: str, date: str, contact: str) -> None:
    records = load_items()
    next_number = 1
    existing_ids = set(records["item_id"])
    while f"LF{next_number:03d}" in existing_ids:
        next_number += 1
    new_row = pd.DataFrame([{
        "item_id": f"LF{next_number:03d}", "status": status, "item_name": item_name.strip(),
        "category": category, "description": description.strip(), "location": location.strip(),
        "date": date, "contact": contact.strip(),
    }])
    pd.concat([records, new_row], ignore_index=True).to_csv(DATA_FILE, index=False)


st.set_page_config(page_title="Campus Lost & Found", page_icon="🔎", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  .hero {padding: 1.5rem 1.8rem; border-radius: 16px; background: linear-gradient(120deg,#12304a,#247b83); color: white; margin-bottom: 1.2rem;}
  .hero h1 {margin: 0 0 .35rem 0; color: white;}
  .hero p {margin: 0; opacity: .92;}
  div[data-testid="stMetric"] {background: #f3f7f8; padding: 1rem; border-radius: 12px;}
</style>
<div class="hero"><h1>🔎 Campus Lost &amp; Found</h1><p>Search reports with information retrieval, or add a lost or found item to the shared register.</p></div>
""", unsafe_allow_html=True)

items = load_items()
lost_count = (items["status"] == "Lost").sum()
found_count = (items["status"] == "Found").sum()
m1, m2, m3 = st.columns(3)
m1.metric("Total reports", len(items))
m2.metric("Lost reports", int(lost_count))
m3.metric("Found reports", int(found_count))

tab_search, tab_report, tab_evaluation, tab_browse = st.tabs(["Search items", "Report an item", "Evaluate the system", "Browse reports"])

with tab_search:
    st.subheader("Describe what you are looking for")
    with st.form("search_form"):
        query = st.text_input("Search description", placeholder="e.g. blue water bottle left near the library")
        c1, c2, c3 = st.columns(3)
        category_options = ["Any category"] + sorted(items["category"].dropna().unique().tolist())
        selected_category = c1.selectbox("Category", category_options)
        search_location = c2.text_input("Location (optional)", placeholder="e.g. library")
        selected_status = c3.selectbox("Show", ["Any status", "Lost", "Found"])
        submitted = st.form_submit_button("Find matching reports", type="primary")
    if submitted:
        results = search_items(query, selected_category, search_location, selected_status)
        if results.empty:
            st.info("No matching reports found. Try a few different words or broaden your filters.")
        else:
            st.caption("Results combine TF-IDF cosine similarity across report text, item names, descriptions, and locations. Matching colors and selected categories also influence rank. The percentage shown is the overall text cosine similarity.")
            for _, item in results.iterrows():
                with st.container(border=True):
                    title_col, score_col = st.columns([5, 1])
                    title_col.markdown(f"### {item['item_name']}  ·  {item['status']}")
                    score_col.metric("Similarity", f"{item['similarity']:.1f}%")
                    st.write(item["description"])
                    st.caption(f"{item['category']}  |  {item['location']}  |  {item['date']}  |  Contact: {item['contact']}")

with tab_report:
    st.subheader("Add a report")
    st.write("Your report is saved in the project CSV file and will still be there after the app restarts.")
    with st.form("report_form", clear_on_submit=True):
        report_status = st.selectbox("Report type", ["Lost", "Found"])
        report_name = st.text_input("Item name", placeholder="e.g. Black umbrella")
        report_category = st.selectbox("Category", ["Electronics", "ID / Cards", "Keys", "Bags", "Clothing", "Books / Stationery", "Accessories", "Drinkware", "Other"])
        report_description = st.text_area("Description", placeholder="Color, brand, distinctive marks, contents...")
        report_location = st.text_input("Where was it lost or found?")
        report_date = st.date_input("Date")
        report_contact = st.text_input("Contact information", placeholder="Email or phone number")
        save_report = st.form_submit_button("Save report", type="primary")
    if save_report:
        if not all([report_name.strip(), report_description.strip(), report_location.strip(), report_contact.strip()]):
            st.error("Please fill in the item name, description, location, and contact information.")
        else:
            add_item(report_status, report_name, report_category, report_description, report_location, str(report_date), report_contact)
            st.success("Report saved. It is now available in search and browse.")
            st.rerun()

with tab_evaluation:
    st.subheader("Precision@K evaluation")
    st.write("This evaluation uses the separate draft ground-truth files in `data/`. It does not supply expected results to the search algorithm.")
    st.markdown(
        "**Precision@K = relevant results in the first K positions ÷ K.** "
        "An ambiguous or unjudged result is not silently counted as relevant or irrelevant: "
        "the displayed range is a lower bound (definitely relevant only) to an upper bound "
        "(if every unresolved result is later judged relevant). Empty result positions still count in K."
    )

    if not EVALUATION_QUERIES_FILE.exists() or not EVALUATION_GROUND_TRUTH_FILE.exists():
        st.error("Evaluation files are missing. Check data/evaluation_queries.csv and data/evaluation_ground_truth.csv.")
    else:
        evaluation_queries = pd.read_csv(EVALUATION_QUERIES_FILE, dtype=str).fillna("")
        ground_truth = pd.read_csv(EVALUATION_GROUND_TRUTH_FILE, dtype=str).fillna("")
        valid_judgments = {"relevant", "not_relevant", "ambiguous"}
        bad_judgments = set(ground_truth["judgment"].str.lower()) - valid_judgments
        if bad_judgments:
            st.error(f"Unknown ground-truth judgment(s): {', '.join(sorted(bad_judgments))}")
        else:
            ground_truth["judgment"] = ground_truth["judgment"].str.lower()
            ks = (1, 3, 5, 10)
            summary_rows = []
            detail_rows = []
            for _, test_case in evaluation_queries.iterrows():
                query_id = test_case["query_id"]
                retrieved = search_items(
                    test_case["query"], test_case["category"], test_case["location"], limit=10
                )
                query_labels = ground_truth[ground_truth["query_id"] == query_id].set_index("item_id")
                result_ids = retrieved["item_id"].tolist()
                row = {
                    "Query": test_case["query"],
                    "Returned": f"{len(result_ids)}/10",
                }
                for k in ks:
                    judged = [
                        query_labels.loc[item_id, "judgment"] if item_id in query_labels.index else "unjudged"
                        for item_id in result_ids[:k]
                    ]
                    definite = sum(label == "relevant" for label in judged)
                    unresolved = sum(label in {"ambiguous", "unjudged"} for label in judged)
                    lower = definite / k
                    upper = (definite + unresolved) / k
                    row[f"P@{k}"] = f"{lower:.2f}" if lower == upper else f"{lower:.2f}–{upper:.2f}*"
                summary_rows.append(row)

                for rank, (_, item) in enumerate(retrieved.iterrows(), 1):
                    if item["item_id"] in query_labels.index:
                        judgment_row = query_labels.loc[item["item_id"]]
                        judgment = judgment_row["judgment"]
                        review_note = judgment_row.get("notes", "")
                    else:
                        judgment = "unjudged"
                        review_note = "No ground-truth judgment yet; review before treating this result as relevant or non-relevant."
                    detail_rows.append({
                        "Query ID": query_id,
                        "Rank": rank,
                        "Item ID": item["item_id"],
                        "Judgment": judgment,
                        "Item name": item["item_name"],
                        "Category": item["category"],
                        "Location": item["location"],
                        "Description": item["description"],
                        "Review note": review_note,
                    })

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            ambiguous_count = int((ground_truth["judgment"] == "ambiguous").sum())
            st.caption(
                f"{len(evaluation_queries)} queries · {len(ground_truth)} judged query-record pairs · "
                f"{ambiguous_count} ambiguous labels flagged for human review. "
                "A * marks a Precision@K range that includes ambiguous or unjudged results."
            )
            for _, test_case in evaluation_queries.iterrows():
                query_id = test_case["query_id"]
                query_details = pd.DataFrame(detail_rows)
                with st.expander(f"{query_id}: {test_case['query']}"):
                    st.caption(
                        f"Category: {test_case['category']} · Location: {test_case['location']} · "
                        f"Topic: {test_case['topic']}"
                    )
                    st.dataframe(
                        query_details[query_details["Query ID"] == query_id].drop(columns="Query ID"),
                        use_container_width=True,
                        hide_index=True,
                    )

with tab_browse:
    st.subheader("All reports")
    if items.empty:
        st.info("There are no reports yet.")
    else:
        st.dataframe(items, use_container_width=True, hide_index=True)
