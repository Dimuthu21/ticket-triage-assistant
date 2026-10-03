# Support Ticket Triage & Answer Assistant

> Work in progress (Day 1 of 14). Results below are filled in as each day finishes.

## What it does
1. Classifies a support ticket into a queue (fine-tuned DistilBERT vs. TF-IDF baseline)
2. Drafts a grounded answer with RAG over a help-article knowledge base, citing the article used
3. Sends low-confidence cases to a human instead of guessing
4. Served through FastAPI, hosted on Hugging Face Spaces

## Data
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
(synthetic English/German tickets). Only **English** tickets are used. Input text = subject + body only;
the `answer`, `type` and `priority` columns are NOT used as features (they would leak information
that is not available when a new ticket arrives). Exact duplicates are removed before splitting.

## Results so far
| Model | Val macro-F1 | Test macro-F1 | Notes |
|---|---|---|---|
| Majority class | _fill from results/day1_metrics.json_ | | floor |
| TF-IDF + best of (LogReg, LinearSVC) | | | Day 1 baseline |
| Fine-tuned DistilBERT | | | Day 3 |

## Reproduce Day 1
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/day1_baseline.py
```
Splits use a fixed seed (42), so results are reproducible.

## Limitations (to be extended)
- The dataset is synthetic; real tickets would be messier.
