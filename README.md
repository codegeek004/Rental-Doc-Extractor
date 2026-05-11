# Rental Document Extractor

A FastAPI service that extracts structured metadata from rental agreements (DOCX and scanned PNG) using a fine-tuned RoBERTa question-answering model.

## Features

- **Question-answering extraction**: Each field is a question against the document, no regex, no rule-based parsing
- **CUAD-pretrained backbone**: Starts from a model already trained on legal-contract QA, then fine-tuned on rental agreements
- **Sliding-window training and inference**: Long documents are handled without truncating the answer span
- **DOCX + scanned PNG support**: python-docx for native Word docs, Tesseract OCR for scanned images
- **Post-processing with parsers, not regex**: dateparser for dates, word2number for written numbers
- **Modular**: Training, extraction, and serving are fully decoupled

## Tech Stack

- **Web Framework**: FastAPI + Uvicorn
- **ML**: PyTorch, Hugging Face Transformers
- **Base model**: `Rakib/roberta-base-on-cuad` (CUAD-fine-tuned RoBERTa)
- **DOCX parsing**: python-docx
- **OCR**: pytesseract + Pillow
- **Post-processing**: dateparser, word2number

## Deployment

This service is deployed on Google Cloud Platform and is publicly accessible for testing:

```
http://34.44.150.96/extract
```

Test it directly with curl:

```bash
curl -X POST "http://34.44.150.96/extract" \
  -F "file=@your-agreement.docx"
```

Or from Python:

```python
import requests

with open("agreement.docx", "rb") as f:
    r = requests.post(
        "http://34.44.150.96/extract",
        files={"file": f}
    )

print(r.json())
```

## Per-Field Recall

Recall on the 10-document training set, comparing normalized predictions against the ground-truth values in `train.csv`:

| Field                   | Recall           |
|-------------------------|------------------|
| Aggrement Value         | 4/10 = 0.40      |
| Aggrement Start Date    | 3/10 = 0.30      |
| Aggrement End Date      | 1/10 = 0.10      |
| Renewal Notice (Days)   | 0/10 = 0.00      |
| Party One               | 4/10 = 0.40      |
| Party Two               | 5/10 = 0.50      |
| **Average**             | **17/60 = 0.28** |

### Why the scores are what they are

- **Tiny training set.** Nine documents yield about 18 usable training spans after label alignment. There's not enough signal for the model to robustly learn rental-agreement layout patterns, it's leaning heavily on what CUAD pretraining already taught it.
- **OCR on PNG files is lossy.** Four of the ten documents are scanned images. Tesseract garbles rent values and dates on some of them ("9,500" becomes "9.99.7.9°"), so the ground truth doesn't appear in the extracted text and the model has nothing to match.
- **Renewal Notice is the worst field.** It's often written in words ("ninety days") and sometimes absent. The normalizer's `word2number` fallback handles "ninety" only if the QA model returns that exact span, which it rarely does.
- **End-date matching is hurt by data issues.** One row in `train.csv` has `31.02.2011`, which isn't a real date, so `dateparser` rejects it under strict parsing and the document is unmatchable.

## Getting Started

### Prerequisites

- Python 3.10+
- Tesseract OCR for scanned PNGs: `sudo apt install tesseract-ocr`

### Installation

##### Clone the repository
```bash
git clone git@github.com:codegeek004/Rental-Doc-Extractor.git
cd Rental-Document-Extractor
```

##### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

##### Install dependencies
```bash
pip install -r requirements.txt
```

##### Train the model
The fine-tuned weights aren't committed (around 500 MB). Run training once before serving:
```bash
python train.py
```
This builds training samples, fine-tunes `Rakib/roberta-base-on-cuad` for 3 epochs, saves the result to `fine_tuned_contract_qa/`, and writes `predictions_train.csv` and `predictions_test.csv`.

##### Start the server
```bash
python main.py
```

Server runs at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

## Architecture & System Design

### Project Structure

```
project/
├── train.py            --> Fine-tuning, prediction, recall
├── extractor.py        --> Text extraction + QA inference
├── normalizer.py       --> Date / number / name post-processing
├── main.py             --> FastAPI service
├── data/
│   ├── train.csv       --> Ground-truth labels
│   ├── test.csv
│   ├── train/          --> DOCX + PNG agreements
│   └── test/
├── fine_tuned_contract_qa/   --> Saved model (created by train.py)
└── requirements.txt
```

`main.py` doesn't know about training or model architecture. It calls `extractor.extract_fields_from_text()` and `normalizer.normalize()`, and that's it. `train.py` and `extractor.py` can run completely independently of the server.

### How it works (inference path)

1. Client sends `POST /extract` with a DOCX or PNG file
2. FastAPI saves the upload to `/tmp` and dispatches by extension
3. For DOCX, `python-docx` walks paragraphs and table cells into a single text string
4. For PNG, `pytesseract` runs OCR on the image
5. The text is passed to the QA model along with six pre-written questions, one per field
6. For each question, the Hugging Face QA pipeline runs a sliding window over the full text (`max_seq_len=384`, `doc_stride=128`) and returns the highest-scoring span across all windows
7. Raw answers are post-processed: dates via `dateparser`, numeric amounts by isolating the longest digit run after dropping thousand-separator commas, days using `word2number`, party names by stripping leading boilerplate (`between`, `lessor`, and so on)
8. The cleaned record is returned as JSON

### How it works (training path)

1. For each `(document, field)` pair in `train.csv`, locate the ground-truth value's character span inside the extracted text:
   - **Dates**: parse with `dateparser` using `STRICT_PARSING=True`, match windows of 1 to 6 words
   - **Numbers**: strip commas, periods, `Rs`, `Php`, `/-`, then exact-match
   - **Party names**: substring search with a case-insensitive fallback
2. Convert each `(question, context, char_span)` triple into one or more SQuAD-style features using a 128-token sliding window. Windows that don't contain the answer get `(start=0, end=0)` = CLS = "no answer".
3. Fine-tune for 3 epochs with `AdamW`, `lr=3e-5`, `batch_size=4`
4. Save the model to `fine_tuned_contract_qa/`
5. Re-run inference on the training set, write `predictions_train.csv`, compute per-field recall

### Field routing

Each of the six fields has a dedicated question. The questions are run independently against the same context, so the model has no shared state between fields.

```
Document text  -->  "What is the monthly rent amount?"             -->  agreement_value
              -->  "When does the rental agreement start?"        -->  agreement_start_date
              -->  "When does the rental agreement end?"          -->  agreement_end_date
              -->  "How many days of notice are required..."      -->  renewal_notice_days
              -->  "Who is the owner or lessor of the property?"  -->  party_one
              -->  "Who is the tenant or lessee?"                 -->  party_two
```

### Why fine-tune at all?

The base `Rakib/roberta-base-on-cuad` is already trained on CUAD, a legal-contract QA dataset, so it has a reasonable prior for questions like "Who is the lessor?". Fine-tuning on rental agreements teaches it the specific phrasing and layout used in this dataset: Indian-format dates, `Rs.` currency prefixes, parties named with honorific prefixes like "MR.K.Kuttan", and so on.

### Why no regex?

All actual field extraction is done by the QA model. The only string-handling in the codebase is:

- Locating ground-truth spans during *training* (so the model has supervision targets, this isn't extraction)
- Cleaning the model's output during post-processing (parsing dates with a library, dropping currency symbols)

No hand-written pattern identifies "the rent value" or "the start date", that's the model's job.

### Why sliding windows?

Most rental agreements tokenize to more than 512 tokens. Without a sliding window, anything past the truncation point is invisible to both training and inference. A `max_seq_len=384` window with `doc_stride=128` overlap means every span in the document is covered by at least one window, and the pipeline picks the best span across all of them.

## API

### POST /extract

| Field | Type | Required |
|-------|------|----------|
| file | DOCX or PNG | Yes |

**Response**

```json
{
  "agreement_value": "8000",
  "agreement_start_date": "01.04.2011",
  "agreement_end_date": "31.03.2012",
  "renewal_notice_days": "90",
  "party_one": "K. Parthasarathy",
  "party_two": "Veerabrahmam Bathini",
  "filename": "54770958-Rental-Agreement.png"
}
```

Date fields are returned as `DD.MM.YYYY`. Numeric fields are returned as plain digit strings. Party names are returned with leading boilerplate (such as "lessor", "between") stripped.

## Usage

### Local

1. Start the server with `python main.py`
2. Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI
3. Upload your DOCX or PNG and hit Execute

```bash
# curl
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "file=@data/test/some-agreement.docx"
```

```python
import requests

with open("agreement.docx", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/extract",
        files={"file": f}
    )

print(r.json())
```

## Reproducing the recall metric

```bash
python train.py
```

The script ends by printing the per-field recall table shown at the top of this README. Predictions are written to `predictions_train.csv` (and `predictions_test.csv` if `data/test.csv` is present) so you can inspect the model's output document by document.

## Acknowledgements

- [Rakib/roberta-base-on-cuad](https://huggingface.co/Rakib/roberta-base-on-cuad): The CUAD-fine-tuned QA model used as the base
- [CUAD](https://www.atticusprojectai.org/cuad): Contract Understanding Atticus Dataset
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [python-docx](https://github.com/python-openxml/python-docx)
- [Tesseract](https://github.com/tesseract-ocr/tesseract)
- [dateparser](https://github.com/scrapinghub/dateparser)
