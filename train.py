import pandas as pd
import os
import random
import dateparser
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from extractor import extract_text
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
train_folder = "data/train"
train_csv_path = "data/train.csv"

ground_truth = pd.read_csv(train_csv_path)

# These are the questions used both during training-label construction and at
# inference time. Keep them aligned with extractor.py.
questions_per_field = {
    "Aggrement Value":       "What is the monthly rent amount?",
    "Aggrement Start Date":  "When does the rental agreement start?",
    "Aggrement End Date":    "When does the rental agreement end or expire?",
    "Renewal Notice (Days)": "How many days of notice are required to terminate or renew the agreement?",
    "Party One":             "Who is the owner or lessor of the property?",
    "Party Two":             "Who is the tenant or lessee?",
}

---------------------------------------------------------------------------


def find_date_in_text(document_text, ground_truth_date):
    ground_truth_parsed = dateparser.parse(
        ground_truth_date, settings={"DATE_ORDER": "DMY"})
    if ground_truth_parsed is None:
        return -1, None
    words = document_text.split()
    # IMPORTANT: try single-word windows first : a date written as "20.05.2007"
    # is a single token. The original code only tried windows of 4-6, missing
    # most compact dates and starving the model of date training signal.
    for window_size in [1, 2, 3, 4, 5, 6]:
        for start_index in range(len(words) - window_size + 1):
            candidate_span = " ".join(
                words[start_index: start_index + window_size])
            candidate_parsed = dateparser.parse(
                candidate_span, settings={"DATE_ORDER": "DMY"})
            if candidate_parsed and candidate_parsed.date() == ground_truth_parsed.date():
                position = document_text.find(candidate_span)
                if position != -1:
                    return position, candidate_span
    return -1, None


def find_number_in_text(document_text, ground_truth_value):
    clean_value = str(ground_truth_value).replace(
        ",", "").replace(".", "").strip()
    if clean_value == "" or clean_value.lower() == "nan":
        return -1, None
    for word in document_text.split():
        clean_word = (word.replace(",", "")
                          .replace(".", "")
                          .replace("/-", "")
                          .replace("/=", "")
                          .replace("Rs", "")
                          .replace("Php", "")
                          .strip())
        if clean_word == clean_value:
            position = document_text.find(word)
            return position, word
    return -1, None


def find_answer_in_text(document_text, answer, field_type):
    if field_type in ["Aggrement Start Date", "Aggrement End Date"]:
        return find_date_in_text(document_text, answer)
    if field_type == "Aggrement Value":
        return find_number_in_text(document_text, answer)
    if field_type == "Renewal Notice (Days)":
        clean_days = str(answer).replace(".0", "").strip()
        if clean_days == "" or clean_days.lower() == "nan":
            return -1, None
        return find_number_in_text(document_text, clean_days)
    # Party names: try exact, then case-insensitive substring
    clean_answer = str(answer).strip()
    if clean_answer == "" or clean_answer.lower() == "nan":
        return -1, None
    position = document_text.find(clean_answer)
    if position != -1:
        return position, clean_answer
    lowered_position = document_text.lower().find(clean_answer.lower())
    if lowered_position != -1:
        return lowered_position, document_text[lowered_position: lowered_position + len(clean_answer)]
    return -1, None


def prepare_training_samples():
    training_samples = []
    for _, row in ground_truth.iterrows():
        filename = row["File Name"]
        actual_file = None
        for file in os.listdir(train_folder):
            if file.startswith(filename):
                actual_file = os.path.join(train_folder, file)
                break
        if actual_file is None:
            print(f"File not found: {filename}")
            continue
        print(f"Building samples for {filename}...")
        document_text = extract_text(actual_file)
        for csv_field, question in questions_per_field.items():
            answer_value = str(row[csv_field]).strip()
            if answer_value == "" or answer_value.lower() == "nan":
                continue
            answer_position, actual_span = find_answer_in_text(
                document_text, answer_value, csv_field)
            if answer_position == -1:
                print(f"  NOT FOUND: {csv_field} = '{answer_value}'")
                continue
            print(f"  Found: {csv_field} = '{actual_span}'")
            training_samples.append({
                "question":     question,
                "context":      document_text,
                "answer":       actual_span,
                "answer_start": answer_position,
            })
    print(f"\nTotal training samples: {len(training_samples)}")
    return training_samples


# Convert (question, context, answer_char_span) into one or more model
# features using a sliding window. Windows that don't contain the answer
# get (start=0, end=0) --> CLS = "no answer". Windows that DO contain the
# answer get the precise token span. This is the standard SQuAD-style
# feature builder and is the main fix vs. the previous code.

def prepare_features(samples, tokenizer, max_length=384, stride=128):
    features = []
    for sample in samples:
        tokenized = tokenizer(
            sample["question"],
            sample["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        answer_start_char = sample["answer_start"]
        answer_end_char = answer_start_char + len(sample["answer"])

        for window_index in range(tokenized["input_ids"].shape[0]):
            offset_mapping = tokenized["offset_mapping"][window_index].tolist()
            sequence_ids = tokenized.sequence_ids(window_index)

            # Locate the context (sequence_id == 1) inside this window.
            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1

            answer_in_window = (
                context_start <= context_end
                and offset_mapping[context_start][0] <= answer_start_char
                and offset_mapping[context_end][1] >= answer_end_char
            )

            if not answer_in_window:
                start_position = 0  # CLS — "no answer" for this window
                end_position = 0
            else:
                idx = context_start
                while idx <= context_end and offset_mapping[idx][0] <= answer_start_char:
                    idx += 1
                start_position = idx - 1

                idx = context_end
                while idx >= context_start and offset_mapping[idx][1] >= answer_end_char:
                    idx -= 1
                end_position = idx + 1

            features.append({
                "input_ids":      tokenized["input_ids"][window_index],
                "attention_mask": tokenized["attention_mask"][window_index],
                "start_position": start_position,
                "end_position":   end_position,
            })
    return features


def fine_tune(training_samples, num_epochs=3, batch_size=4, learning_rate=3e-5):
    model_name = "Rakib/roberta-base-on-cuad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    features = prepare_features(training_samples, tokenizer)
    print(
        f"Created {len(features)} training features from {len(training_samples)} samples")

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        random.shuffle(features)
        total_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(features), batch_size):
            batch = features[batch_start: batch_start + batch_size]
            input_ids = torch.stack([f["input_ids"] for f in batch])
            attention_mask = torch.stack([f["attention_mask"] for f in batch])
            start_positions = torch.tensor(
                [f["start_position"] for f in batch])
            end_positions = torch.tensor([f["end_position"] for f in batch])

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch + 1}/{num_epochs} — avg loss: {total_loss / max(num_batches, 1):.4f}")

    model.save_pretrained("fine_tuned_contract_qa")
    tokenizer.save_pretrained("fine_tuned_contract_qa")
    print("Model saved to fine_tuned_contract_qa/")


# Internal-key → train.csv column. Note: "Aggrement" is misspelled in the
# source CSV; we mirror that exactly so the predictions file matches its
# format.
CSV_COLUMN_FOR_FIELD = {
    "agreement_value":      "Aggrement Value",
    "agreement_start_date": "Aggrement Start Date",
    "agreement_end_date":   "Aggrement End Date",
    "renewal_notice_days":  "Renewal Notice (Days)",
    "party_one":            "Party One",
    "party_two":            "Party Two",
}
OUTPUT_COLUMNS = ["File Name"] + list(CSV_COLUMN_FOR_FIELD.values())


def generate_predictions(data_folder=train_folder,
                         csv_path=train_csv_path,
                         output_csv="predictions_train.csv",
                         model_path="fine_tuned_contract_qa"):
    from extractor import extract_text, extract_fields_from_text, load_model
    from normalizer import normalize

    load_model(model_path)
    all_results = []
    dataframe = pd.read_csv(csv_path)

    for _, row in dataframe.iterrows():
        filename = row["File Name"]
        actual_file_path = None
        for file in os.listdir(data_folder):
            if file.startswith(filename):
                actual_file_path = os.path.join(data_folder, file)
                break

        record = {col: "" for col in OUTPUT_COLUMNS}
        record["File Name"] = filename

        if actual_file_path is None:
            print(f"File not found: {filename}")
            all_results.append(record)
            continue

        print(f"Processing {filename}...")
        try:
            extracted_text = extract_text(actual_file_path)
            fields = normalize(extract_fields_from_text(extracted_text))
            for internal_key, csv_column in CSV_COLUMN_FOR_FIELD.items():
                record[csv_column] = fields.get(internal_key, "")
        except Exception as error:
            print(f"  Failed: {error}")

        all_results.append(record)

    pd.DataFrame(all_results, columns=OUTPUT_COLUMNS).to_csv(
        output_csv, index=False)
    print(f"Saved → {output_csv}")


def calculate_recall(predictions_csv="predictions_train.csv", csv_path=train_csv_path):
    predictions_df = pd.read_csv(predictions_csv)
    ground_truth_df = pd.read_csv(csv_path)
    merged = pd.merge(ground_truth_df, predictions_df,
                      on="File Name", suffixes=("_true", "_pred"))

    fields = list(CSV_COLUMN_FOR_FIELD.values())
    print(f"\nRecall on {len(merged)} documents\n" + "-" * 50)
    for field in fields:
        true_col, pred_col = f"{field}_true", f"{field}_pred"
        if true_col not in merged.columns or pred_col not in merged.columns:
            print(f"{field}: columns missing")
            continue
        true_values = merged[true_col].astype(str).str.strip().str.lower()
        predicted_values = merged[pred_col].astype(str).str.strip().str.lower()
        correct_matches = (true_values == predicted_values).sum()
        total_documents = len(merged)
        print(
            f"{field:25s}: {correct_matches}/{total_documents} = {correct_matches/total_documents:.2f}")


if __name__ == "__main__":
    training_samples = prepare_training_samples()
    if len(training_samples) > 0:
        fine_tune(training_samples)
        generate_predictions(
            data_folder=train_folder,
            csv_path=train_csv_path,
            output_csv="predictions_train.csv",
        )
        calculate_recall()
        # Also produce predictions for the held-out test set
        if os.path.exists("data/test.csv"):
            generate_predictions(
                data_folder="data/test",
                csv_path="data/test.csv",
                output_csv="predictions_test.csv",
            )
    else:
        print("No training samples found. Check your data folder.")
