from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from extractor import extract_text, extract_fields_from_text, load_model
from normalizer import normalize

# load model once when API starts
load_model("fine_tuned_contract_qa")

app = FastAPI(title="Rental Agreement Metadata Extractor")


@app.get("/")
def home():
    return {"message": "running"}


@app.post("/extract")
async def extract_metadata(file: UploadFile = File(...)):
    allowed_extensions = [".docx", ".png"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type. Use {allowed_extensions}"}
        )

    temp_path = f"/tmp/{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = extract_text(temp_path)
        fields = extract_fields_from_text(extracted_text)
        fields = normalize(fields)
        fields["filename"] = file.filename

        return fields

    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={"error": str(error)}
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
