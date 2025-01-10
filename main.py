import os
import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Azure credentials
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise ValueError("Azure Form Recognizer credentials not found in environment variables")

# Initialize Azure Form Recognizer client
client = DocumentAnalysisClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY)
)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Process a document using Azure Form Recognizer and extract specific fields.
    """
    try:
        # Check file type
        allowed_types = {'pdf', 'jpeg', 'jpg', 'png', 'tiff'}
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Please upload: {', '.join(allowed_types)}"
            )

        # Read the file
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        # Process with Form Recognizer
        result = client.begin_analyze_document(
            "prebuilt-document", 
            file_stream
        ).result()

        # Extract required fields
        required_fields = {
            "I.D. No.", 
            "Employee Name", 
            "Date Filed", 
            "Reason For Leave:"
        }
        
        extracted_data = []
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.key.content in required_fields:
                # Get key and value information
                key_content = kv_pair.key.content
                value_content = kv_pair.value.content if kv_pair.value else None
                
                # Get bounding box coordinates if available
                key_coords = (
                    kv_pair.key.bounding_regions[0].polygon 
                    if kv_pair.key.bounding_regions 
                    else None
                )
                value_coords = (
                    kv_pair.value.bounding_regions[0].polygon 
                    if kv_pair.value and kv_pair.value.bounding_regions 
                    else None
                )
                
                extracted_data.append({
                    "key": {
                        "content": key_content,
                        "coordinates": key_coords
                    },
                    "value": {
                        "content": value_content,
                        "coordinates": value_coords
                    },
                    "confidence": kv_pair.confidence
                })

        return {"extracted_data": extracted_data}

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)