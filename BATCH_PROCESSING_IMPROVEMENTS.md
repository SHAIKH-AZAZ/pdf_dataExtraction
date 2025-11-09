# Batch Processing Improvements

## Key Changes Made to CV_data_extraction.py

### 1. **Text Cleaning & Normalization**
- Added `clean_text()` function to remove noise and excessive whitespace
- Limits text length to 4000 characters to prevent overwhelming the model
- Removes special characters that might confuse the AI

### 2. **Improved Prompt Structure**
- Clear instructions that this is ONE resume at a time
- Explicit field names with descriptions
- Lower temperature (0.1) for more consistent output
- Added system message emphasizing single-resume processing

### 3. **Rate Limiting**
- 2-second delay between API calls (configurable)
- Prevents API overload and gives model time to process each resume separately
- Clear progress indicators showing wait times

### 4. **Retry Logic**
- Automatic retry up to 3 times on API failures
- Handles network errors, timeouts, and JSON parsing errors
- Exponential backoff on retries

### 5. **Better Error Handling**
- Try-catch blocks for each resume
- One failed resume won't stop the entire batch
- Returns empty fields for failed extractions
- Detailed error messages for debugging

### 6. **Progress Tracking**
- Shows current file being processed (e.g., [3/8])
- Visual separators between resumes
- Summary at the end with total processed

### 7. **Validation**
- Checks if extracted text is meaningful (>50 characters)
- Validates JSON structure before saving
- Ensures all expected fields are present

## Configuration Options

You can adjust these constants at the top of the file:

```python
MAX_TEXT_LENGTH = 4000  # Maximum text length per resume
RATE_LIMIT_DELAY = 2    # Seconds between API calls
MAX_RETRIES = 3         # Number of retry attempts
```

## How It Prevents Model Confusion

1. **One at a time**: Each resume is processed completely before moving to the next
2. **Clean input**: Removes noise that could confuse the model
3. **Clear context**: Prompt explicitly states "this is ONE resume"
4. **Rate limiting**: Gives the model time to "reset" between requests
5. **Length limits**: Prevents overwhelming the model with too much text
6. **Structured output**: Forces consistent JSON format

## Usage

### Process All Resumes (One at a Time)
```bash
python CV_data_extraction.py
```
This processes all PDFs in the `resumes/` folder **sequentially** - each resume is completely processed before the next one starts.

### Process Single Resume (For Testing)
```bash
python CV_data_extraction.py path/to/resume.pdf
```
This processes just one resume and saves to `single_resume_data.xlsx`.

## How "One at a Time" Works

The script ensures strict sequential processing:

1. **Resume 1**: Extract text → Clean → Send to API → Wait for response → Save
2. **Wait 2 seconds** (rate limit delay)
3. **Resume 2**: Extract text → Clean → Send to API → Wait for response → Save
4. **Wait 2 seconds**
5. **Resume 3**: And so on...

Each resume is **completely finished** before the next one begins. The model never sees data from multiple resumes at once.
