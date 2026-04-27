import pandas as pd
import re
from pathlib import Path
from pyarabic import araby
from transformers import AutoTokenizer

"""
COMBINED DATASET CREATOR FOR ARABIC-FRENCH TRANSLATION
=======================================================
This script creates a DOUBLED dataset by processing the raw data in TWO ways:
1. Arabic WITH tashkeel (diacritics preserved)
2. Arabic WITHOUT tashkeel (diacritics removed)

Then combines both versions to create a larger training dataset.
Target language: French (with accent preservation)
"""

# =============================================================================
# PATTERNS AND CONSTANTS
# =============================================================================
SPECIAL_SYMBOLS_PATTERN = re.compile(r'[ﷻﷺ§•﴿﴾]')
HTTP_PATTERN = re.compile(r'https?\s*:?\s*(?:/{1,3})?\S+', re.IGNORECASE)
MAX_TOKEN_LENGTH = 1024  # Maximum token length for MADLAD model


# =============================================================================
# ARABIC PREPROCESSING FUNCTIONS
# =============================================================================
def preprocess_arabic_with_tashkeel(text: str) -> str:
    """Processes Arabic text while KEEPING tashkeel (diacritics)."""
    if not isinstance(text, str):
        return ''
    
    # Remove the specific numbering style, e.g., ([1057])
    text = re.sub(r'\(\[\d+\]\)', '', text)
    
    # Remove content within both parentheses and square brackets
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Keep tashkeel - only basic cleaning
    processed_text = text
    
    # Convert Arabic numerals to Western numerals
    numeral_mapping = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    translation_table = str.maketrans(numeral_mapping)
    processed_text = processed_text.translate(translation_table)
    
    # Remove special Islamic symbols
    processed_text = SPECIAL_SYMBOLS_PATTERN.sub('', processed_text)
    
    # Keep periods, commas, Arabic characters, numbers, and question marks
    processed_text = re.sub(r'[^\u0600-\u06FF0-9\s.,؟?]', '', processed_text)
    
    # Normalize whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text


def preprocess_arabic_no_tashkeel(text: str) -> str:
    """Processes Arabic text while REMOVING tashkeel (diacritics)."""
    if not isinstance(text, str):
        return ''
    
    # Remove the specific numbering style, e.g., ([1057])
    text = re.sub(r'\(\[\d+\]\)', '', text)
    
    # Remove content within both parentheses and square brackets
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove tashkeel and normalize Arabic text
    processed_text = araby.strip_diacritics(text)
    processed_text = araby.strip_tatweel(processed_text)
    processed_text = araby.normalize_alef(processed_text)
    processed_text = araby.normalize_teh(processed_text)
    processed_text = re.sub(r'ى', 'ي', processed_text)
    
    # Convert Arabic numerals to Western numerals
    numeral_mapping = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    translation_table = str.maketrans(numeral_mapping)
    processed_text = processed_text.translate(translation_table)
    
    # Remove special Islamic symbols
    processed_text = SPECIAL_SYMBOLS_PATTERN.sub('', processed_text)
    
    # Keep periods, commas, Arabic characters, numbers, and question marks
    processed_text = re.sub(r'[^\u0600-\u06FF0-9\s.,؟?]', '', processed_text)
    
    # Normalize whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text


# =============================================================================
# TARGET TEXT CLEANING FUNCTIONS
# =============================================================================
def remove_arabic_chars(text: str) -> str:
    """Removes Arabic characters from a string."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[\u0600-\u06FF]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_target_text(text: str) -> str:
    """A cleaning pipeline for the target text (preserving French accents and special characters)."""
    if not isinstance(text, str):
        return ''
    
    # Remove the specific numbering style, e.g., ([1057])
    text = re.sub(r'\(\[\d+\]\)', '', text)
    
    # Remove content within both parentheses and square brackets
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Keep French characters (including accented ones), numbers, spaces, periods, commas, and basic punctuation
    # Unicode ranges: Latin Basic (0041-007A), Latin-1 Supplement (00C0-00FF) for accents
    text = re.sub(r'[^\u0041-\u007A\u0061-\u007A\u00C0-\u00FF0-9\s.,;:!?\'\"-]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def is_within_token_limit(text: str, tokenizer, max_length: int = MAX_TOKEN_LENGTH) -> bool:
    """Check if text is within the token limit."""
    if not isinstance(text, str) or not text.strip():
        return True  # Empty text is fine, will be filtered elsewhere
    
    try:
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)
        token_count = len(tokens['input_ids'])
        return token_count <= max_length
    except Exception:
        # If tokenization fails, assume it's too long
        return False


def has_http_link(text: str) -> bool:
    """Check if text contains HTTP/HTTPS links."""
    return isinstance(text, str) and bool(HTTP_PATTERN.search(text))


def has_enough_letters(text: str, language: str = 'en', min_count: int = 3) -> bool:
    """Check if the text contains a minimum number of letters."""
    if not isinstance(text, str):
        return False
    
    if language == 'ar':
        pattern = r'[\u0600-\u06FF]'
    elif language == 'fr':
        # French letters including accented characters
        pattern = r'[a-zA-Z\u00C0-\u00FF]'
    else:  # default to English
        pattern = r'[a-zA-Z]'
    
    return len(re.findall(pattern, text)) >= min_count


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def create_combined_dataset():
    """
    Main function to create a combined dataset from raw data.
    Processes data twice (with and without tashkeel) and combines them.
    """
    print("=" * 70)
    print("COMBINED DATASET CREATOR FOR ARABIC-FRENCH TRANSLATION")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load raw data from Excel")
    print("  2. Load MADLAD tokenizer for token length validation")
    print("  3. Create VERSION A: Arabic WITH tashkeel (diacritics)")
    print("  4. Create VERSION B: Arabic WITHOUT tashkeel (no diacritics)")
    print("  5. Filter out entries exceeding 1024 tokens")
    print("  6. Combine both versions to DOUBLE the dataset size")
    print("  7. Save the combined dataset")
    print("=" * 70)

    # --- File Paths ---
    input_path = Path("data_french.xlsx")
    output_path = Path("combined_dataset.xlsx")
    
    # --- Model Configuration ---
    MODEL_CHECKPOINT = "google/madlad400-10b-mt"

    try:
        # =====================================================================
        # STEP 1: LOAD RAW DATA
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 1: LOADING RAW DATA")
        print(f"{'='*70}")
        print(f"Reading from: {input_path}")
        
        df = pd.read_excel(input_path)
        original_rows = len(df)
        print(f"✓ Loaded {original_rows} rows with {df.shape[1]} columns")

        # Validate required columns
        expected_cols = ['source_text', 'target_text']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Input file must contain 'source_text' and 'target_text'. Found: {list(df.columns)}")
        
        # Assign original_id to track duplicates across versions
        df['original_id'] = range(len(df))
        print(f"✓ Assigned original_id to track duplicates across versions")

        # =====================================================================
        # STEP 2: LOAD MADLAD TOKENIZER
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 2: LOADING MADLAD TOKENIZER")
        print(f"{'='*70}")
        print(f"  > Loading tokenizer from: {MODEL_CHECKPOINT}")
        print(f"  > This will be used to filter entries exceeding {MAX_TOKEN_LENGTH} tokens")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        print(f"✓ Tokenizer loaded successfully")

        # =====================================================================
        # STEP 3: CREATE VERSION A (WITH TASHKEEL)
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 3: PROCESSING VERSION A - WITH TASHKEEL")
        print(f"{'='*70}")
        
        df_with_tashkeel = df.copy()
        print("  > Applying Arabic preprocessing (keeping diacritics)...")
        # Keep original_id through processing
        df_with_tashkeel['source_cleaned'] = df_with_tashkeel['source_text'].apply(preprocess_arabic_with_tashkeel)
        df_with_tashkeel['target_cleaned'] = df_with_tashkeel['target_text'].apply(clean_target_text)
        
        print("  > Filtering data...")
        # Filter out rows with HTTP links
        http_mask = df_with_tashkeel.apply(lambda row: has_http_link(row['source_text']) or has_http_link(row['target_text']), axis=1)
        df_with_tashkeel = df_with_tashkeel[~http_mask]
        removed_http = len(df) - len(df_with_tashkeel)
        print(f"    - Removed {removed_http} rows with HTTP links")
        
        # Filter out rows with insufficient text
        short_mask = df_with_tashkeel.apply(lambda row: not has_enough_letters(row['source_cleaned'], language='ar') or not has_enough_letters(row['target_cleaned'], language='fr'), axis=1)
        df_with_tashkeel = df_with_tashkeel[~short_mask]
        removed_short = len(df) - removed_http - len(df_with_tashkeel)
        print(f"    - Removed {removed_short} rows with insufficient text length")
        
        # Filter out empty rows
        empty_mask = (df_with_tashkeel['source_cleaned'] == '') | (df_with_tashkeel['target_cleaned'] == '')
        df_with_tashkeel = df_with_tashkeel[~empty_mask]
        removed_empty = original_rows - removed_http - removed_short - len(df_with_tashkeel)
        print(f"    - Removed {removed_empty} empty rows after cleaning")
        
        # Filter out rows exceeding token limit
        print(f"  > Filtering entries exceeding {MAX_TOKEN_LENGTH} tokens...")
        before_token_filter = len(df_with_tashkeel)
        token_mask = df_with_tashkeel.apply(
            lambda row: is_within_token_limit(row['source_cleaned'], tokenizer) and 
                       is_within_token_limit(row['target_cleaned'], tokenizer),
            axis=1
        )
        df_with_tashkeel = df_with_tashkeel[token_mask]
        removed_long = before_token_filter - len(df_with_tashkeel)
        print(f"    - Removed {removed_long} rows exceeding token limit")
        
        # Add version tracking (original_id already exists from raw data)
        df_with_tashkeel['version'] = 'with_tashkeel'
        
        # Keep only cleaned columns plus tracking info
        df_with_tashkeel = df_with_tashkeel[['source_cleaned', 'target_cleaned', 'original_id', 'version']].copy()
        df_with_tashkeel.columns = ['source_text', 'target_text', 'original_id', 'version']
        
        print(f"✓ VERSION A completed: {len(df_with_tashkeel)} rows")

        # =====================================================================
        # STEP 4: CREATE VERSION B (WITHOUT TASHKEEL)
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 4: PROCESSING VERSION B - WITHOUT TASHKEEL")
        print(f"{'='*70}")
        
        df_no_tashkeel = df.copy()
        print("  > Applying Arabic preprocessing (removing diacritics)...")
        # Keep original_id through processing
        df_no_tashkeel['source_cleaned'] = df_no_tashkeel['source_text'].apply(preprocess_arabic_no_tashkeel)
        df_no_tashkeel['target_cleaned'] = df_no_tashkeel['target_text'].apply(clean_target_text)
        
        print("  > Filtering data...")
        # Filter out rows with HTTP links
        http_mask = df_no_tashkeel.apply(lambda row: has_http_link(row['source_text']) or has_http_link(row['target_text']), axis=1)
        df_no_tashkeel = df_no_tashkeel[~http_mask]
        removed_http = len(df) - len(df_no_tashkeel)
        print(f"    - Removed {removed_http} rows with HTTP links")
        
        # Filter out rows with insufficient text
        short_mask = df_no_tashkeel.apply(lambda row: not has_enough_letters(row['source_cleaned'], language='ar') or not has_enough_letters(row['target_cleaned'], language='fr'), axis=1)
        df_no_tashkeel = df_no_tashkeel[~short_mask]
        removed_short = len(df) - removed_http - len(df_no_tashkeel)
        print(f"    - Removed {removed_short} rows with insufficient text length")
        
        # Filter out empty rows
        empty_mask = (df_no_tashkeel['source_cleaned'] == '') | (df_no_tashkeel['target_cleaned'] == '')
        df_no_tashkeel = df_no_tashkeel[~empty_mask]
        removed_empty = original_rows - removed_http - removed_short - len(df_no_tashkeel)
        print(f"    - Removed {removed_empty} empty rows after cleaning")
        
        # Filter out rows exceeding token limit
        print(f"  > Filtering entries exceeding {MAX_TOKEN_LENGTH} tokens...")
        before_token_filter = len(df_no_tashkeel)
        token_mask = df_no_tashkeel.apply(
            lambda row: is_within_token_limit(row['source_cleaned'], tokenizer) and 
                       is_within_token_limit(row['target_cleaned'], tokenizer),
            axis=1
        )
        df_no_tashkeel = df_no_tashkeel[token_mask]
        removed_long = before_token_filter - len(df_no_tashkeel)
        print(f"    - Removed {removed_long} rows exceeding token limit")
        
        # Add version tracking (original_id already exists from raw data)
        df_no_tashkeel['version'] = 'without_tashkeel'
        
        # Keep only cleaned columns plus tracking info
        df_no_tashkeel = df_no_tashkeel[['source_cleaned', 'target_cleaned', 'original_id', 'version']].copy()
        df_no_tashkeel.columns = ['source_text', 'target_text', 'original_id', 'version']
        
        print(f"✓ VERSION B completed: {len(df_no_tashkeel)} rows")

        # =====================================================================
        # STEP 5: COMBINE BOTH VERSIONS
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 5: COMBINING BOTH VERSIONS")
        print(f"{'='*70}")
        
        df_combined = pd.concat([df_with_tashkeel, df_no_tashkeel], ignore_index=True)
        print(f"  > Concatenated datasets: {len(df_combined)} total rows")
        
        # Verify duplicate tracking
        unique_ids = df_combined['original_id'].nunique()
        print(f"  > Unique original IDs: {unique_ids}")
        print(f"  > Each ID should appear twice (once per version)")
        
        # Shuffle for better training distribution
        print("  > Shuffling combined dataset...")
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Combined dataset created successfully!")

        # =====================================================================
        # STEP 6: SAVE COMBINED DATASET
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 6: SAVING COMBINED DATASET")
        print(f"{'='*70}")
        
        df_combined.to_excel(output_path, index=False)
        print(f"✓ Saved to: {output_path}")

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print(f"\n{'='*70}")
        print("✅ DATASET CREATION COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📊 SUMMARY:")
        print(f"  Original raw data:           {original_rows:,} rows")
        print(f"  VERSION A (with tashkeel):   {len(df_with_tashkeel):,} rows")
        print(f"  VERSION B (without tashkeel): {len(df_no_tashkeel):,} rows")
        print(f"  ─────────────────────────────────────────")
        print(f"  TOTAL COMBINED:              {len(df_combined):,} rows")
        print(f"  Dataset size multiplier:     {len(df_combined) / max(len(df_with_tashkeel), len(df_no_tashkeel)):.2f}x")
        print(f"\n📁 Output file: {output_path}")
        print(f"{'='*70}\n")

    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found at '{input_path}'")
        print("Please ensure the raw data file exists.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_combined_dataset()
