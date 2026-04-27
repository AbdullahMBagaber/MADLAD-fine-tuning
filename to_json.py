# You might need to run this in your terminal first if you don't have these libraries
# pip install pandas openpyxl

import pandas as pd
from pathlib import Path

"""
EXCEL TO JSONL CONVERTER FOR MADLAD TRAINING
=============================================
Converts both training and test Excel files to JSONL format for model training.
"""

print("=" * 70)
print("CONVERTING EXCEL TO JSONL FORMAT")
print("=" * 70)

# 1. Define your file names
train_excel = Path('training_dataset.xlsx')
test_excel = Path('test_dataset.xlsx')
train_jsonl = Path('translations.jsonl')
test_jsonl = Path('test.jsonl')

# ============================================================================
# PROCESS TRAINING SET
# ============================================================================
print(f"\n📂 Processing training dataset...")
if not train_excel.exists():
    print(f"❌ ERROR: Training file '{train_excel}' not found!")
    print("   Please run extract_test_set.py first.")
else:
    print(f"  > Reading from: {train_excel}")
    df_train = pd.read_excel(train_excel)
    
    # Verify columns exist
    if 'source_text' not in df_train.columns or 'target_text' not in df_train.columns:
        print(f"  ❌ ERROR: Expected columns 'source_text' and 'target_text', found: {list(df_train.columns)}")
    else:
        # Remove tracking columns (original_id, version) - not needed for training
        print(f"  > Removing tracking columns (original_id, version)...")
        df_train_clean = df_train[['source_text', 'target_text']].copy()
        
        # Save to JSONL
        df_train_clean.to_json(train_jsonl, orient='records', lines=True, force_ascii=False)
        print(f"  ✓ Successfully saved {len(df_train_clean):,} rows to: {train_jsonl}")

# ============================================================================
# PROCESS TEST SET
# ============================================================================
print(f"\n📂 Processing test dataset...")
if not test_excel.exists():
    print(f"❌ ERROR: Test file '{test_excel}' not found!")
    print("   Please run extract_test_set.py first.")
else:
    print(f"  > Reading from: {test_excel}")
    df_test = pd.read_excel(test_excel)
    
    # Verify columns exist
    if 'source_text' not in df_test.columns or 'target_text' not in df_test.columns:
        print(f"  ❌ ERROR: Expected columns 'source_text' and 'target_text', found: {list(df_test.columns)}")
    else:
        # Remove tracking columns (original_id, version) - not needed for training
        print(f"  > Removing tracking columns (original_id, version)...")
        df_test_clean = df_test[['source_text', 'target_text']].copy()
        
        # Save to JSONL
        df_test_clean.to_json(test_jsonl, orient='records', lines=True, force_ascii=False)
        print(f"  ✓ Successfully saved {len(df_test_clean):,} rows to: {test_jsonl}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("✅ CONVERSION COMPLETE!")
print(f"{'='*70}")
print(f"\n📁 Output files:")
print(f"  - {train_jsonl} (for training)")
print(f"  - {test_jsonl} (for validation)")
print(f"\n💡 Next step: Run MADLAD.py to start fine-tuning")
print(f"{'='*70}\n")