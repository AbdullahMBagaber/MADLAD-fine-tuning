import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

"""
TEST SET EXTRACTION SCRIPT WITH FUZZY DEDUPLICATION
====================================================
This script extracts exactly 200 samples for testing from the combined dataset
and ensures no duplicates exist between train and test sets using:
1. original_id tracking (removes exact duplicate versions)
2. Fuzzy matching (removes near-duplicates with >95% similarity)

Input:  combined_dataset.xlsx
Output: training_dataset.xlsx (remaining rows, deduplicated)
        test_dataset.xlsx (200 unique examples × ~2 versions)
"""

def compute_embeddings_batch(texts, model, batch_size=32, desc="Encoding"):
    """
    Compute sentence embeddings using GPU acceleration.
    Returns embeddings as numpy array.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,  # Show progress bar
        convert_to_numpy=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return embeddings


def find_duplicates_gpu(train_df, test_df, model, similarity_threshold=0.95):
    """
    Find duplicates using GPU-accelerated embedding similarity.
    Much faster than row-by-row string comparison.
    
    Returns indices of training rows to keep (non-duplicates).
    """
    print(f"  > Computing embeddings on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    
    # Extract texts
    train_sources = train_df['source_text'].tolist()
    train_targets = train_df['target_text'].tolist()
    test_sources = test_df['source_text'].tolist()
    test_targets = test_df['target_text'].tolist()
    
    # Compute embeddings in batches (GPU accelerated)
    import time
    
    print(f"\n  🔄 Encoding {len(train_sources):,} training sources...")
    start_time = time.time()
    train_source_emb = compute_embeddings_batch(train_sources, model)
    elapsed = time.time() - start_time
    print(f"     ✓ Done in {elapsed:.1f}s ({len(train_sources)/elapsed:.0f} texts/sec)")
    
    print(f"\n  🔄 Encoding {len(train_targets):,} training targets...")
    start_time = time.time()
    train_target_emb = compute_embeddings_batch(train_targets, model)
    elapsed = time.time() - start_time
    print(f"     ✓ Done in {elapsed:.1f}s ({len(train_targets)/elapsed:.0f} texts/sec)")
    
    print(f"\n  🔄 Encoding {len(test_sources):,} test sources...")
    start_time = time.time()
    test_source_emb = compute_embeddings_batch(test_sources, model)
    elapsed = time.time() - start_time
    print(f"     ✓ Done in {elapsed:.1f}s ({len(test_sources)/elapsed:.0f} texts/sec)")
    
    print(f"\n  🔄 Encoding {len(test_targets):,} test targets...")
    start_time = time.time()
    test_target_emb = compute_embeddings_batch(test_targets, model)
    elapsed = time.time() - start_time
    print(f"     ✓ Done in {elapsed:.1f}s ({len(test_targets)/elapsed:.0f} texts/sec)")
    
    # Compute similarity matrices (vectorized, very fast)
    print(f"\n  💫 Computing similarity matrices...")
    start_time = time.time()
    source_sim_matrix = cosine_similarity(train_source_emb, test_source_emb)  # (train_size, test_size)
    target_sim_matrix = cosine_similarity(train_target_emb, test_target_emb)  # (train_size, test_size)
    elapsed = time.time() - start_time
    print(f"     ✓ Done in {elapsed:.1f}s")
    
    # Find rows where BOTH source AND target are highly similar to ANY test example
    print(f"\n  🔍 Finding duplicates (threshold: {similarity_threshold:.0%})...")
    keep_mask = []
    for i in tqdm(range(len(train_df)), desc="  Checking", unit=" rows"):
        # Get max similarity to any test example for this training example
        max_source_sim = source_sim_matrix[i].max()
        max_target_sim = target_sim_matrix[i].max()
        
        # Keep if NOT both highly similar
        is_duplicate = (max_source_sim >= similarity_threshold and 
                       max_target_sim >= similarity_threshold)
        keep_mask.append(not is_duplicate)
    
    return keep_mask


def extract_test_set():
    """
    Extracts 200 test samples from the combined dataset and saves separate files.
    Uses both original_id tracking and fuzzy matching for comprehensive deduplication.
    """
    print("=" * 70)
    print("TEST SET EXTRACTION WITH GPU-ACCELERATED DEDUPLICATION")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load the combined dataset")
    print("  2. Load multilingual sentence transformer model (GPU if available)")
    print("  3. Randomly sample 200 unique examples for testing")
    print("  4. Remove exact duplicates (same original_id)")
    print("  5. Remove near-duplicates using GPU-accelerated embeddings (>95% similarity)")
    print("  6. Save training and test sets separately")
    print("  7. Verify no duplicates exist between sets")
    print("=" * 70)
    
    # --- File Paths ---
    input_path = Path("combined_dataset.xlsx")
    train_output = Path("training_dataset.xlsx")
    test_output = Path("test_dataset.xlsx")
    
    # --- Configuration ---
    RANDOM_SEED = 42
    TEST_SIZE = 200
    SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold (0.0 to 1.0)
    EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # Supports Arabic & French (50+ languages)
    
    try:
        # =====================================================================
        # STEP 1: LOAD COMBINED DATASET
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 1: LOADING COMBINED DATASET")
        print(f"{'='*70}")
        print(f"Reading from: {input_path}")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file '{input_path}' not found. Please run create_combined_dataset.py first.")
        
        df = pd.read_excel(input_path)
        total_rows = len(df)
        print(f"✓ Loaded {total_rows:,} rows with {df.shape[1]} columns")
        
        # Validate required columns
        expected_cols = ['source_text', 'target_text', 'original_id', 'version']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Input file must contain 'source_text', 'target_text', 'original_id', and 'version'. Found: {list(df.columns)}")
        
        # Check duplicate tracking
        unique_ids = df['original_id'].nunique()
        print(f"  > Found {unique_ids} unique original IDs")
        print(f"  > Total rows: {total_rows} (should be ~2x unique IDs due to doubled dataset)")
        
        # Validate we have enough unique examples
        if unique_ids <= TEST_SIZE:
            raise ValueError(f"Dataset has only {unique_ids} unique examples, cannot extract {TEST_SIZE} test samples. Need at least {TEST_SIZE + 1} unique examples.")
        
        # =====================================================================
        # STEP 2: LOAD SENTENCE TRANSFORMER MODEL
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 2: LOADING MULTILINGUAL EMBEDDING MODEL")
        print(f"{'='*70}")
        print(f"  > Model: {EMBEDDING_MODEL}")
        print(f"  > Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"  > GPU: {torch.cuda.get_device_name(0)}")
        
        model = SentenceTransformer(EMBEDDING_MODEL)
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        print(f"✓ Model loaded successfully")
        
        # =====================================================================
        # STEP 3: RANDOM SAMPLING FOR TEST SET (WITH DUPLICATE HANDLING)
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"STEP 3: EXTRACTING {TEST_SIZE} TEST SAMPLES")
        print(f"{'='*70}")
        print(f"  > Using random seed: {RANDOM_SEED} (for reproducibility)")
        print(f"  > Sampling {TEST_SIZE} unique original_ids")
        print(f"  > Will extract ALL versions of sampled IDs (removes duplicates from training)")
        
        # Set random seed
        np.random.seed(RANDOM_SEED)
        
        # Get unique original_ids
        unique_original_ids = df['original_id'].unique()
        
        # Randomly sample TEST_SIZE unique original_ids
        sampled_ids = np.random.choice(unique_original_ids, size=TEST_SIZE, replace=False)
        print(f"  > Sampled {len(sampled_ids)} unique original_ids")
        
        # Get ALL rows (both versions) that have these original_ids
        test_df = df[df['original_id'].isin(sampled_ids)].copy()
        
        print(f"✓ Extracted {len(test_df)} total rows for testing ({len(sampled_ids)} unique examples × ~2 versions)")
        print(f"  > Breakdown by version:")
        for version in test_df['version'].unique():
            count = len(test_df[test_df['version'] == version])
            print(f"    - {version}: {count} rows")
        
        # =====================================================================
        # STEP 3: CREATE TRAINING SET (REMOVING ALL DUPLICATE VERSIONS)
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 3: CREATING TRAINING SET (REMOVING DUPLICATES)")
        print(f"{'='*70}")
        print(f"  > Removing all rows with sampled original_ids from training set")
        print(f"  > This ensures no test examples appear in training (any version)")
        
        # Remove ALL rows with the sampled original_ids (not just test_df indices)
        train_df = df[~df['original_id'].isin(sampled_ids)].copy()
        
        print(f"✓ Training set created with {len(train_df):,} rows")
        print(f"  > Breakdown by version:")
        for version in train_df['version'].unique():
            count = len(train_df[train_df['version'] == version])
            print(f"    - {version}: {count} rows")
        
        # =====================================================================
        # STEP 4: VERIFY NO DUPLICATES (CRITICAL FOR DOUBLED DATASET)
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 4: VERIFYING NO DUPLICATES")
        print(f"{'='*70}")
        
        # Check for original_id overlap (should be zero - this is the critical check!)
        train_ids = set(train_df['original_id'].unique())
        test_ids = set(test_df['original_id'].unique())
        overlap = train_ids.intersection(test_ids)
        
        if len(overlap) > 0:
            raise ValueError(f"ERROR: Found {len(overlap)} overlapping original_ids between train and test sets! This means duplicates exist.")
        
        print(f"✓ No original_id overlap detected (CRITICAL CHECK PASSED)")
        print(f"  > Training set has {len(train_ids)} unique original_ids")
        print(f"  > Test set has {len(test_ids)} unique original_ids")
        print(f"  > No overlap = No data leakage!")
        
        # Verify total count
        if len(train_df) + len(test_df) != total_rows:
            raise ValueError(f"ERROR: Row count mismatch! Train ({len(train_df)}) + Test ({len(test_df)}) != Total ({total_rows})")
        
        print(f"✓ Row counts verified: {len(train_df):,} + {len(test_df):,} = {total_rows:,}")
        
        # =====================================================================
        # STEP 5: GPU-ACCELERATED FUZZY DEDUPLICATION
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 5: GPU-ACCELERATED SEMANTIC DEDUPLICATION")
        print(f"{'='*70}")
        print(f"  > Using sentence embeddings for semantic similarity")
        print(f"  > Comparing both source AND target texts")
        print(f"  > Similarity threshold: {SIMILARITY_THRESHOLD:.0%}")
        print(f"  > Processing on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
        
        train_before_fuzzy = len(train_df)
        
        # Use GPU-accelerated deduplication with timing
        import time
        dedup_start = time.time()
        keep_mask = find_duplicates_gpu(train_df, test_df, model, SIMILARITY_THRESHOLD)
        dedup_elapsed = time.time() - dedup_start
        
        # Apply the mask
        train_df = train_df[keep_mask].copy()
        removed_fuzzy = train_before_fuzzy - len(train_df)
        
        print(f"\n{'='*70}")
        print(f"✓ GPU-accelerated deduplication complete!")
        print(f"{'='*70}")
        print(f"  ⏱️  Total time: {dedup_elapsed/60:.1f} minutes ({dedup_elapsed:.0f}s)")
        print(f"  🗑️  Removed: {removed_fuzzy:,} near-duplicate rows (>{SIMILARITY_THRESHOLD:.0%} similar)")
        print(f"  📊 Final training set: {len(train_df):,} rows")
        print(f"{'='*70}")
        
        # =====================================================================
        # STEP 6: SAVE DATASETS
        # =====================================================================
        print(f"\n{'='*70}")
        print("STEP 6: SAVING TRAIN AND TEST SETS")
        print(f"{'='*70}")
        
        # Reset indices for clean datasets
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Save to Excel files
        train_df.to_excel(train_output, index=False)
        print(f"✓ Training set saved to: {train_output}")
        
        test_df.to_excel(test_output, index=False)
        print(f"✓ Test set saved to: {test_output}")
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print(f"\n{'='*70}")
        print("✅ TEST SET EXTRACTION COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📊 SUMMARY:")
        print(f"  Original combined dataset:      {total_rows:,} rows ({unique_ids:,} unique examples × ~2 versions)")
        print(f"  Test set:                       {len(test_df):,} rows ({len(test_ids)} unique examples)")
        print(f"  ─────────────────────────────────────────")
        print(f"  DEDUPLICATION RESULTS:")
        print(f"  Step 1 - original_id removal:   {train_before_fuzzy:,} rows remaining")
        print(f"  Step 2 - fuzzy matching removal: {removed_fuzzy:,} near-duplicates removed")
        print(f"  Final training set:             {len(train_df):,} rows")
        print(f"  ─────────────────────────────────────────")
        print(f"  Total removed from training:    {total_rows - len(test_df) - len(train_df):,} rows")
        print(f"  ─────────────────────────────────────────")
        print(f"  ✓ No original_id overlap")
        print(f"  ✓ No near-duplicates (>{SIMILARITY_THRESHOLD:.0%} cosine similarity)")
        print(f"  ✓ GPU-accelerated processing: {'Yes (CUDA)' if torch.cuda.is_available() else 'No (CPU)'}")
        print(f"  ✓ Clean train/test split achieved!")
        print(f"\n📁 Output files:")
        print(f"  - {train_output} ({len(train_df):,} rows)")
        print(f"  - {test_output} ({len(test_df):,} rows)")
        print(f"\n💡 Next step: Run to_json.py to convert to JSONL format")
        print(f"{'='*70}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    extract_test_set()

