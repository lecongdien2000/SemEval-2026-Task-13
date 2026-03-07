import os
import sys

# ensure root package path is available when the file is executed directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.src_TaskA.models.model import HybridClassifier
from src.src_TaskA.dataset.dataset import AgnosticDataset
from src.src_TaskA.dataset.preprocess_features import AgnosticFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def prepare_test_data(test_path, config, device, force_reextract=False):
    """
    Prepares test data by checking if features are present.
    If not, extracts them using AgnosticFeatureExtractor.
    Caches processed data in current working directory (handles read-only /kaggle/input).
    Also checks for pre-downloaded test_processed.parquet in data directory.

    Args:
        test_path: Path to the test parquet file
        config: Configuration dictionary
        device: Device to use for feature extraction
        force_reextract: If True, always re-extract features even if present
    """
    logger.info(f"Loading test data: {test_path}")
    df = pd.read_parquet(test_path)

    def _has_valid_features(df):
        """Check if the dataframe has valid (non-zero) features."""
        if 'agnostic_features' not in df.columns:
            return False
        try:
            feats = np.array(df['agnostic_features'].tolist())
            # Check if more than 50% of rows have all zeros (extraction likely failed)
            zero_rows = (feats == 0).all(axis=1).mean()
            if zero_rows > 0.5:
                logger.warning(f"Features appear invalid: {zero_rows*100:.1f}% of rows have all zeros")
                return False
            return True
        except Exception:
            return False

    if 'agnostic_features' in df.columns and not force_reextract:
        if _has_valid_features(df):
            logger.info("Features already present and valid in dataset.")
            return df
        else:
            logger.warning("Features present but appear invalid. Re-extracting...")

    logger.info("Features missing or invalid. Initializing Feature Extractor (this may take a while)...")

    # Save cache to current working directory instead of input directory (handles read-only /kaggle/input)
    test_filename = os.path.basename(test_path)
    cache_filename = test_filename.replace(".parquet", "_processed.parquet")
    cache_path = os.path.join(os.getcwd(), cache_filename)

    # Also check original location in case it exists there
    original_cache_path = test_path.replace(".parquet", "_processed.parquet")

    # Also check for test_processed.parquet in the data directory (same location as train_processed and val_processed)
    data_dir = os.path.dirname(test_path)
    test_processed_path = os.path.join(data_dir, "test_processed.parquet")

    # Also check in the Google Drive downloaded data directory (same as train_processed and val_processed)
    gdrive_data_dir = "/kaggle/working/SemEval-2026-Task-13/data/Task_A_Processed"
    gdrive_test_processed = os.path.join(gdrive_data_dir, "test_processed.parquet")
    gdrive_test_preprocessed = os.path.join(gdrive_data_dir, "test_preprocessed.parquet")

    def _load_test_data(path):
        """Load test data and add dummy label if missing (required for inference)."""
        df = pd.read_parquet(path)
        if 'label' not in df.columns:
            df['label'] = 0
        return df

    if os.path.exists(cache_path):
        logger.info(f"Found cached processed file: {cache_path}")
        return _load_test_data(cache_path)
    elif os.path.exists(original_cache_path):
        logger.info(f"Found cached processed file: {original_cache_path}")
        return _load_test_data(original_cache_path)
    elif os.path.exists(test_processed_path):
        logger.info(f"Found pre-downloaded test_processed.parquet: {test_processed_path}")
        return _load_test_data(test_processed_path)
    elif os.path.exists(gdrive_test_processed):
        logger.info(f"Found pre-downloaded test_processed.parquet: {gdrive_test_processed}")
        return _load_test_data(gdrive_test_processed)
    elif os.path.exists(gdrive_test_preprocessed):
        logger.info(f"Found pre-downloaded test_preprocessed.parquet: {gdrive_test_preprocessed}")
        return _load_test_data(gdrive_test_preprocessed)

    extractor = AgnosticFeatureExtractor(config, str(device))
    
    features_list = []
    logger.info(f"Extracting features for {len(df)} test samples...")
    
    for code in tqdm(df['code'], desc="Feature Extraction"):
        try:
            feats = extractor.extract_all(code)
            features_list.append(feats)
        except Exception as e:
            logger.debug(f"Feature extraction failed for a sample: {e}")
            features_list.append([0.0] * 11)
            
    df['agnostic_features'] = features_list

    # Add dummy label column for inference (required by dataset)
    if 'label' not in df.columns:
        df['label'] = 0

    logger.info(f"Saving processed test data to {cache_path}")
    df.to_parquet(cache_path)
    
    del extractor
    torch.cuda.empty_cache()
    
    return df

def generate_submission(args):
    """
    Main function to generate submission predictions.
    Loads the trained model and creates submission.csv with ID and predicted labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 1. Load Configuration
    # Try loading config from checkpoint_dir, and if not found, try best_model subdirectory
    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.checkpoint_dir, "best_model", "config.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config not found in {args.checkpoint_dir} or {args.checkpoint_dir}/best_model")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")

    # 2. Prepare Test Data (extract features if needed)
    test_df = prepare_test_data(args.test_file, config, device, force_reextract=args.force_reextract)
    logger.info(f"Test data loaded: {len(test_df)} samples")

    # DEBUG: Check raw features in test data
    print("\n" + "="*60)
    print("RAW TEST DATA FEATURES (Debug)".center(60))
    print("="*60)
    test_has_valid_features = False
    if 'agnostic_features' in test_df.columns:
        raw_feats = np.array(test_df['agnostic_features'].tolist())
        print(f"Feature matrix shape: {raw_feats.shape}")
        print(f"Perplexity (idx 0)  - Mean: {raw_feats[:, 0].mean():.4f}, Std: {raw_feats[:, 0].std():.4f}")
        print(f"  Zero count: {(raw_feats[:, 0] == 0).sum()} / {len(raw_feats[:, 0])}")
        print(f"ID Len Avg (idx 1)  - Mean: {raw_feats[:, 1].mean():.4f}, Std: {raw_feats[:, 1].std():.4f}")
        print(f"Style Consistency    - Mean: {raw_feats[:, 5].mean():.4f}, Std: {raw_feats[:, 5].std():.4f}") if raw_feats.shape[1] > 5 else None
        print(f"Line Len Std (idx 7)- Mean: {raw_feats[:, 7].mean():.4f}, Std: {raw_feats[:, 7].std():.4f}") if raw_feats.shape[1] > 7 else None

        # Check if features are mostly zeros (indicates extraction failure)
        zero_ratio = (raw_feats == 0).all(axis=1).mean()
        print(f"\nRows with ALL zeros: {zero_ratio*100:.1f}%")
        if zero_ratio > 0.5:
            print("WARNING: >50% of test samples have zero features! Feature extraction likely failed.")
        test_has_valid_features = zero_ratio < 0.5
    else:
        print("WARNING: No 'agnostic_features' column found in test data!")
    print("="*60 + "\n")

    # Try to load training data for comparison (if available)
    train_data_path = config["data"].get("data_dir", "data/Task_A_Processed")
    train_file = os.path.join(train_data_path, "train_processed.parquet")
    if os.path.exists(train_file):
        print("\n" + "="*60)
        print("TRAIN DATA FEATURES (for comparison)".center(60))
        print("="*60)
        train_df_debug = pd.read_parquet(train_file)
        if 'agnostic_features' in train_df_debug.columns:
            train_feats = np.array(train_df_debug['agnostic_features'].tolist())
            print(f"Feature matrix shape: {train_feats.shape}")
            print(f"Perplexity (idx 0)  - Mean: {train_feats[:, 0].mean():.4f}, Std: {train_feats[:, 0].std():.4f}")
            print(f"ID Len Avg (idx 1)  - Mean: {train_feats[:, 1].mean():.4f}, Std: {train_feats[:, 1].std():.4f}")
            print(f"Style Consistency    - Mean: {train_feats[:, 5].mean():.4f}, Std: {train_feats[:, 5].std():.4f}") if train_feats.shape[1] > 5 else None
            print(f"Line Len Std (idx 7)- Mean: {train_feats[:, 7].mean():.4f}, Std: {train_feats[:, 7].std():.4f}") if train_feats.shape[1] > 7 else None
        print("="*60 + "\n")

    # Ensure 'id' column exists
    if 'id' not in test_df.columns and 'ID' not in test_df.columns:
        # If no ID column, use the index
        test_df['id'] = range(len(test_df))
        logger.warning("No 'id' column found. Using index as ID.")
    elif 'ID' in test_df.columns:
        test_df['id'] = test_df['ID']

    # 3. Load Model & Tokenizer
    logger.info("Loading Model & Tokenizer...")
    # Determine the actual checkpoint directory (might be in best_model subdirectory)
    actual_checkpoint_dir = args.checkpoint_dir
    best_model_dir = os.path.join(args.checkpoint_dir, "best_model")
    if os.path.exists(best_model_dir):
        actual_checkpoint_dir = best_model_dir
    
    tokenizer = AutoTokenizer.from_pretrained(actual_checkpoint_dir)
    
    model = HybridClassifier(config)
    
    weights_path = os.path.join(actual_checkpoint_dir, "model_state.bin")
    if not os.path.exists(weights_path):
        logger.error(f"Model weights not found at {weights_path}")
        sys.exit(1)
        
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")

    # 4. Create DataLoader
    dataset = AgnosticDataset(
        test_df, 
        tokenizer, 
        max_length=config["data"]["max_length"], 
        is_train=False
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # 5. Run Inference
    logger.info("Running inference on test set...")
    all_predictions = []
    all_probabilities = []
    all_ids = []

    # DEBUG: Track feature statistics
    feature_stats = {
        'perplexity': [],
        'id_len_avg': [],
        'style_consistency': [],
        'line_len_std': []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            feats = batch["extra_features"].to(device, non_blocking=True)

            # DEBUG: Collect feature statistics before model inference
            feature_stats['perplexity'].extend(feats[:, 0].cpu().numpy().tolist())
            feature_stats['id_len_avg'].extend(feats[:, 1].cpu().numpy().tolist())
            feature_stats['style_consistency'].extend(feats[:, 5].cpu().numpy().tolist() if feats.shape[1] > 5 else [])
            feature_stats['line_len_std'].extend(feats[:, 7].cpu().numpy().tolist() if feats.shape[1] > 7 else [])

            # Forward pass
            logits, _, _ = model(input_ids, attention_mask, feats, labels=None)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())
            # all_ids.extend(batch["id"])  # This should be a list of string IDs

    logger.info(f"Inference complete. Generated {len(all_predictions)} predictions.")

    # DEBUG: Print feature statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS (Debug)".center(60))
    print("="*60)
    if feature_stats['perplexity']:
        ppl_arr = np.array(feature_stats['perplexity'])
        print(f"Perplexity (idx 0)  - Mean: {ppl_arr.mean():.4f}, Std: {ppl_arr.std():.4f}, Min: {ppl_arr.min():.4f}, Max: {ppl_arr.max():.4f}")
        print(f"  Zero count: {(ppl_arr == 0).sum()} / {len(ppl_arr)}")
    if feature_stats['id_len_avg']:
        id_arr = np.array(feature_stats['id_len_avg'])
        print(f"ID Len Avg (idx 1)  - Mean: {id_arr.mean():.4f}, Std: {id_arr.std():.4f}, Min: {id_arr.min():.4f}, Max: {id_arr.max():.4f}")
    if feature_stats['style_consistency']:
        style_arr = np.array(feature_stats['style_consistency'])
        print(f"Style Consistency    - Mean: {style_arr.mean():.4f}, Std: {style_arr.std():.4f}")
    if feature_stats['line_len_std']:
        line_arr = np.array(feature_stats['line_len_std'])
        print(f"Line Len Std (idx 7)- Mean: {line_arr.mean():.4f}, Std: {line_arr.std():.4f}")
    print("="*60 + "\n")

    # Print probability distribution
    prob_arr = np.array(all_probabilities)
    print(f"Probability Stats - Class 0 (Human): mean={prob_arr[:, 0].mean():.4f}, std={prob_arr[:, 0].std():.4f}")
    print(f"                  - Class 1 (AI):     mean={prob_arr[:, 1].mean():.4f}, std={prob_arr[:, 1].std():.4f}")
    print("="*60 + "\n")

    all_ids = test_df['id'].tolist()

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({
        'ID': all_ids,
        'label': all_predictions
    })
    
    # Ensure ID column is of the expected type (convert to int if possible)
    try:
        submission_df['ID'] = submission_df['ID'].astype(int)
    except (ValueError, TypeError):
        logger.warning("Could not convert ID column to int. Keeping as string.")

    # 7. Save Submission File
    output_path = args.output_file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    
    # 8. Print Summary
    print("\n" + "="*60)
    print("SUBMISSION GENERATION COMPLETE".center(60))
    print("="*60)
    print(f"Total predictions: {len(submission_df)}")
    # NOTE: According to official SemEval Task A convention:
    # - 0 = machine-generated (AI)
    # - 1 = human-written (Human)
    print(f"AI (0): {(submission_df['label'] == 0).sum()}")
    print(f"Human (1): {(submission_df['label'] == 1).sum()}")
    print(f"Output file: {output_path}")
    print("="*60 + "\n")
    
    return submission_df

class SubmissionArgs:
    """Simple argument container for non-CLI usage (e.g., Kaggle notebooks)."""
    def __init__(self, test_file, checkpoint_dir, output_file="submission.csv", batch_size=32):
        self.test_file = test_file
        self.checkpoint_dir = checkpoint_dir
        self.output_file = output_file
        self.batch_size = batch_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate submission CSV for SemEval Task A using trained model"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        required=True, 
        help="Path to test .parquet file (e.g., test_sample.parquet)"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True, 
        help="Path to the saved model checkpoint directory"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="submission.csv",
        help="Path where the submission CSV will be saved (default: submission.csv)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="Force re-extraction of features even if cached file exists"
    )

    args = parser.parse_args()
    
    submission_df = generate_submission(args)

# ============================================================================
# USAGE IN KAGGLE NOTEBOOKS
# ============================================================================
# 
# In a Kaggle notebook cell, use the following code:
#
# from src.src_TaskA.generate_submission_AI import generate_submission, SubmissionArgs
#
# args = SubmissionArgs(
#     test_file="/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A/test.parquet",
#     checkpoint_dir="/kaggle/input/your-model-checkpoint",  # or any local path
#     output_file="submission.csv",
#     batch_size=32
# )
#
# submission_df = generate_submission(args)
#
