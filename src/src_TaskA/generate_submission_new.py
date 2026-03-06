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

def prepare_test_data(test_path, config, device):
    """
    Prepares test data by checking if features are present.
    If not, extracts them using AgnosticFeatureExtractor.
    Caches processed data in current working directory (handles read-only /kaggle/input).
    Also checks for pre-downloaded test_processed.parquet in data directory.
    """
    logger.info(f"Loading test data: {test_path}")
    df = pd.read_parquet(test_path)

    if 'agnostic_features' in df.columns:
        logger.info("Features already present in dataset.")
        return df

    logger.info("Features missing. Initializing Feature Extractor (this may take a while)...")

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
    test_df = prepare_test_data(args.test_file, config, device)
    logger.info(f"Test data loaded: {len(test_df)} samples")
    
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            feats = batch["extra_features"].to(device, non_blocking=True)
            
            # Forward pass
            logits, _, _ = model(input_ids, attention_mask, feats, labels=None)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())
            # all_ids.extend(batch["id"])  # This should be a list of string IDs

    logger.info(f"Inference complete. Generated {len(all_predictions)} predictions.")

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
    print(f"Human (0): {(submission_df['label'] == 0).sum()}")
    print(f"AI (1): {(submission_df['label'] == 1).sum()}")
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
