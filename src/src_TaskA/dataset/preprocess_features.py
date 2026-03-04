import os
import sys
import re
import math
import json
import glob
import yaml
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 1. SETUP LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & RESOURCES
# -----------------------------------------------------------------------------
MULTI_LANG_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
    'int', 'float', 'double', 'char', 'void', 'bool', 'boolean', 'string', 'var', 'let', 'const',
    'def', 'function', 'class', 'struct', 'interface', 'package', 'import', 'using', 'namespace',
    'public', 'private', 'protected', 'static', 'final', 'const', 'try', 'catch', 'finally',
    'throw', 'throws', 'new', 'delete', 'true', 'false', 'null', 'nil', 'None', 'self', 'this',
    'func', 'defer', 'go', 'map', 'chan', 'type', 'range'
}

# -----------------------------------------------------------------------------
# 3. FEATURE EXTRACTOR CLASS
# -----------------------------------------------------------------------------
class AgnosticFeatureExtractor:
    """
    Estrae feature stilometriche e language-agnostic dal codice sorgente.
    Combina metriche neurali (Perplexity) con metriche statistiche (Consistenza, Entropia).
    """
    def __init__(self, config: Dict, device: str):
        self.device = device
        self.config = config
        
        # A. Setup Perplexity Model
        model_name = config["data"].get("perplexity_model", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        logger.info(f"Loading Perplexity Model: {model_name} on {device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if "cuda" in device else torch.float32, 
                device_map=device,
                trust_remote_code=True
            ).eval()
            self.max_len = 512
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

        # B. Compile Regex for Performance
        # Tokenizer semplice per codice: parole alfanumeriche
        self.re_words = re.compile(r'\w+')
        # Rilevamento CamelCase (es. myVar, MyClass)
        self.re_camel = re.compile(r'[a-z][A-Z]')
        # Rilevamento SnakeCase (es. my_var)
        self.re_snake = re.compile(r'_')
        # Rilevamento Numeri
        self.re_digits = re.compile(r'\d')
        # Rilevamento Spaziature Operatori (es. " = " vs "=")
        self.re_eq_spaced = re.compile(r' = ')
        self.re_eq_nospaced = re.compile(r'(?<=[^\s])=(?=[^\s])') # "=" senza spazi attorno

    def get_feature_names(self) -> List[str]:
        """Restituisce i nomi delle feature per analisi future."""
        return [
            "perplexity",          # 0
            "id_len_avg",          # 1
            "id_entropy",          # 2
            "id_short_ratio",      # 3
            "id_num_ratio",        # 4
            "style_consistency",   # 5
            "spacing_ratio",       # 6
            "line_len_std",        # 7
            "ttr",                 # 8
            "comment_ratio",       # 9
            "human_markers"        # 10
        ]

    def _compute_perplexity(self, code: str) -> float:
        """Calcola la Cross-Entropy Loss (Perplexity logaritmica)."""
        if not code.strip():
            return 0.0
            
        try:
            inputs = self.tokenizer(
                code, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_len
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=inputs.input_ids, labels=inputs.input_ids)
                return outputs.loss.item()
        except Exception:
            return 0.0

    def _analyze_identifiers(self, words: List[str]) -> List[float]:
        """Analizza le proprietà dei nomi di variabili/funzioni."""
        # Filtra keyword e numeri puri
        identifiers = [w for w in words if w not in MULTI_LANG_KEYWORDS and not w.isdigit()]
        
        if not identifiers:
            return [0.0, 0.0, 0.0, 0.0]

        # 1. Lunghezza Media
        lens = [len(w) for w in identifiers]
        avg_len = np.mean(lens)

        # 2. Entropia Caratteri
        all_chars = "".join(identifiers)
        if not all_chars:
            entropy = 0.0
        else:
            char_counts = Counter(all_chars)
            total_chars = sum(char_counts.values())
            entropy = -sum((c / total_chars) * math.log2(c / total_chars) for c in char_counts.values())

        # 3. Ratio ID Corti (<= 2 chars) -> Tipico umano (i, j, k, x)
        short_ids = sum(1 for w in identifiers if len(w) <= 2)
        short_ratio = short_ids / len(identifiers)

        # 4. Ratio ID con Numeri (var1, temp2) -> Tipico umano/legacy
        num_ids = sum(1 for w in identifiers if self.re_digits.search(w))
        num_ratio = num_ids / len(identifiers)

        return [avg_len, entropy, short_ratio, num_ratio]

    def _analyze_consistency(self, code: str, words: List[str]) -> List[float]:
        """Analizza la coerenza di formattazione e stile."""
        identifiers = [w for w in words if w not in MULTI_LANG_KEYWORDS]
        
        # 1. Style Consistency (Camel vs Snake)
        snake_count = sum(1 for w in identifiers if '_' in w)
        camel_count = sum(1 for w in identifiers if self.re_camel.search(w))
        total_style = snake_count + camel_count
        
        if total_style == 0:
            consistency = 0.0 # Neutro
        else:
            # 1.0 = usa solo uno stile, 0.0 = mix perfetto (caos umano)
            consistency = abs(snake_count - camel_count) / total_style

        # 2. Spacing Consistency (attorno a '=')
        # Gli LLM tendono a mettere sempre spazi (" = "), gli umani sono pigri ("=")
        spaced = len(self.re_eq_spaced.findall(code))
        nospaced = len(self.re_eq_nospaced.findall(code))
        total_eq = spaced + nospaced
        
        if total_eq == 0:
            spacing_ratio = 0.0
        else:
            # Ratio della forma "sporca" (senza spazi)
            spacing_ratio = nospaced / total_eq

        return [consistency, spacing_ratio]

    def _analyze_structure(self, code: str, words: List[str]) -> List[float]:
        """Analizza struttura righe e commenti."""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # 1. Varianza Lunghezza Righe (Human -> High Variance)
        if non_empty_lines:
            line_lens = [len(l) for l in non_empty_lines]
            line_std = np.std(line_lens)
        else:
            line_std = 0.0
            
        # 2. Type-Token Ratio (Ripetitività)
        if words:
            ttr = len(set(words)) / len(words)
        else:
            ttr = 0.0

        # 3. Comment Ratio & Human Markers
        comment_lines = sum(1 for l in non_empty_lines if l.strip().startswith(('#', '//', '/*')))
        comment_ratio = comment_lines / (len(non_empty_lines) + 1)
        
        # Human Markers (TODO, FIXME, DEBUG) - forti indicatori umani
        markers = len(re.findall(r'\b(TODO|FIXME|XXX|HACK|DEBUG)\b', code, re.IGNORECASE))
        marker_score = 1.0 if markers > 0 else 0.0

        return [line_std, ttr, comment_ratio, marker_score]

    def extract_all(self, code: str) -> List[float]:
        """Entry point principale per estrarre il vettore completo."""
        if not isinstance(code, str):
            code = str(code)

        # 1. Neural Metric (Slowest)
        ppl = self._compute_perplexity(code)
        
        # 2. Pre-computazioni lessicali
        words = self.re_words.findall(code)
        
        # 3. Estrazione sottogruppi
        f_ids = self._analyze_identifiers(words)
        f_const = self._analyze_consistency(code, words)
        f_struct = self._analyze_structure(code, words)
        
        # Concatenazione: [1] + [4] + [2] + [4] = 11 Features
        return [ppl] + f_ids + f_const + f_struct

# -----------------------------------------------------------------------------
# 4. PROCESSING PIPELINE
# -----------------------------------------------------------------------------
def _extract_range_from_chunk_name(chunk_path: str):
    base = os.path.basename(chunk_path)
    match = re.search(r"chunk_(\d+)_(\d+)\.parquet$", base)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _get_parquet_row_count(parquet_path: str):
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        return None


def _is_split_already_processed(
    output_path: str,
    split_name: str,
    expected_rows: int = None,
    require_done_marker: bool = False
):
    if not os.path.exists(output_path):
        return False

    if require_done_marker:
        done_marker_path = os.path.join(
            os.path.dirname(output_path),
            f"{split_name}_processed.done.json"
        )
        if not os.path.exists(done_marker_path):
            return False
        try:
            with open(done_marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
            if not marker.get("complete", False):
                return False
        except Exception:
            return False

    if expected_rows is not None and expected_rows > 0:
        row_count = _get_parquet_row_count(output_path)
        if row_count != int(expected_rows):
            return False

    return True


def process_data_split(
    input_path: str,
    output_path: str,
    extractor: AgnosticFeatureExtractor,
    split_name: str = "train",
    checkpoint_every: int = 5000,
    enable_resume_checkpoints: bool = False,
    expected_rows: int = None
):
    """Processa un file parquet e salva i risultati (con checkpoint/resume opzionale)."""
    if not os.path.exists(input_path):
        logger.warning(f"Input file not found: {input_path}. Skipping.")
        return

    logger.info(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Check robustezza colonne
    if 'code' not in df.columns:
        logger.error("Column 'code' missing in dataframe.")
        return

    total_rows = len(df)
    if expected_rows is not None and total_rows != expected_rows:
        logger.warning(
            f"{split_name}: input rows ({total_rows}) != expected_rows ({expected_rows})."
        )

    # Simple full-pass mode (val/test or when checkpointing disabled)
    if not enable_resume_checkpoints:
        features_list = []
        logger.info(f"Extracting features for {total_rows} samples...")
        for code in tqdm(df['code'], desc=f"Processing {split_name}", dynamic_ncols=True):
            try:
                feats = extractor.extract_all(code)
                features_list.append(feats)
            except Exception as e:
                logger.warning(f"Error processing snippet: {e}. Using zero-vector.")
                features_list.append([0.0] * 11)

        df['agnostic_features'] = features_list
        logger.info(f"Saving processed data to {output_path}...")
        df.to_parquet(output_path, index=False)

        logger.info("Feature names: " + str(extractor.get_feature_names()))
        sample_feat = np.array(features_list)
        logger.info(f"Feature Matrix Shape: {sample_feat.shape}")
        logger.info(f"Avg Perplexity: {np.mean(sample_feat[:, 0]):.4f}")
        return

    # Checkpoint/resume mode (designed for TRAIN split)
    checkpoint_dir = os.path.join(os.path.dirname(output_path), "checkpoints", f"{split_name}_chunks")
    os.makedirs(checkpoint_dir, exist_ok=True)
    progress_path = os.path.join(checkpoint_dir, "progress.json")
    done_marker_path = os.path.join(os.path.dirname(output_path), f"{split_name}_processed.done.json")

    chunk_glob = os.path.join(checkpoint_dir, f"{split_name}_chunk_*.parquet")
    existing_chunks = sorted(glob.glob(chunk_glob))

    resume_idx = 0
    for cpath in existing_chunks:
        rng = _extract_range_from_chunk_name(cpath)
        if rng is None:
            continue
        _, end = rng
        resume_idx = max(resume_idx, end)

    logger.info(
        f"{split_name}: checkpoint mode ON | total_rows={total_rows} | "
        f"checkpoint_every={checkpoint_every} | resume_idx={resume_idx}"
    )

    for start in range(resume_idx, total_rows, checkpoint_every):
        end = min(start + checkpoint_every, total_rows)
        chunk_df = df.iloc[start:end].copy()
        chunk_features = []

        logger.info(f"{split_name}: extracting rows [{start}:{end}) ...")
        for code in tqdm(chunk_df["code"], desc=f"{split_name} {start}:{end}", dynamic_ncols=True):
            try:
                feats = extractor.extract_all(code)
                chunk_features.append(feats)
            except Exception as e:
                logger.warning(f"Error processing snippet: {e}. Using zero-vector.")
                chunk_features.append([0.0] * 11)

        chunk_df["agnostic_features"] = chunk_features
        chunk_path = os.path.join(
            checkpoint_dir,
            f"{split_name}_chunk_{start:07d}_{end:07d}.parquet"
        )
        chunk_df.to_parquet(chunk_path, index=False)

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split_name,
                    "processed_rows": end,
                    "total_rows": total_rows,
                    "checkpoint_every": checkpoint_every,
                    "latest_chunk": os.path.basename(chunk_path),
                    "complete": False
                },
                f,
                indent=2
            )
        logger.info(f"{split_name}: checkpoint saved -> {chunk_path}")

    final_chunk_paths = sorted(glob.glob(chunk_glob), key=lambda p: _extract_range_from_chunk_name(p) or (0, 0))
    if not final_chunk_paths:
        logger.error(f"{split_name}: no checkpoint chunks found in {checkpoint_dir}.")
        return

    logger.info(f"{split_name}: combining {len(final_chunk_paths)} chunk files into final parquet...")
    final_df = pd.concat((pd.read_parquet(p) for p in final_chunk_paths), ignore_index=True)
    final_df = final_df.iloc[:total_rows].copy()

    if len(final_df) != total_rows:
        logger.error(
            f"{split_name}: final row count mismatch after combine "
            f"({len(final_df)} vs expected {total_rows})."
        )
        return

    final_df.to_parquet(output_path, index=False)

    feature_matrix = np.array(final_df["agnostic_features"].tolist(), dtype=np.float32)
    logger.info(f"{split_name}: final processed saved to {output_path}")
    logger.info("Feature names: " + str(extractor.get_feature_names()))
    logger.info(f"Feature Matrix Shape: {feature_matrix.shape}")
    logger.info(f"Avg Perplexity: {np.mean(feature_matrix[:, 0]):.4f}")

    with open(done_marker_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split_name,
                "complete": True,
                "total_rows": total_rows,
                "checkpoint_every": checkpoint_every,
                "output_path": output_path
            },
            f,
            indent=2
        )

    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split_name,
                "processed_rows": total_rows,
                "total_rows": total_rows,
                "checkpoint_every": checkpoint_every,
                "complete": True
            },
            f,
            indent=2
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemEval Task A - Feature Preprocessing")
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Rows per saved checkpoint chunk for train split")
    parser.add_argument("--expected-train-rows", type=int, default=500000, help="Expected final train row count")
    args = parser.parse_args()

    # 1. Load Config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 3. Initialize Extractor
    extractor = AgnosticFeatureExtractor(config, device)

    # 4. Paths Definitions
    raw_dir = config["data"].get("raw_data_dir", "data/Task_A")
    proc_dir = config["data"].get("data_dir", "data/Task_A_Processed")
    os.makedirs(proc_dir, exist_ok=True)

    # 5. Run Processing
    train_input = os.path.join(raw_dir, "train_binary.parquet") 
    val_input = os.path.join(raw_dir, "val_binary.parquet")
    
    train_output = os.path.join(proc_dir, "train_processed.parquet")
    val_output = os.path.join(proc_dir, "val_processed.parquet")
    expected_train_rows = int(args.expected_train_rows) if args.expected_train_rows > 0 else None

    train_already_done = _is_split_already_processed(
        output_path=train_output,
        split_name="train",
        expected_rows=expected_train_rows,
        require_done_marker=True
    )
    val_already_done = _is_split_already_processed(
        output_path=val_output,
        split_name="val",
        expected_rows=None,
        require_done_marker=False
    )

    if train_already_done and val_already_done:
        logger.info(
            "Preprocessed outputs already exist and are valid. "
            "Skipping preprocessing."
        )
        logger.info(f"Using existing train file: {train_output}")
        logger.info(f"Using existing val file: {val_output}")
        logger.info("Preprocessing Completed Successfully.")
        sys.exit(0)

    if train_already_done:
        logger.info(f"Train preprocessed file already ready. Skipping: {train_output}")
    else:
        logger.info("--- Starting TRAIN set processing ---")
        process_data_split(
            train_input,
            train_output,
            extractor,
            split_name="train",
            checkpoint_every=max(1, int(args.checkpoint_every)),
            enable_resume_checkpoints=True,
            expected_rows=expected_train_rows
        )
    
    if val_already_done:
        logger.info(f"Val preprocessed file already ready. Skipping: {val_output}")
    else:
        logger.info("--- Starting VAL set processing ---")
        process_data_split(
            val_input,
            val_output,
            extractor,
            split_name="val",
            checkpoint_every=max(1, int(args.checkpoint_every)),
            enable_resume_checkpoints=False
        )
    
    logger.info("Preprocessing Completed Successfully.")
