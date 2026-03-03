import os
import sys
import shutil
import re
import json
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import torch
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from dotenv import load_dotenv
from comet_ml import Experiment
from sklearn.metrics import confusion_matrix
from pytorch_metric_learning import losses

from src.src_TaskA.models.model import HybridClassifier
from src.src_TaskA.dataset.dataset import load_data
from src.src_TaskB.utils.utils import set_seed, evaluate_model

# -----------------------------------------------------------------------------
# 1. SETUP & UTILS
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    """Gestisce l'output formattato per monitorare il training."""
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        priority_keys = ["loss", "f1_macro", "acc", "task_loss", "supcon_loss"]
        
        for k in priority_keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        
        for k, v in metrics.items():
            if k not in priority_keys:
                if isinstance(v, float):
                    log_str += f"{k}: {v:.4f} | "
        
        logger.info(log_str.strip(" | "))

def save_checkpoint(model, tokenizer, path, epoch, metrics, config):
    """Salva stato modello, tokenizer e configurazione."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving checkpoint to {path}...")
    
    tokenizer.save_pretrained(path)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(path, "model_state.bin"))
    
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.dump({"epoch": epoch, "metrics": metrics}, f)

    checkpoint_type = None
    global_samples_seen = None
    if isinstance(metrics, dict):
        checkpoint_type = metrics.get("checkpoint_type", None)
        global_samples_seen = metrics.get("global_samples_seen", None)

    context_parts = []
    if checkpoint_type is not None:
        context_parts.append(f"type={checkpoint_type}")
    if epoch is not None:
        context_parts.append(f"epoch={epoch}")
    if global_samples_seen is not None:
        context_parts.append(f"global_samples_seen={global_samples_seen}")

    if context_parts:
        print(f"Checkpoint saved at: {path} | {' | '.join(context_parts)}")
    else:
        print(f"Checkpoint saved at: {path}")

def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _extract_samples_from_path(path):
    match = re.search(r"samples_(\d+)", path.replace("\\", "/"))
    return _safe_int(match.group(1), 0) if match else 0

def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint available in checkpoint_dir recursively.
    Ranking priority:
    1) global_samples_seen (if available)
    2) epoch
    3) step
    4) file mtime
    """
    if not os.path.exists(checkpoint_dir):
        return None

    candidates = []
    for root, _, files in os.walk(checkpoint_dir):
        if "model_state.bin" not in files:
            continue

        model_path = os.path.join(root, "model_state.bin")
        meta_path = os.path.join(root, "training_meta.yaml")

        epoch = 0
        step = 0
        global_samples_seen = 0
        f1_macro = None

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f) or {}
                epoch = _safe_int(meta.get("epoch", 0), 0)
                metric_obj = meta.get("metrics", {}) or {}
                global_samples_seen = _safe_int(
                    metric_obj.get("global_samples_seen", 0), 0
                )
                step = _safe_int(metric_obj.get("step", 0), 0)
                f1_macro = metric_obj.get("f1_macro", None)
            except Exception:
                pass

        if global_samples_seen == 0:
            global_samples_seen = _extract_samples_from_path(root)

        candidates.append({
            "path": root,
            "epoch": epoch,
            "step": step,
            "global_samples_seen": global_samples_seen,
            "f1_macro": f1_macro,
            "mtime": os.path.getmtime(model_path)
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: (
            x["global_samples_seen"],
            x["epoch"],
            x["step"],
            x["mtime"]
        )
    )


def _get_parquet_row_count(parquet_path):
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        return None


def verify_train_preprocessed_ready(data_dir, expected_train_rows=500000):
    train_processed_path = os.path.join(data_dir, "train_processed.parquet")
    done_marker_path = os.path.join(data_dir, "train_processed.done.json")

    if not os.path.exists(train_processed_path):
        logger.error(
            f"Missing preprocessed train file: {train_processed_path}. "
            f"Run preprocess first."
        )
        return False

    if not os.path.exists(done_marker_path):
        logger.error(
            f"Missing completion marker: {done_marker_path}. "
            f"Preprocess may still be incomplete."
        )
        return False

    try:
        with open(done_marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception as e:
        logger.error(f"Cannot read completion marker {done_marker_path}: {e}")
        return False

    if not marker.get("complete", False):
        logger.error("train_processed.done.json indicates preprocessing is not complete yet.")
        return False

    row_count = _get_parquet_row_count(train_processed_path)
    if row_count is None:
        logger.error(
            f"Could not read row count from {train_processed_path}. "
            f"Please check parquet integrity."
        )
        return False

    if expected_train_rows and row_count != expected_train_rows:
        logger.error(
            f"train_processed row count mismatch: got {row_count}, expected {expected_train_rows}. "
            f"Training will not start."
        )
        return False

    logger.info(
        f"Preprocessed train data ready: {train_processed_path} "
        f"(rows={row_count}, expected={expected_train_rows})"
    )
    return True

# -----------------------------------------------------------------------------
# 2. TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device,
                   epoch_idx, acc_steps=1, supcon_fn=None,
                   global_samples_seen=0, next_ckpt_at=None,
                   periodic_every_samples=None, periodic_ckpt_callback=None):
    model.train()
    
    tracker = {"loss": 0.0, "task_loss": 0.0, "supcon_loss": 0.0}
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(dataloader, desc=f"Train Ep {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        feats = batch["extra_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            _, task_loss, combined_features = model(
                input_ids, attention_mask, feats, labels=labels
            )
            
            supcon_loss = torch.tensor(0.0, device=device)
            if supcon_fn is not None:
                features_norm = torch.nn.functional.normalize(combined_features, dim=1)
                supcon_loss = supcon_fn(features_norm, labels)
            
            total_loss = (task_loss + 0.1 * supcon_loss) / acc_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % acc_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                scheduler.step()
        
        current_loss = total_loss.item() * acc_steps
        tracker["loss"] += current_loss
        tracker["task_loss"] += task_loss.item()
        tracker["supcon_loss"] += supcon_loss.item()

        batch_size_actual = labels.size(0)
        global_samples_seen += batch_size_actual

        if (
            periodic_ckpt_callback is not None
            and periodic_every_samples is not None
            and periodic_every_samples > 0
            and next_ckpt_at is not None
        ):
            while global_samples_seen >= next_ckpt_at:
                processed_batches = step + 1
                running_train_metrics = {
                    k: (v / processed_batches) for k, v in tracker.items()
                }
                periodic_ckpt_callback(
                    checkpoint_samples=next_ckpt_at,
                    global_samples_seen=global_samples_seen,
                    epoch_idx=epoch_idx,
                    step_idx=step,
                    running_train_metrics=running_train_metrics
                )
                next_ckpt_at += periodic_every_samples
        
        pbar.set_postfix({
            "Loss": f"{current_loss:.3f}",
            "SupCon": f"{supcon_loss.item():.3f}" if supcon_fn else "0.0"
        })

    num_batches = len(dataloader)
    return (
        {k: v / num_batches for k, v in tracker.items()},
        global_samples_seen,
        next_ckpt_at
    )

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="SemEval Task A - Generalization Training")
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml")
    parser.add_argument("--expected-train-rows", type=int, default=500000)
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval Task 13 - Subtask A [Generalization]")

    # 1. Configurazione
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
        
    config = raw_config
    common_cfg = config.get("common", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    checkpoint_dir = train_cfg["checkpoint_dir"]
    auto_resume_latest = train_cfg.get("auto_resume_latest_checkpoint", True)

    set_seed(common_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 2. Comet ML Tracking
    api_key = os.getenv("COMET_API_KEY")
    experiment = None
    if api_key:
        try:
            experiment = Experiment(
                api_key=api_key,
                project_name=common_cfg.get("project_name", "semeval-task-a"),
                auto_metric_logging=False
            )
            experiment.log_parameters(config)
            experiment.add_tag("TaskA_Hybrid")
        except Exception as e:
            logger.warning(f"Comet Init Failed: {e}. Proceeding without it.")

    # 3. Resume Checkpoint Discovery (optional)
    resume_info = None
    if auto_resume_latest:
        resume_info = find_latest_checkpoint(checkpoint_dir)
        if resume_info:
            logger.info(
                f"Auto-resume enabled. Found latest checkpoint: {resume_info['path']} "
                f"(epoch={resume_info['epoch']}, samples={resume_info['global_samples_seen']})"
            )

    # 4. Data Loading
    if not verify_train_preprocessed_ready(
        data_cfg["data_dir"],
        expected_train_rows=max(0, int(args.expected_train_rows))
    ):
        sys.exit(1)

    tokenizer_load_source = model_cfg["base_model"]
    if resume_info:
        tokenizer_cfg_path = os.path.join(resume_info["path"], "tokenizer_config.json")
        if os.path.exists(tokenizer_cfg_path):
            tokenizer_load_source = resume_info["path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_source)
    
    logger.info("Initializing Datasets...")
    train_ds, val_ds = load_data(config, tokenizer)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=train_cfg["batch_size"], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=train_cfg["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 5. Model Setup
    model = HybridClassifier(config)
    if resume_info:
        weights_path = os.path.join(resume_info["path"], "model_state.bin")
        logger.info(f"Loading model weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    
    # 6. Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=float(train_cfg["learning_rate"]), 
        weight_decay=train_cfg.get("weight_decay", 0.01)
    )
    
    # SupCon Loss
    supcon_fn = None
    if train_cfg.get("use_supcon", False):
        logger.info("Activating Supervised Contrastive Loss...")
        supcon_fn = losses.SupConLoss(temperature=0.1).to(device)

    scaler = GradScaler()
    acc_steps = train_cfg.get("gradient_accumulation_steps", 1)
    total_steps = (len(train_dl) // acc_steps) * train_cfg["num_epochs"]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(train_cfg["learning_rate"]), 
        total_steps=total_steps,
        pct_start=0.1
    )

    # 7. Training Loop
    best_f1 = 0.0
    start_epoch = 0
    if resume_info:
        start_epoch = _safe_int(resume_info.get("epoch", 0), 0) + 1
        resumed_best_f1 = resume_info.get("f1_macro", None)
        if resumed_best_f1 is not None:
            try:
                best_f1 = float(resumed_best_f1)
            except (TypeError, ValueError):
                pass

    patience = train_cfg.get("early_stop_patience", 3)
    patience_counter = 0
    periodic_enabled = train_cfg.get("enable_periodic_checkpoint", True)
    periodic_every_samples = int(train_cfg.get("periodic_checkpoint_every_samples", 5000))
    periodic_max_to_keep = train_cfg.get("periodic_checkpoint_max_to_keep", None)
    if periodic_max_to_keep is not None:
        periodic_max_to_keep = int(periodic_max_to_keep)
    if periodic_every_samples <= 0:
        logger.warning("Invalid periodic_checkpoint_every_samples <= 0. Disabling periodic checkpoints.")
        periodic_enabled = False

    global_samples_seen = _safe_int(
        resume_info.get("global_samples_seen", 0), 0
    ) if resume_info else 0
    if periodic_enabled:
        next_ckpt_at = ((global_samples_seen // periodic_every_samples) + 1) * periodic_every_samples
    else:
        next_ckpt_at = None
    periodic_saved_paths = []
    
    label_names = ["Human", "AI"] 

    if start_epoch >= train_cfg["num_epochs"]:
        logger.info(
            f"Resume checkpoint epoch ({start_epoch}) already reached/exceeded num_epochs "
            f"({train_cfg['num_epochs']}). Nothing to train."
        )
        if experiment:
            experiment.end()
        sys.exit(0)

    logger.info(
        f"Starting Training for {train_cfg['num_epochs']} epochs "
        f"(start_epoch={start_epoch + 1}, global_samples_seen={global_samples_seen})..."
    )

    def save_periodic_checkpoint(checkpoint_samples, global_samples_seen, epoch_idx, step_idx, running_train_metrics):
        periodic_dir = os.path.join(checkpoint_dir, "periodic", f"samples_{checkpoint_samples}")
        periodic_meta = {
            "checkpoint_type": "periodic",
            "epoch": epoch_idx,
            "step": step_idx,
            "global_samples_seen": global_samples_seen,
            "last_train_metrics": running_train_metrics
        }
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            path=periodic_dir,
            epoch=epoch_idx,
            metrics=periodic_meta,
            config=config
        )
        logger.info(f"Periodic checkpoint saved at {checkpoint_samples} samples: {periodic_dir}")

        periodic_saved_paths.append(periodic_dir)
        if periodic_max_to_keep and periodic_max_to_keep > 0:
            while len(periodic_saved_paths) > periodic_max_to_keep:
                old_path = periodic_saved_paths.pop(0)
                if os.path.exists(old_path):
                    shutil.rmtree(old_path, ignore_errors=True)
                    logger.info(f"Removed old periodic checkpoint due to max_to_keep={periodic_max_to_keep}: {old_path}")

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        
        # --- TRAIN ---
        train_metrics, global_samples_seen, next_ckpt_at = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, acc_steps, supcon_fn,
            global_samples_seen=global_samples_seen,
            next_ckpt_at=next_ckpt_at,
            periodic_every_samples=periodic_every_samples if periodic_enabled else None,
            periodic_ckpt_callback=save_periodic_checkpoint if periodic_enabled else None
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # --- VALIDATION ---
        val_metrics, val_preds, val_labels, report = evaluate_model(model, val_dl, device, label_names)
        
        ConsoleUX.log_metrics("Val", val_metrics)
        logger.info(f"\n{report}")
        
        if experiment:
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # --- CHECKPOINTING ---
        current_f1 = val_metrics["f1_macro"]
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            logger.info(f"--> New Best F1: {best_f1:.4f}. Saving Model...")
            
            save_checkpoint(
                model, tokenizer, 
                os.path.join(checkpoint_dir, "best_model"), 
                epoch, val_metrics, config
            )
            
            if experiment:
                cm = confusion_matrix(val_labels, val_preds)
                experiment.log_confusion_matrix(matrix=cm, labels=label_names, title="Best Model CM")
                
        else:
            patience_counter += 1
            logger.warning(f"--> No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break

    if experiment:
        experiment.end()
    
    logger.info("Training Finished.")
