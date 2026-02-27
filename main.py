import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import math, os, time, logging, json, sys, gc
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# LOGGING
# ================================================================
log_dir = r'D:\base\mullti\ids_results'
os.makedirs(log_dir, exist_ok=True)

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger = logging.getLogger('IDS')
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
for _h in [logging.FileHandler(f'{log_dir}/training_log.txt', mode='w'), FlushHandler(sys.stdout)]:
    _h.setFormatter(_fmt); logger.addHandler(_h)
logger.propagate = False

def L(msg):  logger.info(f"  {msg}");  sys.stdout.flush()
def LS(title):
    logger.info("\n" + "="*65 + f"\n  {title}\n" + "="*65)
    sys.stdout.flush()
def ram_log():
    try:
        import psutil
        r = psutil.virtual_memory()
        L(f"RAM  used={r.used/1e9:.1f}GB  free={r.available/1e9:.1f}GB  "
          f"total={r.total/1e9:.1f}GB  ({r.percent}%)")
    except: pass

# ================================================================
# TIMER
# ================================================================
class Timer:
    def __init__(self): self.r = {}
    def start(self, n): self.r[n]={'start':time.time()}; L(f"[TIMER ▶] {n}")
    def stop(self, n):
        e=time.time()-self.r[n]['start']; self.r[n]['elapsed']=e
        L(f"[TIMER ■] {n}  →  {e:.1f}s  ({e/60:.2f} min)"); return e
    def save(self, p):
        with open(p,'w') as f:
            json.dump({k:round(v.get('elapsed',0),3) for k,v in self.r.items()},f,indent=2)

timer = Timer()
torch.manual_seed(42); np.random.seed(42)

# ================================================================
# STEP 1 — SCAN FILES
# ================================================================
def scan_files(dataset_dir, chunksize=25_000):      # ← 25k chunks for 4GB RAM
    LS("STEP 1 : SCANNING CSV FILES")
    timer.start('scan')

    csv_files = sorted([os.path.join(dataset_dir, f)
                        for f in os.listdir(dataset_dir) if f.endswith('.csv')])
    L(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        L(f"  {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.1f} MB)")

    feature_names = None
    label_set     = set()
    total_rows    = 0

    for fi, fpath in enumerate(csv_files):
        fname = os.path.basename(fpath)
        for ci, chunk in enumerate(pd.read_csv(fpath, chunksize=chunksize,
                                                low_memory=False,
                                                dtype=np.float32 if False else None)):
            if feature_names is None:
                feature_names = chunk.columns[:-1].tolist()
                L(f"  Schema: {len(feature_names)} features | "
                  f"label col: '{chunk.columns[-1]}'")
            label_set.update(chunk.iloc[:, -1].astype(str).unique())
            total_rows += len(chunk)
            L(f"  Scan  file {fi+1}/{len(csv_files)}  "
              f"chunk {ci+1}  |  rows so far: {total_rows:,}")
            del chunk; gc.collect()

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(label_set))
    num_classes = len(label_encoder.classes_)
    L(f"Total rows: {total_rows:,}  |  "
      f"Classes ({num_classes}): {list(label_encoder.classes_)}")
    ram_log()
    timer.stop('scan')
    return csv_files, feature_names, label_encoder, num_classes, total_rows

# ================================================================
# STEP 2 — FIT SCALER
# ================================================================
def fit_scaler(csv_files, chunksize=25_000):
    LS("STEP 2 : FITTING SCALER")
    timer.start('scaler')
    scaler = StandardScaler()
    total_chunks = 0

    for fi, fpath in enumerate(csv_files):
        for ci, chunk in enumerate(pd.read_csv(fpath, chunksize=chunksize,
                                                low_memory=False)):
            X = chunk.iloc[:, :-1].values.astype(np.float32)
            scaler.partial_fit(X)
            total_chunks += 1
            L(f"  Scaler  file {fi+1}/{len(csv_files)}  "
              f"chunk {ci+1}  |  total chunks: {total_chunks}")
            del X, chunk; gc.collect()

    ram_log()
    timer.stop('scaler')
    return scaler

# ================================================================
# STEP 3 — WRITE MEMORY-MAPPED FILES
# ================================================================
def write_mmap(csv_files, scaler, label_encoder, total_rows,
               n_features, out_dir, chunksize=25_000):
    LS("STEP 3 : WRITING MEMORY-MAPPED FILES")
    timer.start('mmap_write')

    X_path = os.path.join(out_dir, 'X_mmap.npy')
    y_path = os.path.join(out_dir, 'y_mmap.npy')

    X_mm = np.lib.format.open_memmap(X_path, mode='w+',
                                      dtype=np.float32,
                                      shape=(total_rows, n_features))
    y_mm = np.lib.format.open_memmap(y_path, mode='w+',
                                      dtype=np.int32,
                                      shape=(total_rows,))
    L(f"  Pre-allocated  X: {X_mm.nbytes/1e6:.0f} MB  "
      f"y: {y_mm.nbytes/1e6:.0f} MB  (on disk, not in RAM)")

    row_ptr = 0
    for fi, fpath in enumerate(csv_files):
        for ci, chunk in enumerate(pd.read_csv(fpath, chunksize=chunksize,
                                                low_memory=False)):
            n       = len(chunk)
            X_chunk = scaler.transform(
                          chunk.iloc[:, :-1].values.astype(np.float32))
            y_chunk = label_encoder.transform(
                          chunk.iloc[:, -1].astype(str).values).astype(np.int32)
            X_mm[row_ptr:row_ptr+n] = X_chunk
            y_mm[row_ptr:row_ptr+n] = y_chunk
            row_ptr += n
            pct = 100*row_ptr/total_rows
            L(f"  Write  file {fi+1}/{len(csv_files)}  chunk {ci+1}  |  "
              f"{row_ptr:,}/{total_rows:,}  ({pct:.1f}%)")
            del X_chunk, y_chunk, chunk; gc.collect()

    # flush & close mmap before use
    X_mm.flush(); y_mm.flush()
    del X_mm, y_mm; gc.collect()
    ram_log()
    timer.stop('mmap_write')
    L(f"  Saved →  {X_path}")
    L(f"  Saved →  {y_path}")
    return X_path, y_path

# ================================================================
# MMAP DATASET
# ================================================================
class MmapDataset(Dataset):
    def __init__(self, X_path, y_path, indices):
        # mmap_mode='r' = file stays on disk, only reads needed rows
        self.X       = np.load(X_path, mmap_mode='r')
        self.y       = np.load(y_path, mmap_mode='r')
        self.indices = indices

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (torch.tensor(np.array(self.X[idx]), dtype=torch.float32),
                torch.tensor(int(self.y[idx]),       dtype=torch.long))

# ================================================================
# STEP 4 — BUILD DATALOADERS
# ================================================================
def build_loaders(X_path, y_path, total_rows,
                  batch_size=256, test_size=0.2, val_size=0.1):
    LS("STEP 4 : BUILDING DATALOADERS")
    timer.start('loaders')

    idx     = np.random.permutation(total_rows)
    test_n  = int(total_rows * test_size)
    val_n   = int(total_rows * val_size)
    train_n = total_rows - test_n - val_n

    train_idx = idx[:train_n]
    val_idx   = idx[train_n:train_n+val_n]
    test_idx  = idx[train_n+val_n:]
    L(f"  Train: {train_n:,}  |  Val: {val_n:,}  |  Test: {test_n:,}")

    def make_loader(indices, shuffle):
        ds = MmapDataset(X_path, y_path, indices)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = 0,        # ← MUST be 0 on Windows / Jupyter
            pin_memory  = False     # ← False when num_workers=0
        )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    ram_log()
    timer.stop('loaders')
    return train_loader, val_loader, test_loader

# ================================================================
# MODEL  (lightweight for 4 GB RAM)
# ================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', torch.zeros(1, 1, d_model))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class TransformerIDSClassifier(nn.Module):
    def __init__(self, input_dim, num_classes,
                 d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embed   = nn.Linear(input_dim, d_model)
        self.pos     = PositionalEncoding(d_model, dropout)
        self.norm0   = nn.LayerNorm(d_model)
        enc_layer    = nn.TransformerEncoderLayer(
                           d_model=d_model, nhead=nhead,
                           dim_feedforward=dim_feedforward,
                           dropout=dropout, activation='relu',
                           batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc1     = nn.Linear(d_model, dim_feedforward)
        self.drop    = nn.Dropout(dropout)
        self.fc2     = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x):
        x = self.norm0(self.pos(self.embed(x).unsqueeze(1)))
        x = self.encoder(x).squeeze(1)
        return self.fc2(self.drop(torch.relu(self.fc1(x))))

    def feature_importance(self, x):
        x = x.detach().requires_grad_(True)
        out = self(x)
        out.sum().backward()
        return x.grad.abs().cpu().numpy()

# ================================================================
# STEP 5 — TRAINING
# ================================================================
def train_model(model, train_loader, val_loader,
                num_epochs=50, lr=0.001, device='cpu'):
    LS("STEP 5 : TRAINING")
    timer.start('training')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc, best_state = 0.0, None
    train_losses, val_accs, epoch_times, lr_hist = [], [], [], []
    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for bi, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item()
            t_total   += y.size(0)
            t_correct += out.argmax(1).eq(y).sum().item()

            # print every 25% of batches
            if (bi+1) % max(1, total_batches//4) == 0:
                L(f"  Epoch {epoch+1:>3}/{num_epochs} | "
                  f"Batch {bi+1:>5}/{total_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100*t_correct/t_total:.2f}%")

        avg_loss  = t_loss / total_batches
        train_acc = 100 * t_correct / t_total
        train_losses.append(avg_loss)

        # ── validation ──
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                v_correct += model(X).argmax(1).eq(y).sum().item()
                v_total   += y.size(0)

        val_acc = 100 * v_correct / v_total
        val_accs.append(val_acc)
        scheduler.step(val_acc)
        cur_lr     = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0
        lr_hist.append(cur_lr)
        epoch_times.append(epoch_time)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        tag = "  ◀ BEST" if is_best else ""
        L(f"  EPOCH {epoch+1:>3}/{num_epochs} | "
          f"Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | "
          f"Val: {val_acc:.2f}% | LR: {cur_lr:.6f} | "
          f"Time: {epoch_time:.1f}s{tag}")
        ram_log()
        gc.collect()   # free any lingering tensors each epoch

    timer.stop('training')
    L(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    model.load_state_dict(best_state)
    return model, train_losses, val_accs, epoch_times, lr_hist

# ================================================================
# STEP 6 — EVALUATION
# ================================================================
def evaluate_model(model, test_loader, label_encoder, device):
    LS("STEP 6 : EVALUATION")
    timer.start('evaluation')
    model.eval()
    preds, labels = [], []
    total = len(test_loader)

    with torch.no_grad():
        for bi, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            preds.extend(model(X).argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            if (bi+1) % max(1, total//5) == 0:
                L(f"  Eval batch {bi+1}/{total}  "
                  f"({100*(bi+1)/total:.0f}%)")

    timer.stop('evaluation')
    acc         = accuracy_score(labels, preds)
    p,r,f1,_    = precision_recall_fscore_support(labels, preds,
                      average='weighted', zero_division=0)
    pm,rm,f1m,_ = precision_recall_fscore_support(labels, preds,
                      average='macro', zero_division=0)

    LS("TEST RESULTS")
    L(f"Accuracy           : {acc*100:.2f}%")
    L(f"Weighted Precision : {p*100:.2f}%  "
      f"Recall: {r*100:.2f}%  F1: {f1*100:.2f}%")
    L(f"Macro    Precision : {pm*100:.2f}%  "
      f"Recall: {rm*100:.2f}%  F1: {f1m*100:.2f}%")
    logger.info("\n" + classification_report(
        labels, preds,
        target_names=label_encoder.classes_, zero_division=0))
    sys.stdout.flush()
    return acc, p, r, f1, labels, preds

# ================================================================
# STEP 7 — CHARTS
# ================================================================
def save_charts(train_losses, val_accs, epoch_times, lr_hist,
                labels, preds, label_encoder,
                model, test_loader, feature_names, device, out_dir):
    LS("STEP 7 : SAVING RESEARCH CHARTS")
    timer.start('charting')
    classes = label_encoder.classes_

    # 1. Training dashboard
    L("  Chart 1/6 — Training Dashboard ...")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Transformer IDS — Training Dashboard',
                 fontsize=14, fontweight='bold')
    ax[0,0].plot(train_losses, color='steelblue')
    ax[0,0].set_title('Train Loss'); ax[0,0].grid(alpha=.3)
    ax[0,1].plot(val_accs, color='darkorange')
    ax[0,1].axhline(max(val_accs), ls='--', color='red', alpha=.5,
                    label=f'Best {max(val_accs):.2f}%')
    ax[0,1].set_title('Val Accuracy')
    ax[0,1].legend(); ax[0,1].grid(alpha=.3)
    ax[1,0].plot(epoch_times, color='green')
    ax[1,0].axhline(np.mean(epoch_times), ls='--', color='red',
                    label=f'Mean {np.mean(epoch_times):.1f}s')
    ax[1,0].set_title('Epoch Time'); ax[1,0].legend(); ax[1,0].grid(alpha=.3)
    ax[1,1].plot(lr_hist, color='purple')
    ax[1,1].set_yscale('log')
    ax[1,1].set_title('LR Schedule'); ax[1,1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/training_dashboard.png', dpi=100, bbox_inches='tight')
    plt.close(); gc.collect()
    L("  Saved: training_dashboard.png")

    # 2. Confusion matrix
    L("  Chart 2/6 — Confusion Matrix ...")
    cm  = confusion_matrix(labels, preds)
    sz  = max(8, len(classes))
    fig, ax = plt.subplots(figsize=(sz, sz-1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close(); gc.collect()
    L("  Saved: confusion_matrix.png")

    # 3. Normalized confusion matrix
    L("  Chart 3/6 — Normalized Confusion Matrix ...")
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(sz, sz-1))
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Normalized Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/confusion_matrix_normalized.png',
                dpi=100, bbox_inches='tight')
    plt.close(); gc.collect()
    L("  Saved: confusion_matrix_normalized.png")

    # 4. Per-class metrics
    L("  Chart 4/6 — Per-Class Metrics ...")
    rpt = classification_report(labels, preds, target_names=classes,
                                output_dict=True, zero_division=0)
    x = np.arange(len(classes)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(classes)*1.3), 5))
    for i,(m,k) in enumerate([('Precision','precision'),
                               ('Recall','recall'),
                               ('F1','f1-score')]):
        ax.bar(x+i*w, [rpt[c][k] for c in classes], w, label=m, alpha=0.85)
    ax.set_xticks(x+w)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=.3)
    ax.set_title('Per-Class Precision / Recall / F1', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/per_class_metrics.png', dpi=100, bbox_inches='tight')
    plt.close(); gc.collect()
    L("  Saved: per_class_metrics.png")

    # 5. Feature importance
    L("  Chart 5/6 — Feature Importance ...")
    model.eval()
    X_s, _ = next(iter(test_loader))
    att    = model.feature_importance(X_s[:8].to(device))
    mean_a = att.mean(0)
    top_n  = min(20, len(feature_names))
    top_i  = np.argsort(mean_a)[-top_n:][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), mean_a[top_i][::-1], color='teal', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_i][::-1], fontsize=9)
    ax.set_title(f'Top {top_n} Feature Importances', fontweight='bold')
    ax.grid(axis='x', alpha=.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close(); gc.collect()
    L("  Saved: feature_importance.png")

    # 6. Timing summary
    L("  Chart 6/6 — Pipeline Timing ...")
    td = {k: v.get('elapsed', 0) for k, v in timer.r.items()}
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['steelblue','orange','green','purple','crimson','teal','gray']
    bars   = ax.bar(td.keys(), [v/60 for v in td.values()],
                    color=colors[:len(td)])
    for b, v in zip(bars, td.values()):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{v/60:.2f}m', ha='center', fontsize=8)
    ax.set_ylabel('Minutes')
    ax.set_title('Pipeline Stage Timing', fontweight='bold')
    ax.grid(axis='y', alpha=.3)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/timing_summary.png', dpi=100, bbox_inches='tight')
    plt.close(); gc.collect()
    L("  Saved: timing_summary.png")

    timer.stop('charting')
    LS(f"ALL CHARTS SAVED  →  {out_dir}/")

# ================================================================
# MAIN
# ================================================================
def main():
    LS("TRANSFORMER IDS — EXPERIMENT START")

    # ── config ── ONLY CHANGE THESE TWO PATHS ───────────────────
    DATASET_DIR = r'D:\base\6g-iot-security\data'   # ← your CSV folder
    # log_dir already set at top of file

    BATCH_SIZE  = 256       # ← reduced from 512  (saves ~500MB RAM)
    NUM_EPOCHS  = 50
    LR          = 0.001
    D_MODEL     = 64
    NHEAD       = 4
    NUM_LAYERS  = 2
    FF_DIM      = 128
    DROPOUT     = 0.1
    CHUNKSIZE   = 25_000    # ← reduced from 50k  (saves ~300MB RAM)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L(f"Device : {device}")
    if torch.cuda.is_available():
        L(f"GPU : {torch.cuda.get_device_name(0)}  "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    ram_log()
    timer.start('total')

    # STEP 1 — scan
    csv_files, feature_names, label_encoder, num_classes, total_rows = \
        scan_files(DATASET_DIR, chunksize=CHUNKSIZE)

    # STEP 2 — scaler
    scaler = fit_scaler(csv_files, chunksize=CHUNKSIZE)

    # STEP 3 — mmap
    X_path, y_path = write_mmap(
        csv_files, scaler, label_encoder,
        total_rows, len(feature_names),
        log_dir, chunksize=CHUNKSIZE)

    # STEP 4 — loaders
    train_loader, val_loader, test_loader = build_loaders(
        X_path, y_path, total_rows, batch_size=BATCH_SIZE)

    # STEP 5 — model
    LS("STEP 5 : MODEL INIT")
    model = TransformerIDSClassifier(
        input_dim       = len(feature_names),
        num_classes     = num_classes,
        d_model         = D_MODEL,
        nhead           = NHEAD,
        num_layers      = NUM_LAYERS,
        dim_feedforward = FF_DIM,
        dropout         = DROPOUT
    ).to(device)
    L(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ram_log()

    # STEP 6 — train
    model, train_losses, val_accs, epoch_times, lr_hist = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LR, device=device)

    # save checkpoint
    ckpt = f'{log_dir}/transformer_ids_model.pth'
    torch.save({
        'model_state_dict' : model.state_dict(),
        'label_encoder'    : label_encoder,
        'scaler'           : scaler,
        'input_dim'        : len(feature_names),
        'num_classes'      : num_classes
    }, ckpt)
    L(f"Model saved → {ckpt}")

    # STEP 7 — evaluate
    acc, p, r, f1, all_labels, all_preds = evaluate_model(
        model, test_loader, label_encoder, device)

    # timings
    timer.stop('total')
    timer.save(f'{log_dir}/timing.json')

    # STEP 8 — charts
    save_charts(train_losses, val_accs, epoch_times, lr_hist,
                all_labels, all_preds, label_encoder,
                model, test_loader, feature_names, device, log_dir)

    LS("EXPERIMENT COMPLETE")
    L(f"Accuracy      : {acc*100:.2f}%")
    L(f"F1 (weighted) : {f1*100:.2f}%")
    L(f"All outputs   : {log_dir}/")
    sys.stdout.flush()

main()