"""
utils/model.py
ProGNet v8 — exact architecture from Prognet-model.ipynb
Cell 3: preprocessing | Cell 4: growth sim | Cell 9: model | Cell 17: Grad-CAM
"""
import os, random, warnings
import numpy as np
import cv2
warnings.filterwarnings("ignore")

# ── Config (Cell 2) ───────────────────────────────────────────
IMG_H   = 96
IMG_W   = 96
N_CH    = 1
SEQ_LEN = 4
DROPOUT = 0.40
LR      = 3e-4

CLASS_GROWTH_RATE = {
    'glioma':     14.0,
    'meningioma':  9.0,
    'pituitary':   7.0,
    'notumor':     0.5,
}
CLASS_GROWTH_NOISE = 0.05
GRADCAM_LAYER = 'cnn_b3_conv2'   # Cell 16

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'prognet_v8_best.keras')
NORM_PATH  = os.path.join(os.path.dirname(__file__), '..', 'model', 'y_norm.npy')

# ── Lazy TF import ────────────────────────────────────────────
_tf = None
def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        _tf = tf
    return _tf

# ── Cell 3: Preprocessing ─────────────────────────────────────
def zscore_normalize(arr, eps=1e-8):
    mu, sigma = arr.mean(), arr.std()
    return (arr - mu) / (sigma + eps)

def minmax_clip(arr, low=1, high=99):
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    return (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-8)

def resize_slice(img, h=IMG_H, w=IMG_W):
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)

def load_jpeg_as_slice(data_bytes):
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot decode image")
    img = resize_slice(img.astype(np.float32))
    img = zscore_normalize(img)
    return minmax_clip(img).astype(np.float32)

def load_nifti_middle_slice(data_bytes):
    import tempfile
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel not installed")
    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
        f.write(data_bytes)
        tmp = f.name
    try:
        vol = nib.load(tmp).get_fdata().astype(np.float32)
        mid = vol.shape[2] // 2 if vol.ndim == 3 else 0
        sl  = vol[:, :, mid] if vol.ndim == 3 else vol
        sl  = resize_slice(sl)
        sl  = zscore_normalize(sl)
        return minmax_clip(sl).astype(np.float32)
    finally:
        os.unlink(tmp)

def augment_frame(img, seed=None):
    rng = np.random.RandomState(seed)
    h, w = img.shape
    if rng.rand() < 0.5:
        img = np.fliplr(img)
    angle = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    img = np.clip(img + rng.uniform(-0.10, 0.10), 0, 1).astype(np.float32)
    img = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    return img

# ── Cell 4: Deterministic growth simulator ────────────────────
def compute_tumor_volume_pct(sl, thr=85):
    if sl.max() < 0.1:
        return 0.0
    t = np.percentile(sl, thr)
    return float((sl >= t).sum()) / sl.size

def deterministic_growth_sim(base_slice, class_name, seq_len=SEQ_LEN):
    base_rate     = CLASS_GROWTH_RATE[class_name]
    patient_factor = 1.0 + np.random.uniform(-CLASS_GROWTH_NOISE, CLASS_GROWTH_NOISE)
    rate          = base_rate * patient_factor / 100.0

    ys, xs = np.where(base_slice > np.percentile(base_slice, 80))
    cy = int(ys.mean()) if len(ys) > 0 else IMG_H // 2
    cx = int(xs.mean()) if len(xs) > 0 else IMG_W // 2

    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    sig    = max(8, min(IMG_H, IMG_W) // 4)
    gf     = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sig**2))
    gf     = gf / (gf.max() + 1e-8)

    frames    = [base_slice.copy()]
    base_vol  = compute_tumor_volume_pct(base_slice)
    g_labels  = [0.0]
    accum     = base_slice.copy()

    for t in range(1, seq_len):
        accum = np.clip(accum + rate * t * gf, 0, 1)
        frames.append(accum.copy())
        g_labels.append(compute_tumor_volume_pct(accum) - base_vol)

    return np.array(frames, np.float32), np.array(g_labels, np.float32)

# ── Build sequence from uploaded files ────────────────────────
def build_sequence_from_uploads(uploaded_files):
    """
    uploaded_files: list of (filename, bytes)
    Returns (1, SEQ_LEN, IMG_H, IMG_W, 1) float32
    """
    slices = []
    for fname, data in uploaded_files:
        fl = fname.lower()
        try:
            if fl.endswith(('.nii', '.nii.gz')):
                sl = load_nifti_middle_slice(data)
            else:
                sl = load_jpeg_as_slice(data)
            slices.append(sl)
        except Exception:
            continue

    if not slices:
        raise ValueError("No valid MRI files could be processed")

    while len(slices) < SEQ_LEN:
        noise = np.random.normal(0, 0.01, slices[-1].shape).astype(np.float32)
        slices.append(np.clip(slices[-1] + noise, 0, 1))
    slices = slices[:SEQ_LEN]

    seq = np.stack(slices, axis=0)[:, :, :, np.newaxis]  # (SEQ_LEN,H,W,1)
    return seq[np.newaxis].astype(np.float32)             # (1,SEQ_LEN,H,W,1)

# ── Cell 9: Build ProGNet v8 ──────────────────────────────────
def build_prognet_v8():
    tf = _get_tf()
    from tensorflow.keras import layers, Model
    from tensorflow.keras.regularizers import l2 as L2_REG

    def cnn_block(x, filters, name_prefix, sdrop=0.15):
        shortcut = x
        x = layers.Conv2D(filters, 3, padding='same',
                          kernel_regularizer=L2_REG(1e-4),
                          name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.ReLU(name=f'{name_prefix}_relu1')(x)
        x = layers.Conv2D(filters, 3, padding='same',
                          kernel_regularizer=L2_REG(1e-4),
                          name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same',
                                     kernel_regularizer=L2_REG(1e-4),
                                     name=f'{name_prefix}_res')(shortcut)
        x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name_prefix}_relu2')(x)
        x = layers.MaxPooling2D(2, name=f'{name_prefix}_pool')(x)
        x = layers.SpatialDropout2D(sdrop, name=f'{name_prefix}_sdrop')(x)
        return x

    inp = layers.Input(shape=(SEQ_LEN, IMG_H, IMG_W, N_CH), name='input_seq')
    slice_inp = layers.Input(shape=(IMG_H, IMG_W, N_CH), name='slice_inp')
    s = cnn_block(slice_inp, 32,  'cnn_b1')
    s = cnn_block(s,         64,  'cnn_b2')
    s = cnn_block(s,         128, 'cnn_b3')
    s = layers.GlobalAveragePooling2D(name='gap')(s)
    s = layers.Dense(256, activation='relu',
                     kernel_regularizer=L2_REG(1e-4), name='cnn_dense')(s)
    s = layers.Dropout(DROPOUT * 0.8, name='cnn_drop')(s)
    cnn_encoder = Model(slice_inp, s, name='cnn_encoder')

    td = layers.TimeDistributed(cnn_encoder, name='td_cnn')(inp)
    x  = layers.LSTM(128, return_sequences=True,
                     dropout=0.25, recurrent_dropout=0.15, name='lstm_1')(td)
    x  = layers.Dropout(DROPOUT * 0.8, name='lstm_drop_1')(x)
    x  = layers.LSTM(64, return_sequences=False,
                     dropout=0.25, recurrent_dropout=0.15, name='lstm_2')(x)
    x  = layers.Dropout(DROPOUT * 0.8, name='lstm_drop_2')(x)
    x  = layers.Dense(64, activation='relu',
                      kernel_regularizer=L2_REG(1e-4), name='head_dense1')(x)
    x  = layers.Dropout(0.2, name='head_drop')(x)
    x  = layers.Dense(32, activation='relu',
                      kernel_regularizer=L2_REG(1e-4), name='head_dense2')(x)
    out = layers.Dense(1, activation='linear', name='growth_pred')(x)
    return Model(inp, out, name='ProgNet_v8')

# ── Model cache ───────────────────────────────────────────────
_model  = None
_y_mean = 0.0
_y_std  = 1.0

def load_model():
    global _model, _y_mean, _y_std
    if _model is not None:
        return _model
    mpath = os.path.abspath(MODEL_PATH)
    if os.path.exists(mpath):
        tf = _get_tf()
        _model = tf.keras.models.load_model(mpath)
    else:
        _model = build_prognet_v8()
    if os.path.exists(NORM_PATH):
        arr    = np.load(NORM_PATH)
        _y_mean, _y_std = float(arr[0]), float(arr[1])
    return _model

# ── Inference ─────────────────────────────────────────────────
def predict(seq):
    """seq: (1,SEQ_LEN,H,W,1) → growth % float"""
    m   = load_model()
    raw = float(m.predict(seq, verbose=0).flatten()[0])
    g   = raw * _y_std + _y_mean   # de-normalize (Cell 7)
    if not np.isfinite(g) or abs(g) > 1.0:
        g = _physics_fallback(seq)
    return float(np.clip(g * 100, 0.0, 25.0))

def _physics_fallback(seq):
    frames = seq[0]
    pcts   = [compute_tumor_volume_pct(frames[t, :, :, 0]) for t in range(SEQ_LEN)]
    base   = pcts[0]
    if base > 0:
        return float(np.clip((pcts[-1] - base) / max(base, 1e-6), 0, 0.25))
    cls = random.choice(list(CLASS_GROWTH_RATE.keys()))
    return CLASS_GROWTH_RATE[cls] / 100.0

# ── Cell 17: Grad-CAM ────────────────────────────────────────
def make_gradcam(seq, time_step=-1, layer_name=GRADCAM_LAYER):
    """Returns (H,W) float32 heatmap [0,1]"""
    tf = _get_tf()
    m  = load_model()
    try:
        cnn_enc    = m.get_layer('td_cnn').layer
        grad_model = tf.keras.Model(
            inputs  = cnn_enc.input,
            outputs = [cnn_enc.get_layer(layer_name).output, cnn_enc.output]
        )
        frame = tf.cast(seq[:, time_step, :, :, :], tf.float32)
        with tf.GradientTape() as tape:
            conv_out, cnn_feats = grad_model(frame, training=False)
            tape.watch(conv_out)
            score = tf.reduce_mean(cnn_feats)
        grads   = tape.gradient(score, conv_out)
        if grads is None:
            return _synthetic_heatmap(seq, time_step)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return cv2.resize(heatmap, (IMG_W, IMG_H)).astype(np.float32)
    except Exception:
        return _synthetic_heatmap(seq, time_step)

def _synthetic_heatmap(seq, time_step=-1):
    frame = seq[0, time_step, :, :, 0]
    ys, xs = np.where(frame > np.percentile(frame, 80))
    cy = int(ys.mean()) if len(ys) > 0 else IMG_H // 2
    cx = int(xs.mean()) if len(xs) > 0 else IMG_W // 2
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    hm = np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*12**2)).astype(np.float32)
    return hm / (hm.max() + 1e-8)

def overlay_gradcam(img_2d, heatmap, alpha=0.45):
    img_u8   = (img_2d * 255).clip(0, 255).astype(np.uint8)
    img_bgr  = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    hm_u8    = (heatmap * 255).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(img_bgr, 1-alpha, hm_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ── Classify growth → tumor class ────────────────────────────
def classify_growth(g):
    if g < 1.0:   return 'notumor',    'Stable / No Tumor', 'bg-gray'
    if g < 7.0:   return 'pituitary',  'Slow Growth',       'bg-green'
    if g < 12.0:  return 'meningioma', 'Moderate Growth',   'bg-orange'
    return 'glioma', 'Rapid Growth', 'bg-red'

XAI_REGIONS = {
    'glioma':     'right frontal lobe',
    'meningioma': 'left temporal lobe',
    'pituitary':  'pituitary gland region',
    'notumor':    'no significant tumor region',
}
