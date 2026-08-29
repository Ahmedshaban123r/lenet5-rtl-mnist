"""
golden_model.py — LeNet-5 Q7.9 Fixed-Point Golden Model
========================================================
Runs full inference through all layers in Q7.9 fixed-point,
exports hex files for every layer, and verifies against float32.

Usage:
  python golden_model.py --model_path lenet5_mnist.zip --image_index 0
  python golden_model.py --model_path lenet5_mnist.zip --image_index 0 --output_dir tb/golden
  python golden_model.py --model_path lenet5_mnist.zip --all  # run all 10k test images, report accuracy

Output (in --output_dir):
  conv1_input.hex, conv1_weights.hex, conv1_bias.hex, conv1_expected.hex
  pool1_expected.hex
  conv2_weights.hex, conv2_bias.hex, conv2_expected.hex
  pool2_expected.hex
  fc1_weights.hex, fc1_bias.hex, fc1_expected.hex
  fc2_weights.hex, fc2_bias.hex, fc2_expected.hex
  fc3_weights.hex, fc3_bias.hex, fc3_expected.hex
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# ════════════════════════════════════════════
# Q7.9 Fixed-Point Helpers
# ════════════════════════════════════════════

FRAC_BITS  = 9
SCALE      = 2 ** FRAC_BITS        # 512
TOTAL_BITS = 16
MIN_INT    = -(2 ** (TOTAL_BITS - 1))   # -32768
MAX_INT    = (2 ** (TOTAL_BITS - 1)) - 1  # 32767


def quantize(x):
    """Float → Q7.9 (int16)."""
    return np.clip(np.round(x * SCALE).astype(np.int32), MIN_INT, MAX_INT).astype(np.int16)


def to_float(x):
    """Q7.9 (int16) → float."""
    return x.astype(np.float64) / SCALE


def to_hex(val):
    """int16 → 4-char hex string (two's complement)."""
    return f"{int(val) & 0xFFFF:04X}"


def write_hex(path, values, label=""):
    """Write flat array of int16 to .hex file."""
    flat = values.flatten()
    with open(path, 'w') as f:
        for v in flat:
            f.write(to_hex(np.int16(v)) + '\n')
    if label:
        print(f"  {label:45s} → {os.path.basename(path):30s} ({len(flat)} values)")


# ════════════════════════════════════════════
# Fixed-Point Layer Operations
# ════════════════════════════════════════════

def fp_conv2d(x_q, w_q, b_q, padding=0):
    """
    Fixed-point Conv2D matching RTL behavior:
      Q7.9 × Q7.9 → Q14.18 accumulator → +bias(aligned) → round → >>9 → Q7.9
    """
    N, C_in, H, W = x_q.shape
    C_out, _, KH, KW = w_q.shape
    H_out = H - KH + 1 + 2 * padding
    W_out = W - KW + 1 + 2 * padding

    if padding > 0:
        x_q = np.pad(x_q, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                      mode='constant', constant_values=0)

    out = np.zeros((N, C_out, H_out, W_out), dtype=np.int16)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    acc = np.int32(0)
                    for ci in range(C_in):
                        for ky in range(KH):
                            for kx in range(KW):
                                a = np.int32(x_q[n, ci, i + ky, j + kx])
                                w = np.int32(w_q[co, ci, ky, kx])
                                acc += a * w
                    # Bias: shift Q7.9 → Q14.18
                    acc += np.int32(b_q[co]) << FRAC_BITS
                    # Round and shift back to Q7.9
                    acc = (acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS
                    acc = max(MIN_INT, min(MAX_INT, acc))
                    out[n, co, i, j] = np.int16(acc)

    return out


def fp_avgpool2d(x_q, kernel=2):
    """Fixed-point 2×2 average pooling: (a + b + c + d + 2) >> 2."""
    N, C, H, W = x_q.shape
    H2, W2 = H // kernel, W // kernel
    out = np.zeros((N, C, H2, W2), dtype=np.int16)

    for n in range(N):
        for c in range(C):
            for i in range(H2):
                for j in range(W2):
                    a = np.int32(x_q[n, c, 2*i,     2*j])
                    b = np.int32(x_q[n, c, 2*i,     2*j + 1])
                    c_ = np.int32(x_q[n, c, 2*i + 1, 2*j])
                    d = np.int32(x_q[n, c, 2*i + 1, 2*j + 1])
                    out[n, c, i, j] = np.int16((a + b + c_ + d + 2) >> 2)

    return out


def fp_relu(x_q):
    """ReLU on Q7.9 integers."""
    return np.maximum(x_q, np.int16(0))


def fp_linear(x_q, w_q, b_q):
    """
    Fixed-point fully-connected: x @ w^T + bias.
    Same accumulator logic as conv: Q14.18 → round → Q7.9.
    """
    N, in_f = x_q.shape
    out_f = w_q.shape[0]
    out = np.zeros((N, out_f), dtype=np.int16)

    for n in range(N):
        for o in range(out_f):
            acc = np.int32(0)
            for i in range(in_f):
                acc += np.int32(x_q[n, i]) * np.int32(w_q[o, i])
            acc += np.int32(b_q[o]) << FRAC_BITS
            acc = (acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS
            acc = max(MIN_INT, min(MAX_INT, acc))
            out[n, o] = np.int16(acc)

    return out


# ════════════════════════════════════════════
# Float32 LeNet-5 (for comparison)
# ════════════════════════════════════════════

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ════════════════════════════════════════════
# Full Fixed-Point Inference
# ════════════════════════════════════════════

def run_inference(img_q, weights, verbose=True):
    """
    Run full LeNet-5 inference in Q7.9 fixed-point.
    Returns predicted digit and dict of all intermediate outputs.
    """
    intermediates = {}

    # ── Conv1: 28×28×1 → 28×28×6 + ReLU ──
    x = fp_conv2d(img_q.reshape(1, 1, 28, 28), weights['conv1.weight'], weights['conv1.bias'], padding=2)
    x = fp_relu(x)
    intermediates['conv1'] = x.copy()
    if verbose: print(f"  Conv1+ReLU:  shape={x.shape}  range=[{x.min()}, {x.max()}]")

    # ── Pool1: 28×28×6 → 14×14×6 ──
    x = fp_avgpool2d(x)
    intermediates['pool1'] = x.copy()
    if verbose: print(f"  Pool1:       shape={x.shape}  range=[{x.min()}, {x.max()}]")

    # ── Conv2: 14×14×6 → 10×10×16 + ReLU ──
    x = fp_conv2d(x, weights['conv2.weight'], weights['conv2.bias'], padding=0)
    x = fp_relu(x)
    intermediates['conv2'] = x.copy()
    if verbose: print(f"  Conv2+ReLU:  shape={x.shape}  range=[{x.min()}, {x.max()}]")

    # ── Pool2: 10×10×16 → 5×5×16 ──
    x = fp_avgpool2d(x)
    intermediates['pool2'] = x.copy()
    if verbose: print(f"  Pool2:       shape={x.shape}  range=[{x.min()}, {x.max()}]")

    # ── Flatten: 16×5×5 → 400 ──
    x = x.reshape(1, 400)

    # ── FC1: 400 → 120 + ReLU ──
    x = fp_linear(x, weights['fc1.weight'], weights['fc1.bias'])
    x = fp_relu(x)
    intermediates['fc1'] = x.copy()
    if verbose: print(f"  FC1+ReLU:    shape={x.shape}  range=[{x.min()}, {x.max()}]")

    # ── FC2: 120 → 84 + ReLU ──
    x = fp_linear(x, weights['fc2.weight'], weights['fc2.bias'])
    x = fp_relu(x)
    intermediates['fc2'] = x.copy()
    if verbose: print(f"  FC2+ReLU:    shape={x.shape}  range=[{x.min()}, {x.max()}]")

    # ── FC3: 84 → 10 (no ReLU) ──
    x = fp_linear(x, weights['fc3.weight'], weights['fc3.bias'])
    intermediates['fc3'] = x.copy()
    if verbose: print(f"  FC3 logits:  shape={x.shape}  range=[{x.min()}, {x.max()}]")

    # ── Argmax ──
    digit = int(x.argmax())
    if verbose: print(f"  Predicted:   {digit}")

    return digit, intermediates


# ════════════════════════════════════════════
# Export Hex Files
# ════════════════════════════════════════════

def export_hex(img_q, weights, intermediates, output_dir):
    """Write all hex files for Verilog testbenches."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting hex files to {output_dir}/")

    # Input image
    write_hex(os.path.join(output_dir, 'conv1_input.hex'),
              img_q.reshape(28, 28), 'Input image (28×28)')

    # Conv1
    write_hex(os.path.join(output_dir, 'conv1_weights.hex'),
              weights['conv1.weight'], 'Conv1 weights (6×1×5×5)')
    write_hex(os.path.join(output_dir, 'conv1_bias.hex'),
              weights['conv1.bias'], 'Conv1 bias (6)')
    write_hex(os.path.join(output_dir, 'conv1_expected.hex'),
              intermediates['conv1'], 'Conv1+ReLU expected (6×28×28)')

    # Pool1
    write_hex(os.path.join(output_dir, 'pool1_expected.hex'),
              intermediates['pool1'], 'Pool1 expected (6×14×14)')

    # Conv2
    write_hex(os.path.join(output_dir, 'conv2_weights.hex'),
              weights['conv2.weight'], 'Conv2 weights (16×6×5×5)')
    write_hex(os.path.join(output_dir, 'conv2_bias.hex'),
              weights['conv2.bias'], 'Conv2 bias (16)')
    write_hex(os.path.join(output_dir, 'conv2_expected.hex'),
              intermediates['conv2'], 'Conv2+ReLU expected (16×10×10)')

    # Pool2
    write_hex(os.path.join(output_dir, 'pool2_expected.hex'),
              intermediates['pool2'], 'Pool2 expected (16×5×5)')

    # FC1
    write_hex(os.path.join(output_dir, 'fc1_weights.hex'),
              weights['fc1.weight'], 'FC1 weights (120×400)')
    write_hex(os.path.join(output_dir, 'fc1_bias.hex'),
              weights['fc1.bias'], 'FC1 bias (120)')
    write_hex(os.path.join(output_dir, 'fc1_expected.hex'),
              intermediates['fc1'], 'FC1+ReLU expected (120)')

    # FC2
    write_hex(os.path.join(output_dir, 'fc2_weights.hex'),
              weights['fc2.weight'], 'FC2 weights (84×120)')
    write_hex(os.path.join(output_dir, 'fc2_bias.hex'),
              weights['fc2.bias'], 'FC2 bias (84)')
    write_hex(os.path.join(output_dir, 'fc2_expected.hex'),
              intermediates['fc2'], 'FC2+ReLU expected (84)')

    # FC3
    write_hex(os.path.join(output_dir, 'fc3_weights.hex'),
              weights['fc3.weight'], 'FC3 weights (10×84)')
    write_hex(os.path.join(output_dir, 'fc3_bias.hex'),
              weights['fc3.bias'], 'FC3 bias (10)')
    write_hex(os.path.join(output_dir, 'fc3_expected.hex'),
              intermediates['fc3'], 'FC3 logits expected (10)')

    print(f"  Done — 18 hex files written.\n")


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='LeNet-5 Q7.9 Fixed-Point Golden Model')
    parser.add_argument('--model_path', default='lenet5_mnist.zip',
                        help='Path to PyTorch state_dict')
    parser.add_argument('--data_dir', default='./data',
                        help='MNIST data directory')
    parser.add_argument('--image_index', type=int, default=0,
                        help='MNIST test image index')
    parser.add_argument('--output_dir', default='./tb/golden',
                        help='Where to write hex files')
    parser.add_argument('--all', action='store_true',
                        help='Run all 10k test images and report accuracy')
    args = parser.parse_args()

    # ── Load model ──
    state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # ── Quantize weights ──
    q_weights = {}
    print("Weight quantization (float32 → Q7.9):")
    for name in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
                 'fc3.weight', 'fc3.bias']:
        w = state_dict[name].numpy().astype(np.float64)
        q = quantize(w)
        q_weights[name] = q
        err = np.abs(w - to_float(q))
        print(f"  {name:20s}  max_err={err.max():.6f}  mean_err={err.mean():.6f}")
    print()

    # ── Load MNIST ──
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    if args.all:
        # ── Full accuracy run ──
        print("Running full 10k test set...")
        float_model = LeNet5()
        float_model.load_state_dict(state_dict)
        float_model.eval()

        fp_correct = 0
        float_correct = 0

        for i in range(len(test_set)):
            img, label = test_set[i]
            img_q = quantize(img.numpy())

            # Fixed-point
            digit, _ = run_inference(img_q, q_weights, verbose=False)
            if digit == label:
                fp_correct += 1

            # Float32
            with torch.no_grad():
                logits = float_model(img.unsqueeze(0))
            if logits.argmax().item() == label:
                float_correct += 1

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/10000  FP={fp_correct}  Float={float_correct}")

        n = len(test_set)
        print(f"\n{'='*50}")
        print(f"  Float32 accuracy: {float_correct}/{n} = {100*float_correct/n:.2f}%")
        print(f"  Q7.9 accuracy:    {fp_correct}/{n} = {100*fp_correct/n:.2f}%")
        print(f"  Accuracy drop:    {100*(float_correct-fp_correct)/n:.2f}%")
        print(f"{'='*50}")

    else:
        # ── Single image inference + hex export ──
        img, label = test_set[args.image_index]
        img_q = quantize(img.numpy())

        print(f"Image index: {args.image_index}")
        print(f"True label:  {label}")
        print(f"Input range: [{img.min().item():.4f}, {img.max().item():.4f}]")
        print(f"Quantized:   [{img_q.min()}, {img_q.max()}]")
        print()

        print("Fixed-point inference:")
        digit, intermediates = run_inference(img_q, q_weights, verbose=True)

        print(f"\n{'='*50}")
        if digit == label:
            print(f"  CORRECT — predicted {digit}, true label {label}")
        else:
            print(f"  WRONG   — predicted {digit}, true label {label}")
        print(f"{'='*50}")

        export_hex(img_q, q_weights, intermediates, args.output_dir)

        # ── Verify against float32 ──
        print("Verification against float32:")
        float_model = LeNet5()
        float_model.load_state_dict(state_dict)
        float_model.eval()
        with torch.no_grad():
            float_logits = float_model(img.unsqueeze(0))
        float_pred = float_logits.argmax().item()

        fp_logits = to_float(intermediates['fc3'])
        fl_logits = float_logits.numpy()
        err = np.abs(fl_logits - fp_logits)
        print(f"  Float32 predicted: {float_pred}")
        print(f"  Q7.9 predicted:    {digit}")
        print(f"  Logit max error:   {err.max():.4f}")
        print(f"  Logit mean error:  {err.mean():.4f}")


if __name__ == '__main__':
    main()
