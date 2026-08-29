"""
export_golden_hex.py
====================
Loads the trained LeNet-5, quantizes to Q7.9, runs one MNIST image
through Conv1 + ReLU, and exports .hex files for Verilog $readmemh.

Output files (in tb/golden/):
  conv1_input.hex      — 784 lines  (28x28 input image, Q7.9)
  conv1_weights.hex    — 150 lines  (6 filters x 5x5, Q7.9)
  conv1_bias.hex       — 6 lines    (one per filter, Q7.9)
  conv1_expected.hex   — 4704 lines (6 x 28x28 output after ReLU, Q7.9)

Each line: 4-digit hex, 16-bit two's complement.
Memory layout is row-major for each channel.

Usage:
  python export_golden_hex.py --model_path lenet5_mnist.zip --image_index 0
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


# ── Q7.9 fixed-point helpers ──

FRAC_BITS = 9
SCALE = 2 ** FRAC_BITS          # 512
TOTAL_BITS = 16
MIN_INT = -(2 ** (TOTAL_BITS - 1))   # -32768
MAX_INT = (2 ** (TOTAL_BITS - 1)) - 1  # 32767


def float_to_q79(x: np.ndarray) -> np.ndarray:
    """Float → Q7.9 integer (signed 16-bit)."""
    scaled = np.round(x * SCALE).astype(np.int32)
    return np.clip(scaled, MIN_INT, MAX_INT).astype(np.int16)


def q79_to_float(x: np.ndarray) -> np.ndarray:
    """Q7.9 integer → float (for verification)."""
    return x.astype(np.float64) / SCALE


def to_hex(val: np.int16) -> str:
    """Signed 16-bit integer → 4-digit hex string (two's complement)."""
    # Convert to unsigned representation for hex
    unsigned = int(val) & 0xFFFF
    return f"{unsigned:04X}"


def write_hex_file(path: str, values: np.ndarray, label: str):
    """Write a flat array of Q7.9 int16 values to a .hex file."""
    flat = values.flatten()
    with open(path, 'w') as f:
        for v in flat:
            f.write(to_hex(np.int16(v)) + '\n')
    print(f"  {label:30s} → {path}  ({len(flat)} values)")


# ── Conv1 fixed-point forward pass (matches golden model exactly) ──

def conv2d_q79(x_q, w_q, b_q, padding=0):
    """
    Fixed-point Conv2D in numpy, matching the RTL behavior:
      - inputs/weights/biases are Q7.9 integers (int16)
      - multiply: int16 × int16 → int32 (Q14.18)
      - accumulate in int32
      - add bias: bias is Q7.9, shift left by 9 to align to Q14.18
      - result: shift right by 9 to get back to Q7.9
      - clip to 16-bit signed range
    """
    N, C_in, H, W = x_q.shape
    C_out, _, KH, KW = w_q.shape
    H_out = H - KH + 1 + 2 * padding
    W_out = W - KW + 1 + 2 * padding

    if padding > 0:
        x_q = np.pad(x_q, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                      mode='constant', constant_values=0)

    out = np.zeros((N, C_out, H_out, W_out), dtype=np.int32)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    # MAC in Q14.18
                    acc = np.int32(0)
                    for ci in range(C_in):
                        for ky in range(KH):
                            for kx in range(KW):
                                a = np.int32(x_q[n, ci, i + ky, j + kx])
                                w = np.int32(w_q[co, ci, ky, kx])
                                acc += a * w

                    # Add bias: shift bias from Q7.9 to Q14.18
                    acc += np.int32(b_q[co]) << FRAC_BITS

                    # Shift back: Q14.18 → Q7.9 with rounding
                    # Add 0.5 LSB for rounding before shift
                    acc = (acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS

                    # Clip to 16-bit signed
                    acc = max(MIN_INT, min(MAX_INT, acc))
                    out[n, co, i, j] = acc

    return out.astype(np.int16)


def relu_q79(x_q):
    """ReLU on Q7.9 integers — just clamp negatives to 0."""
    return np.maximum(x_q, np.int16(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='lenet5_mnist.zip')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--image_index', type=int, default=0,
                        help='Which MNIST test image to use')
    parser.add_argument('--output_dir', default='./tb/golden')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model weights ──
    state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # ── Load one MNIST test image ──
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(args.data_dir, train=False, download=False, transform=transform)
    image, label = test_set[args.image_index]

    print(f"Image index: {args.image_index}, True label: {label}")
    print(f"Image range: [{image.min().item():.4f}, {image.max().item():.4f}]")
    print()

    # ── Quantize input image to Q7.9 ──
    img_np = image.numpy().astype(np.float64)           # (1, 28, 28)
    img_q = float_to_q79(img_np)                         # int16

    print(f"Quantized input range: [{img_q.min()}, {img_q.max()}]")
    print(f"  (as float: [{q79_to_float(img_q).min():.4f}, {q79_to_float(img_q).max():.4f}])")

    # ── Quantize Conv1 weights and bias ──
    w1 = state_dict['conv1.weight'].numpy().astype(np.float64)   # (6, 1, 5, 5)
    b1 = state_dict['conv1.bias'].numpy().astype(np.float64)     # (6,)
    w1_q = float_to_q79(w1)
    b1_q = float_to_q79(b1)

    print(f"Conv1 weights range: [{w1_q.min()}, {w1_q.max()}]")
    print(f"Conv1 bias range:    [{b1_q.min()}, {b1_q.max()}]")
    print()

    # ── Run Conv1 + ReLU in fixed-point ──
    print("Running Conv1 (Q7.9 fixed-point, 25 MACs × 4704 outputs)...")
    img_q_batch = img_q.reshape(1, 1, 28, 28)   # add batch dim
    conv1_out_q = conv2d_q79(img_q_batch, w1_q, b1_q, padding=2)
    conv1_relu_q = relu_q79(conv1_out_q)

    print(f"Conv1 output shape: {conv1_relu_q.shape}")
    print(f"Conv1 output range: [{conv1_relu_q.min()}, {conv1_relu_q.max()}]")
    print(f"  (as float: [{q79_to_float(conv1_relu_q).min():.4f}, {q79_to_float(conv1_relu_q).max():.4f}])")
    print()

    # ── Also run in float for accuracy check ──
    with torch.no_grad():
        float_conv1 = F.conv2d(image.unsqueeze(0), state_dict['conv1.weight'],
                               state_dict['conv1.bias'], padding=2)
        float_conv1_relu = F.relu(float_conv1)

    float_out = float_conv1_relu.numpy()
    fixed_out = q79_to_float(conv1_relu_q)
    error = np.abs(float_out - fixed_out)
    print(f"Verification vs float32:")
    print(f"  Max error:  {error.max():.6f}")
    print(f"  Mean error: {error.mean():.6f}")
    print()

    # ── Write hex files ──
    print("Writing hex files:")

    # Input: (1, 28, 28) → flatten row-major: row0, row1, ..., row27
    write_hex_file(
        os.path.join(args.output_dir, 'conv1_input.hex'),
        img_q.reshape(28, 28),
        'conv1_input.hex (28×28)'
    )

    # Weights: (6, 1, 5, 5) → flatten as filter0[row0..row4], filter1[...], ...
    write_hex_file(
        os.path.join(args.output_dir, 'conv1_weights.hex'),
        w1_q.reshape(6, 25),
        'conv1_weights.hex (6×25)'
    )

    # Bias: (6,)
    write_hex_file(
        os.path.join(args.output_dir, 'conv1_bias.hex'),
        b1_q,
        'conv1_bias.hex (6)'
    )

    # Expected output: (1, 6, 28, 28) → flatten as ch0[row0..row27], ch1[...], ...
    write_hex_file(
        os.path.join(args.output_dir, 'conv1_expected.hex'),
        conv1_relu_q.reshape(6, 28 * 28),
        'conv1_expected.hex (6×784)'
    )

    print(f"\nDone. All files in: {args.output_dir}")


if __name__ == '__main__':
    main()
