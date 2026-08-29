# lenet5-rtl-mnist

LeNet-5 MNIST classifier in synthesizable Verilog (Q7.9 fixed-point). Multiple architecture versions exploring area–latency tradeoffs, verified bit-for-bit against a Python golden model.

## Architecture

Input 28×28 → Conv1 → ReLU → AvgPool → Conv2 → ReLU → AvgPool → FC1 → ReLU → FC2 → ReLU → FC3 → Argmax → Digit 0–9



| Layer | Operation | Input | Output | MACs/pixel | Total MACs |
|-------|-----------|-------|--------|------------|------------|
| Conv1 | 5×5 conv, pad=2 | 28×28×1 | 28×28×6 | 25 | 117,600 |
| Pool1 | 2×2 avg pool | 28×28×6 | 14×14×6 | — | — |
| Conv2 | 5×5 conv, pad=0 | 14×14×6 | 10×10×16 | 150 | 240,000 |
| Pool2 | 2×2 avg pool | 10×10×16 | 5×5×16 | — | — |
| FC1 | fully connected + ReLU | 400 | 120 | 400 | 48,000 |
| FC2 | fully connected + ReLU | 120 | 84 | 120 | 10,080 |
| FC3 | fully connected | 84 | 10 | 84 | 840 |
| Argmax | max of 10 logits | 10 | 1 | — | — |

## Fixed-Point Format

All activations, weights, and biases use **Q7.9** (16-bit signed):
- Range: −64.0 to +63.998
- Resolution: 1/512 ≈ 0.00195
- Accuracy: 92.60% (vs 92.66% float32 — only 0.06% drop)

MAC accumulator is 32-bit (Q14.18). After accumulation: bias is aligned by shifting left 9, then the result is rounded and shifted right 9 to return to Q7.9.

## Versions

### v1 — Sequential (`leNET_RTL_v1_seq/`)

Single MAC unit reused for all operations. Smallest area, highest latency.

| Metric | Value |
|--------|-------|
| Total cycles | 430,963 |
| Multipliers | 1 |
| Verified | bit-for-bit against golden model |

**RTL modules:**

| File | Description |
|------|-------------|
| `conv1_seq.v` | Conv1 with padding logic, internal image/weight memory |
| `conv2_seq.v` | Conv2, 6 input channels, external input port |
| `avgpool.v` | Parameterized 2×2 average pooling (used for Pool1 and Pool2) |
| `fc_seq.v` | Parameterized FC layer (used for FC1, FC2, FC3) |
| `argmax.v` | Finds index of max among 10 logits |
| `lenet5_top.v` | Top-level: wires all layers, sequences with FSM |

### v2 — Parallel (planned)

Multiple MAC units per layer to reduce latency.

### v3 — Pipelined (planned)

Layer-level pipelining for maximum throughput.

## Golden Model (`leNET_golden_model/`)

Python fixed-point model that exactly mirrors the RTL arithmetic. Used to generate `.hex` test vectors and verify RTL outputs.

```bash
# Single image — generates all 18 hex files for Verilog testbenches
python golden_model.py --model_path lenet5_mnist.zip --image_index 0

# Full 10k accuracy report
python golden_model.py --model_path lenet5_mnist.zip --all
```

## Running the RTL (ModelSim)

```powershell
cd leNET_RTL_v1_seq/build
vlib work

# Compile all modules
vlog ..\rtl\conv1_seq.v ..\rtl\avgpool.v ..\rtl\conv2_seq.v ..\rtl\fc_seq.v ..\rtl\argmax.v ..\rtl\lenet5_top.v ..\tb\tb_lenet5_top.v

# Copy hex files
Copy-Item ..\tb\golden\*.hex -Destination .

# Run end-to-end inference
vsim -c -do "run -all; quit" work.tb_lenet5_top
```

Expected output:

PREDICTED DIGIT: 7
PASS — correct classification!

## Project Structure

lenet5-rtl-mnist/
├── leNET_RTL_v1_seq/
│ ├── rtl/
│ │ ├── conv1_seq.v
│ │ ├── avgpool.v
│ │ ├── conv2_seq.v
│ │ ├── fc_seq.v
│ │ ├── argmax.v
│ │ └── lenet5_top.v
│ ├── tb/
│ │ ├── tb_conv1_seq.v
│ │ ├── tb_avgpool.v
│ │ ├── tb_conv2_seq.v
│ │ ├── tb_fc.v
│ │ ├── tb_lenet5_top.v
│ │ └── golden/
│ │ └── *.hex
│ └── build/
├── leNET_golden_model/
│ └── golden_model.py
└── README.md


## Tools

- **RTL simulation:** ModelSim — Intel FPGA Edition 2020.1
- **Golden model:** Python 3, PyTorch, NumPy
- **Training:** PyTorch (model trained externally, weights frozen)
