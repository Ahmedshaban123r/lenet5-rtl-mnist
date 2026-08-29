// ============================================================
// conv1_seq.v — Conv1 Sequential MAC Engine
// ============================================================
// Input:   28x28x1  image         (Q7.9, 16-bit signed)
// Output:  28x28x6  feature map   (Q7.9, 16-bit signed, after ReLU)
// Kernel:  5x5, padding=2, stride=1, 6 output channels
//
// Architecture: single signed multiplier, reused for all MACs.
//   25 MACs per output pixel × 4704 pixels = 117,600 MAC cycles.
//
// Memory: input image, weights, biases loaded via $readmemh.
//         Output stored in internal RAM.
//
// FSM: IDLE → MAC (25 cycles per pixel) → WRITE → ... → DONE
// ============================================================

module conv1_seq (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,

    // Output read port (downstream layer reads results)
    input  wire [12:0] out_addr,   // 0..4703 (6*28*28 - 1)
    output wire [15:0] out_data    // Q7.9
);

    // ────────────────────────────────────────
    // Parameters
    // ────────────────────────────────────────
    localparam IMG_H     = 28;
    localparam IMG_W     = 28;
    localparam K         = 5;
    localparam PAD       = 2;
    localparam N_FILTERS = 6;
    localparam IN_CH     = 1;
    localparam FRAC      = 9;

    localparam OUT_H     = IMG_H;  // with padding=2, output = input size
    localparam OUT_W     = IMG_W;
    localparam MAC_COUNT = K * K * IN_CH;  // 25

    localparam IMG_SIZE  = IMG_H * IMG_W;           // 784
    localparam W_SIZE    = N_FILTERS * IN_CH * K*K; // 150
    localparam OUT_SIZE  = N_FILTERS * OUT_H * OUT_W; // 4704

    // ────────────────────────────────────────
    // FSM states
    // ────────────────────────────────────────
    localparam S_IDLE  = 2'd0;
    localparam S_MAC   = 2'd1;
    localparam S_WRITE = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0] state;

    // ────────────────────────────────────────
    // Memory arrays
    // ────────────────────────────────────────
    reg signed [15:0] img_mem  [0:IMG_SIZE-1];   // input image
    reg signed [15:0] w_mem    [0:W_SIZE-1];     // weights
    reg signed [15:0] b_mem    [0:N_FILTERS-1];  // biases
    reg signed [15:0] out_mem  [0:OUT_SIZE-1];   // output feature map

    // Load memories from hex files
    initial begin
        $readmemh("conv1_input.hex",   img_mem);
        $readmemh("conv1_weights.hex", w_mem);
        $readmemh("conv1_bias.hex",    b_mem);
    end

    // Output read port
    assign out_data = out_mem[out_addr];

    // ────────────────────────────────────────
    // Loop counters
    // ────────────────────────────────────────
    reg [2:0] oc;      // output channel:  0..5
    reg [4:0] row;     // output row:      0..27
    reg [4:0] col;     // output col:      0..27
    reg [2:0] ky;      // kernel row:      0..4
    reg [2:0] kx;      // kernel col:      0..4

    // ────────────────────────────────────────
    // Datapath signals
    // ────────────────────────────────────────
    reg signed [31:0] acc;            // MAC accumulator (Q14.18)
    wire signed [15:0] img_val;       // pixel from memory (or 0 if padded)
    wire signed [15:0] w_val;         // weight from memory
    wire signed [31:0] product;       // multiplier output

    // ── Padding logic ──
    // Compute actual image coordinates
    wire signed [5:0] img_r = row + ky - PAD;  // signed to detect < 0
    wire signed [5:0] img_c = col + kx - PAD;

    wire pad_zero = (img_r < 0) || (img_r >= IMG_H) ||
                    (img_c < 0) || (img_c >= IMG_W);

    // Image address: row * 28 + col
    wire [9:0] img_addr = img_r[4:0] * IMG_W + img_c[4:0];

    // Input value: 0 if in padding region, else read from memory
    assign img_val = pad_zero ? 16'sd0 : img_mem[img_addr];

    // ── Weight address ──
    // Layout: filter oc, then kernel flattened row-major
    // addr = oc * 25 + ky * 5 + kx
    wire [7:0] w_addr = oc * (K*K) + ky * K + kx;
    assign w_val = w_mem[w_addr];

    // ── Multiplier ──
    assign product = img_val * w_val;  // Q7.9 × Q7.9 = Q14.18

    // ── Output address ──
    // Layout: channel-major → oc * 784 + row * 28 + col
    wire [12:0] wr_addr = oc * (OUT_H * OUT_W) + row * OUT_W + col;

    // ── Bias-aligned, rounded, ReLU'd result ──
    wire signed [31:0] acc_plus_bias;
    wire signed [31:0] acc_rounded;
    wire signed [15:0] acc_q79;
    wire signed [15:0] acc_relu;

    // Bias is Q7.9 — shift left by FRAC to align to Q14.18
    assign acc_plus_bias = acc + ({{16{b_mem[oc][15]}}, b_mem[oc]} <<< FRAC);

    // Round: add 0.5 LSB (= 1 << (FRAC-1) = 256) then shift right
    assign acc_rounded = (acc_plus_bias + 32'sd256) >>> FRAC;

    // Saturate to 16-bit signed range
    wire overflow_pos = (~acc_rounded[31]) && (|acc_rounded[30:15]);
    wire overflow_neg =  (acc_rounded[31]) && (~&acc_rounded[30:15]);
    assign acc_q79 = overflow_pos ? 16'sh7FFF :
                     overflow_neg ? 16'sh8000 :
                     acc_rounded[15:0];

    // ReLU
    assign acc_relu = acc_q79[15] ? 16'sd0 : acc_q79;

    // ────────────────────────────────────────
    // FSM + Datapath control
    // ────────────────────────────────────────
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            done  <= 1'b0;
            oc    <= 0;
            row   <= 0;
            col   <= 0;
            ky    <= 0;
            kx    <= 0;
            acc   <= 0;
        end
        else begin
            case (state)
                // ── IDLE: wait for start ──
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        oc  <= 0;
                        row <= 0;
                        col <= 0;
                        ky  <= 0;
                        kx  <= 0;
                        acc <= 0;
                        state <= S_MAC;
                    end
                end

                // ── MAC: one multiply-accumulate per clock ──
                S_MAC: begin
                    acc <= acc + product;

                    // Advance kernel counters
                    if (kx == K - 1) begin
                        kx <= 0;
                        if (ky == K - 1) begin
                            ky <= 0;
                            // All 25 MACs done for this pixel
                            state <= S_WRITE;
                        end
                        else begin
                            ky <= ky + 1;
                        end
                    end
                    else begin
                        kx <= kx + 1;
                    end
                end

                // ── WRITE: add bias, round, ReLU, store result ──
                S_WRITE: begin
                    out_mem[wr_addr] <= acc_relu;

                    // Advance spatial/channel counters
                    if (col == OUT_W - 1) begin
                        col <= 0;
                        if (row == OUT_H - 1) begin
                            row <= 0;
                            if (oc == N_FILTERS - 1) begin
                                // All done
                                state <= S_DONE;
                            end
                            else begin
                                oc <= oc + 1;
                                acc <= 0;
                                state <= S_MAC;
                            end
                        end
                        else begin
                            row <= row + 1;
                            acc <= 0;
                            state <= S_MAC;
                        end
                    end
                    else begin
                        col <= col + 1;
                        acc <= 0;
                        state <= S_MAC;
                    end
                end

                // ── DONE: hold until reset ──
                S_DONE: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

endmodule