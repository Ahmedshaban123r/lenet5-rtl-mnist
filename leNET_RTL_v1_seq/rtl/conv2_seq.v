// ============================================================
// conv2_seq.v — Conv2 Sequential MAC Engine
// ============================================================
// Input:   14×14×6   feature map  (Q7.9, from Pool1)
// Output:  10×10×16  feature map  (Q7.9, after ReLU)
// Kernel:  5×5×6, padding=0, stride=1, 16 output channels
//
// Each output pixel: 5×5×6 = 150 MACs
// Total outputs:     16×10×10 = 1600
// Total MACs:        1600 × 150 = 240,000
//
// No padding — every kernel position is fully inside the input.
// ============================================================

module conv2_seq (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,

    // Input read port (reads from Pool1 output)
    output wire [12:0] in_addr,
    input  wire [15:0] in_data,

    // Output read port (downstream reads results)
    input  wire [12:0] out_addr,
    output wire [15:0] out_data
);

    // ────────────────────────────────────────
    // Parameters
    // ────────────────────────────────────────
    localparam IN_H      = 14;
    localparam IN_W      = 14;
    localparam IN_CH     = 6;
    localparam K         = 5;
    localparam OUT_CH    = 16;
    localparam FRAC      = 9;

    localparam OUT_H     = IN_H - K + 1;  // 10
    localparam OUT_W     = IN_W - K + 1;  // 10
    localparam MAC_COUNT = K * K * IN_CH;  // 150

    localparam W_SIZE    = OUT_CH * IN_CH * K * K;  // 2400
    localparam OUT_SIZE  = OUT_CH * OUT_H * OUT_W;  // 1600

    // ────────────────────────────────────────
    // FSM states
    // ────────────────────────────────────────
    localparam S_IDLE  = 2'd0;
    localparam S_MAC   = 2'd1;
    localparam S_WRITE = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0] state;

    // ────────────────────────────────────────
    // Weight and bias memory (loaded from hex)
    // ────────────────────────────────────────
    reg signed [15:0] w_mem [0:W_SIZE-1];
    reg signed [15:0] b_mem [0:OUT_CH-1];

    initial begin
        $readmemh("conv2_weights.hex", w_mem);
        $readmemh("conv2_bias.hex",    b_mem);
    end

    // ────────────────────────────────────────
    // Output memory
    // ────────────────────────────────────────
    reg signed [15:0] out_mem [0:OUT_SIZE-1];

    assign out_data = out_mem[out_addr];

    // ────────────────────────────────────────
    // Loop counters
    // ────────────────────────────────────────
    reg [4:0] oc;      // output channel:   0..15
    reg [3:0] row;     // output row:       0..9
    reg [3:0] col;     // output col:       0..9
    reg [2:0] ic;      // input channel:    0..5
    reg [2:0] ky;      // kernel row:       0..4
    reg [2:0] kx;      // kernel col:       0..4

    // ────────────────────────────────────────
    // Datapath
    // ────────────────────────────────────────
    reg signed [31:0] acc;

    // ── Input address ──
    // No padding, so image coords are simply (row+ky, col+kx)
    // Layout: channel-major → ic * 14*14 + (row+ky) * 14 + (col+kx)
    wire [12:0] rd_addr = ic * (IN_H * IN_W) + (row + ky) * IN_W + (col + kx);
    assign in_addr = rd_addr;

    // ── Weight address ──
    // Layout: oc * (6*25) + ic * 25 + ky * 5 + kx
    wire [11:0] w_addr = oc * (IN_CH * K * K) + ic * (K * K) + ky * K + kx;
    wire signed [15:0] w_val = w_mem[w_addr];

    // ── Multiplier ──
    wire signed [31:0] product = $signed(in_data) * w_val;

    // ── Output address ──
    wire [12:0] wr_addr = oc * (OUT_H * OUT_W) + row * OUT_W + col;

    // ── Bias + Round + Saturate + ReLU ──
    wire signed [31:0] acc_plus_bias = acc + ({{16{b_mem[oc][15]}}, b_mem[oc]} <<< FRAC);
    wire signed [31:0] acc_rounded   = (acc_plus_bias + 32'sd256) >>> FRAC;

    wire overflow_pos = (~acc_rounded[31]) && (|acc_rounded[30:15]);
    wire overflow_neg =  (acc_rounded[31]) && (~&acc_rounded[30:15]);
    wire signed [15:0] acc_q79 = overflow_pos ? 16'sh7FFF :
                                 overflow_neg ? 16'sh8000 :
                                 acc_rounded[15:0];

    wire signed [15:0] acc_relu = acc_q79[15] ? 16'sd0 : acc_q79;

    // ────────────────────────────────────────
    // FSM
    // ────────────────────────────────────────
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            done  <= 1'b0;
            oc    <= 0;
            row   <= 0;
            col   <= 0;
            ic    <= 0;
            ky    <= 0;
            kx    <= 0;
            acc   <= 0;
        end
        else begin
            case (state)

                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        oc  <= 0;
                        row <= 0;
                        col <= 0;
                        ic  <= 0;
                        ky  <= 0;
                        kx  <= 0;
                        acc <= 0;
                        state <= S_MAC;
                    end
                end

                // One MAC per clock: 150 cycles per output pixel
                S_MAC: begin
                    acc <= acc + product;

                    // Advance: kx → ky → ic (innermost to outermost)
                    if (kx == K - 1) begin
                        kx <= 0;
                        if (ky == K - 1) begin
                            ky <= 0;
                            if (ic == IN_CH - 1) begin
                                ic <= 0;
                                // All 150 MACs done
                                state <= S_WRITE;
                            end
                            else begin
                                ic <= ic + 1;
                            end
                        end
                        else begin
                            ky <= ky + 1;
                        end
                    end
                    else begin
                        kx <= kx + 1;
                    end
                end

                // Store result, advance to next output pixel
                S_WRITE: begin
                    out_mem[wr_addr] <= acc_relu;

                    if (col == OUT_W - 1) begin
                        col <= 0;
                        if (row == OUT_H - 1) begin
                            row <= 0;
                            if (oc == OUT_CH - 1) begin
                                state <= S_DONE;
                            end
                            else begin
                                oc    <= oc + 1;
                                acc   <= 0;
                                state <= S_MAC;
                            end
                        end
                        else begin
                            row   <= row + 1;
                            acc   <= 0;
                            state <= S_MAC;
                        end
                    end
                    else begin
                        col   <= col + 1;
                        acc   <= 0;
                        state <= S_MAC;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                end

            endcase
        end
    end

endmodule
