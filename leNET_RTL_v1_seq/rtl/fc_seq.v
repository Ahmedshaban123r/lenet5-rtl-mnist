// ============================================================
// fc_seq.v — Fully-Connected Layer (Sequential MAC)
// ============================================================
// Parameterized for any input/output size:
//   FC1: IN_FEAT=400, OUT_FEAT=120, HAS_RELU=1
//   FC2: IN_FEAT=120, OUT_FEAT=84,  HAS_RELU=1
//   FC3: IN_FEAT=84,  OUT_FEAT=10,  HAS_RELU=0
//
// Each output neuron: IN_FEAT MACs + bias.
// Total MACs: OUT_FEAT × IN_FEAT.
//
// Weights and biases loaded from hex via $readmemh.
// Input is read through an external port (from upstream layer).
// ============================================================

module fc_seq #(
    parameter IN_FEAT     = 400,
    parameter OUT_FEAT    = 120,
    parameter HAS_RELU    = 1,
    parameter W_HEX_FILE  = "fc1_weights.hex",
    parameter B_HEX_FILE  = "fc1_bias.hex"
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,

    // Input read port
    output wire [12:0] in_addr,
    input  wire [15:0] in_data,

    // Output read port
    input  wire [12:0] out_addr,
    output wire [15:0] out_data
);

    localparam FRAC    = 9;
    localparam W_SIZE  = OUT_FEAT * IN_FEAT;

    // ────────────────────────────────────────
    // FSM states
    // ────────────────────────────────────────
    localparam S_IDLE  = 2'd0;
    localparam S_MAC   = 2'd1;
    localparam S_WRITE = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0] state;

    // ────────────────────────────────────────
    // Weight and bias memory
    // ────────────────────────────────────────
    reg signed [15:0] w_mem [0:W_SIZE-1];
    reg signed [15:0] b_mem [0:OUT_FEAT-1];

    initial begin
        $readmemh(W_HEX_FILE, w_mem);
        $readmemh(B_HEX_FILE, b_mem);
    end

    // ────────────────────────────────────────
    // Output memory
    // ────────────────────────────────────────
    reg signed [15:0] out_mem [0:OUT_FEAT-1];

    assign out_data = out_mem[out_addr];

    // ────────────────────────────────────────
    // Loop counters
    // ────────────────────────────────────────
    reg [6:0]  neuron;    // output neuron:  0..OUT_FEAT-1
    reg [12:0] idx;       // input index:    0..IN_FEAT-1

    // ────────────────────────────────────────
    // Datapath
    // ────────────────────────────────────────
    reg signed [31:0] acc;

    // Input address = idx (just reading sequentially)
    assign in_addr = idx;

    // Weight address = neuron * IN_FEAT + idx
    wire [16:0] w_addr = neuron * IN_FEAT + idx;
    wire signed [15:0] w_val = w_mem[w_addr];

    // Multiplier
    wire signed [31:0] product = $signed(in_data) * w_val;

    // Bias + Round + Saturate
    wire signed [31:0] acc_plus_bias = acc + ({{16{b_mem[neuron][15]}}, b_mem[neuron]} <<< FRAC);
    wire signed [31:0] acc_rounded   = (acc_plus_bias + 32'sd256) >>> FRAC;

    wire overflow_pos = (~acc_rounded[31]) && (|acc_rounded[30:15]);
    wire overflow_neg =  (acc_rounded[31]) && (~&acc_rounded[30:15]);
    wire signed [15:0] acc_q79 = overflow_pos ? 16'sh7FFF :
                                 overflow_neg ? 16'sh8000 :
                                 acc_rounded[15:0];

    // ReLU (conditional on parameter)
    wire signed [15:0] acc_out = (HAS_RELU && acc_q79[15]) ? 16'sd0 : acc_q79;

    // ────────────────────────────────────────
    // FSM
    // ────────────────────────────────────────
    always @(posedge clk) begin
        if (rst) begin
            state  <= S_IDLE;
            done   <= 1'b0;
            neuron <= 0;
            idx    <= 0;
            acc    <= 0;
        end
        else begin
            case (state)

                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        neuron <= 0;
                        idx    <= 0;
                        acc    <= 0;
                        state  <= S_MAC;
                    end
                end

                S_MAC: begin
                    acc <= acc + product;

                    if (idx == IN_FEAT - 1) begin
                        idx   <= 0;
                        state <= S_WRITE;
                    end
                    else begin
                        idx <= idx + 1;
                    end
                end

                S_WRITE: begin
                    out_mem[neuron] <= acc_out;

                    if (neuron == OUT_FEAT - 1) begin
                        state <= S_DONE;
                    end
                    else begin
                        neuron <= neuron + 1;
                        acc    <= 0;
                        state  <= S_MAC;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                end

            endcase
        end
    end

endmodule
