// ============================================================
// lenet5_top.v — Full LeNet-5 Inference Pipeline
// ============================================================
// Sequences all layers:
//   Conv1 → Pool1 → Conv2 → Pool2 → FC1 → FC2 → FC3 → Argmax
//
// Interface:
//   start  → begin inference (image loaded in conv1 via hex)
//   done   → inference complete
//   digit  → predicted class (0–9)
//
// Each layer runs to completion before the next starts.
// ============================================================

module lenet5_top (
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    output reg        done,
    output wire [3:0] digit
);

    // ────────────────────────────────────────
    // Top-level FSM
    // ────────────────────────────────────────
    localparam S_IDLE    = 4'd0;
    localparam S_CONV1   = 4'd1;
    localparam S_POOL1   = 4'd2;
    localparam S_CONV2   = 4'd3;
    localparam S_POOL2   = 4'd4;
    localparam S_FC1     = 4'd5;
    localparam S_FC2     = 4'd6;
    localparam S_FC3     = 4'd7;
    localparam S_ARGMAX  = 4'd8;
    localparam S_DONE    = 4'd9;

    reg [3:0] state;

    // Per-layer start signals
    reg conv1_start, pool1_start, conv2_start, pool2_start;
    reg fc1_start, fc2_start, fc3_start, argmax_start;

    // Per-layer done signals
    wire conv1_done, pool1_done, conv2_done, pool2_done;
    wire fc1_done, fc2_done, fc3_done, argmax_done;

    // ────────────────────────────────────────
    // Inter-layer wiring
    // ────────────────────────────────────────

    // Conv1 output port
    wire [12:0] conv1_out_addr;
    wire [15:0] conv1_out_data;

    // Pool1 ↔ Conv1
    wire [12:0] pool1_in_addr;
    wire [15:0] pool1_out_data;
    wire [12:0] pool1_out_addr_port;

    // Conv2 ↔ Pool1
    wire [12:0] conv2_in_addr;
    wire [15:0] conv2_out_data;
    wire [12:0] conv2_out_addr_port;

    // Pool2 ↔ Conv2
    wire [12:0] pool2_in_addr;
    wire [15:0] pool2_out_data;
    wire [12:0] pool2_out_addr_port;

    // FC1 ↔ Pool2
    wire [12:0] fc1_in_addr;
    wire [15:0] fc1_out_data;
    wire [12:0] fc1_out_addr_port;

    // FC2 ↔ FC1
    wire [12:0] fc2_in_addr;
    wire [15:0] fc2_out_data;
    wire [12:0] fc2_out_addr_port;

    // FC3 ↔ FC2
    wire [12:0] fc3_in_addr;
    wire [15:0] fc3_out_data;
    wire [12:0] fc3_out_addr_port;

    // Argmax ↔ FC3
    wire [12:0] argmax_in_addr;

    // ────────────────────────────────────────
    // Address muxing
    // ────────────────────────────────────────
    // Each layer's output address port is driven by
    // whichever downstream layer is currently reading it.

    // Conv1 out_addr ← Pool1 reads during S_POOL1
    assign conv1_out_addr = pool1_in_addr;

    // Pool1 out_addr ← Conv2 reads during S_CONV2
    assign pool1_out_addr_port = conv2_in_addr;

    // Conv2 out_addr ← Pool2 reads during S_POOL2
    assign conv2_out_addr_port = pool2_in_addr;

    // Pool2 out_addr ← FC1 reads during S_FC1
    assign pool2_out_addr_port = fc1_in_addr;

    // FC1 out_addr ← FC2 reads during S_FC2
    assign fc1_out_addr_port = fc2_in_addr;

    // FC2 out_addr ← FC3 reads during S_FC3
    assign fc2_out_addr_port = fc3_in_addr;

    // FC3 out_addr ← Argmax reads during S_ARGMAX
    assign fc3_out_addr_port = argmax_in_addr;

    // ────────────────────────────────────────
    // Layer instantiations
    // ────────────────────────────────────────

    // ── Conv1: 28×28×1 → 28×28×6 ──
    conv1_seq u_conv1 (
        .clk      (clk),
        .rst      (rst),
        .start    (conv1_start),
        .done     (conv1_done),
        .out_addr (conv1_out_addr),
        .out_data (conv1_out_data)
    );

    // ── Pool1: 28×28×6 → 14×14×6 ──
    avgpool #(.IN_H(28), .IN_W(28), .N_CH(6)) u_pool1 (
        .clk      (clk),
        .rst      (rst),
        .start    (pool1_start),
        .done     (pool1_done),
        .in_addr  (pool1_in_addr),
        .in_data  (conv1_out_data),
        .out_addr (pool1_out_addr_port),
        .out_data (pool1_out_data)
    );

    // ── Conv2: 14×14×6 → 10×10×16 ──
    conv2_seq u_conv2 (
        .clk      (clk),
        .rst      (rst),
        .start    (conv2_start),
        .done     (conv2_done),
        .in_addr  (conv2_in_addr),
        .in_data  (pool1_out_data),
        .out_addr (conv2_out_addr_port),
        .out_data (conv2_out_data)
    );

    // ── Pool2: 10×10×16 → 5×5×16 ──
    avgpool #(.IN_H(10), .IN_W(10), .N_CH(16)) u_pool2 (
        .clk      (clk),
        .rst      (rst),
        .start    (pool2_start),
        .done     (pool2_done),
        .in_addr  (pool2_in_addr),
        .in_data  (conv2_out_data),
        .out_addr (pool2_out_addr_port),
        .out_data (pool2_out_data)
    );

    // ── FC1: 400 → 120 + ReLU ──
    fc_seq #(
        .IN_FEAT(400), .OUT_FEAT(120), .HAS_RELU(1),
        .W_HEX_FILE("fc1_weights.hex"), .B_HEX_FILE("fc1_bias.hex")
    ) u_fc1 (
        .clk      (clk),
        .rst      (rst),
        .start    (fc1_start),
        .done     (fc1_done),
        .in_addr  (fc1_in_addr),
        .in_data  (pool2_out_data),
        .out_addr (fc1_out_addr_port),
        .out_data (fc1_out_data)
    );

    // ── FC2: 120 → 84 + ReLU ──
    fc_seq #(
        .IN_FEAT(120), .OUT_FEAT(84), .HAS_RELU(1),
        .W_HEX_FILE("fc2_weights.hex"), .B_HEX_FILE("fc2_bias.hex")
    ) u_fc2 (
        .clk      (clk),
        .rst      (rst),
        .start    (fc2_start),
        .done     (fc2_done),
        .in_addr  (fc2_in_addr),
        .in_data  (fc1_out_data),
        .out_addr (fc2_out_addr_port),
        .out_data (fc2_out_data)
    );

    // ── FC3: 84 → 10, no ReLU ──
    fc_seq #(
        .IN_FEAT(84), .OUT_FEAT(10), .HAS_RELU(0),
        .W_HEX_FILE("fc3_weights.hex"), .B_HEX_FILE("fc3_bias.hex")
    ) u_fc3 (
        .clk      (clk),
        .rst      (rst),
        .start    (fc3_start),
        .done     (fc3_done),
        .in_addr  (fc3_in_addr),
        .in_data  (fc2_out_data),
        .out_addr (fc3_out_addr_port),
        .out_data (fc3_out_data)
    );

    // ── Argmax: 10 → 1 digit ──
    argmax u_argmax (
        .clk      (clk),
        .rst      (rst),
        .start    (argmax_start),
        .done     (argmax_done),
        .in_addr  (argmax_in_addr),
        .in_data  (fc3_out_data),
        .digit    (digit)
    );

    // ────────────────────────────────────────
    // Top-level FSM
    // ────────────────────────────────────────
    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            done         <= 1'b0;
            conv1_start  <= 0;
            pool1_start  <= 0;
            conv2_start  <= 0;
            pool2_start  <= 0;
            fc1_start    <= 0;
            fc2_start    <= 0;
            fc3_start    <= 0;
            argmax_start <= 0;
        end
        else begin
            // Default: deassert all starts after one cycle
            conv1_start  <= 0;
            pool1_start  <= 0;
            conv2_start  <= 0;
            pool2_start  <= 0;
            fc1_start    <= 0;
            fc2_start    <= 0;
            fc3_start    <= 0;
            argmax_start <= 0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        conv1_start <= 1;
                        state <= S_CONV1;
                    end
                end

                S_CONV1: begin
                    if (conv1_done) begin
                        pool1_start <= 1;
                        state <= S_POOL1;
                    end
                end

                S_POOL1: begin
                    if (pool1_done) begin
                        conv2_start <= 1;
                        state <= S_CONV2;
                    end
                end

                S_CONV2: begin
                    if (conv2_done) begin
                        pool2_start <= 1;
                        state <= S_POOL2;
                    end
                end

                S_POOL2: begin
                    if (pool2_done) begin
                        fc1_start <= 1;
                        state <= S_FC1;
                    end
                end

                S_FC1: begin
                    if (fc1_done) begin
                        fc2_start <= 1;
                        state <= S_FC2;
                    end
                end

                S_FC2: begin
                    if (fc2_done) begin
                        fc3_start <= 1;
                        state <= S_FC3;
                    end
                end

                S_FC3: begin
                    if (fc3_done) begin
                        argmax_start <= 1;
                        state <= S_ARGMAX;
                    end
                end

                S_ARGMAX: begin
                    if (argmax_done) begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

endmodule
