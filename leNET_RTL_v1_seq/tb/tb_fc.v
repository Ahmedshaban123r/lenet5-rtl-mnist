// ============================================================
// tb_fc.v — Testbench for FC1 → FC2 → FC3
// ============================================================
// Tests all three FC layers independently, each against
// its own golden hex. Also verifies Pool2 using avgpool.
// ============================================================

`timescale 1ns / 1ps

module tb_fc;

    reg clk;
    reg rst;

    initial clk = 0;
    always #5 clk = ~clk;

    // ────────────────────────────────────────
    // Pool2 input (Conv2 output)
    // ────────────────────────────────────────
    localparam CONV2_SIZE = 1600;  // 16*10*10
    reg signed [15:0] conv2_mem [0:CONV2_SIZE-1];
    initial $readmemh("conv2_expected.hex", conv2_mem);

    // ────────────────────────────────────────
    // Pool2: 10×10×16 → 5×5×16 (reuse avgpool)
    // ────────────────────────────────────────
    reg         pool2_start;
    wire        pool2_done;
    wire [12:0] pool2_in_addr;
    wire [15:0] pool2_in_data = conv2_mem[pool2_in_addr];
    reg  [12:0] pool2_out_addr;
    wire [15:0] pool2_out_data;

    avgpool #(.IN_H(10), .IN_W(10), .N_CH(16)) pool2 (
        .clk(clk), .rst(rst), .start(pool2_start), .done(pool2_done),
        .in_addr(pool2_in_addr), .in_data(pool2_in_data),
        .out_addr(pool2_out_addr), .out_data(pool2_out_data)
    );

    // Pool2 golden
    localparam POOL2_SIZE = 400;
    reg signed [15:0] pool2_exp [0:POOL2_SIZE-1];
    initial $readmemh("pool2_expected.hex", pool2_exp);

    // ────────────────────────────────────────
    // FC1: 400 → 120 + ReLU
    // ────────────────────────────────────────
    reg         fc1_start;
    wire        fc1_done;
    wire [12:0] fc1_in_addr;
    // FC1 reads from Pool2 output
    wire [15:0] fc1_in_data;
    reg  [12:0] fc1_out_addr;
    wire [15:0] fc1_out_data;

    // Connect FC1 input to Pool2 output
    assign fc1_in_data = pool2_out_data;

    fc_seq #(
        .IN_FEAT(400), .OUT_FEAT(120), .HAS_RELU(1),
        .W_HEX_FILE("fc1_weights.hex"), .B_HEX_FILE("fc1_bias.hex")
    ) fc1 (
        .clk(clk), .rst(rst), .start(fc1_start), .done(fc1_done),
        .in_addr(fc1_in_addr), .in_data(fc1_in_data),
        .out_addr(fc1_out_addr), .out_data(fc1_out_data)
    );

    // Mux Pool2 out_addr: FC1 reads during FC1, testbench reads during verify
    reg fc1_running;
    always @(*) begin
        if (fc1_running)
            pool2_out_addr = fc1_in_addr;
        else
            pool2_out_addr = 0;
    end

    localparam FC1_SIZE = 120;
    reg signed [15:0] fc1_exp [0:FC1_SIZE-1];
    initial $readmemh("fc1_expected.hex", fc1_exp);

    // ────────────────────────────────────────
    // FC2: 120 → 84 + ReLU
    // ────────────────────────────────────────
    reg         fc2_start;
    wire        fc2_done;
    wire [12:0] fc2_in_addr;
    wire [15:0] fc2_in_data;
    reg  [12:0] fc2_out_addr;
    wire [15:0] fc2_out_data;

    // FC2 reads from FC1 output
    assign fc2_in_data = fc1_out_data;

    fc_seq #(
        .IN_FEAT(120), .OUT_FEAT(84), .HAS_RELU(1),
        .W_HEX_FILE("fc2_weights.hex"), .B_HEX_FILE("fc2_bias.hex")
    ) fc2 (
        .clk(clk), .rst(rst), .start(fc2_start), .done(fc2_done),
        .in_addr(fc2_in_addr), .in_data(fc2_in_data),
        .out_addr(fc2_out_addr), .out_data(fc2_out_data)
    );

    reg fc2_running;
    always @(*) begin
        if (fc2_running)
            fc1_out_addr = fc2_in_addr;
        else
            fc1_out_addr = 0;
    end

    localparam FC2_SIZE = 84;
    reg signed [15:0] fc2_exp [0:FC2_SIZE-1];
    initial $readmemh("fc2_expected.hex", fc2_exp);

    // ────────────────────────────────────────
    // FC3: 84 → 10, no ReLU
    // ────────────────────────────────────────
    reg         fc3_start;
    wire        fc3_done;
    wire [12:0] fc3_in_addr;
    wire [15:0] fc3_in_data;
    reg  [12:0] fc3_out_addr;
    wire [15:0] fc3_out_data;

    assign fc3_in_data = fc2_out_data;

    fc_seq #(
        .IN_FEAT(84), .OUT_FEAT(10), .HAS_RELU(0),
        .W_HEX_FILE("fc3_weights.hex"), .B_HEX_FILE("fc3_bias.hex")
    ) fc3 (
        .clk(clk), .rst(rst), .start(fc3_start), .done(fc3_done),
        .in_addr(fc3_in_addr), .in_data(fc3_in_data),
        .out_addr(fc3_out_addr), .out_data(fc3_out_data)
    );

    reg fc3_running;
    always @(*) begin
        if (fc3_running)
            fc2_out_addr = fc3_in_addr;
        else
            fc2_out_addr = 0;
    end

    localparam FC3_SIZE = 10;
    reg signed [15:0] fc3_exp [0:FC3_SIZE-1];
    initial $readmemh("fc3_expected.hex", fc3_exp);

    // ────────────────────────────────────────
    // VCD + timeout
    // ────────────────────────────────────────
    initial begin
        $dumpfile("fc.vcd");
        $dumpvars(0, tb_fc);
    end

    initial begin
        #50_000_000;
        $display("ERROR: Simulation timed out.");
        $finish;
    end

    // ────────────────────────────────────────
    // Verification task
    // ────────────────────────────────────────
    integer i, mismatches;

    // ────────────────────────────────────────
    // Main test sequence
    // ────────────────────────────────────────
    initial begin
        rst = 1;
        pool2_start = 0; fc1_start = 0; fc2_start = 0; fc3_start = 0;
        fc1_running = 0; fc2_running = 0; fc3_running = 0;
        #100;
        rst = 0;
        #20;

        // ══════════════ Pool2 ══════════════
        $display("============================================");
        $display(" Pool2 (10x10x16 -> 5x5x16)");
        $display("============================================");
        pool2_start = 1; #10; pool2_start = 0;
        @(posedge pool2_done);
        $display(" Done at %0t ns", $time);
        #20;

        mismatches = 0;
        for (i = 0; i < POOL2_SIZE; i = i + 1) begin
            pool2_out_addr = i; #10;
            if (pool2_out_data !== pool2_exp[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display(" MISMATCH [%0d] RTL=%0d Exp=%0d", i, $signed(pool2_out_data), $signed(pool2_exp[i]));
            end
        end
        if (mismatches == 0) $display(" PASS: %0d/%0d", POOL2_SIZE, POOL2_SIZE);
        else                 $display(" FAIL: %0d mismatches", mismatches);

        // ══════════════ FC1 ══════════════
        $display("============================================");
        $display(" FC1 (400 -> 120 + ReLU)");
        $display("============================================");
        fc1_running = 1;
        fc1_start = 1; #10; fc1_start = 0;
        @(posedge fc1_done);
        fc1_running = 0;
        $display(" Done at %0t ns", $time);
        #20;

        mismatches = 0;
        for (i = 0; i < FC1_SIZE; i = i + 1) begin
            fc1_out_addr = i; #10;
            if (fc1_out_data !== fc1_exp[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display(" MISMATCH [%0d] RTL=%0d Exp=%0d", i, $signed(fc1_out_data), $signed(fc1_exp[i]));
            end
        end
        if (mismatches == 0) $display(" PASS: %0d/%0d", FC1_SIZE, FC1_SIZE);
        else                 $display(" FAIL: %0d mismatches", mismatches);

        // ══════════════ FC2 ══════════════
        $display("============================================");
        $display(" FC2 (120 -> 84 + ReLU)");
        $display("============================================");
        fc2_running = 1;
        fc2_start = 1; #10; fc2_start = 0;
        @(posedge fc2_done);
        fc2_running = 0;
        $display(" Done at %0t ns", $time);
        #20;

        mismatches = 0;
        for (i = 0; i < FC2_SIZE; i = i + 1) begin
            fc2_out_addr = i; #10;
            if (fc2_out_data !== fc2_exp[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display(" MISMATCH [%0d] RTL=%0d Exp=%0d", i, $signed(fc2_out_data), $signed(fc2_exp[i]));
            end
        end
        if (mismatches == 0) $display(" PASS: %0d/%0d", FC2_SIZE, FC2_SIZE);
        else                 $display(" FAIL: %0d mismatches", mismatches);

        // ══════════════ FC3 ══════════════
        $display("============================================");
        $display(" FC3 (84 -> 10, no ReLU)");
        $display("============================================");
        fc3_running = 1;
        fc3_start = 1; #10; fc3_start = 0;
        @(posedge fc3_done);
        fc3_running = 0;
        $display(" Done at %0t ns", $time);
        #20;

        mismatches = 0;
        for (i = 0; i < FC3_SIZE; i = i + 1) begin
            fc3_out_addr = i; #10;
            if (fc3_out_data !== fc3_exp[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10)
                    $display(" MISMATCH [%0d] RTL=%0d Exp=%0d", i, $signed(fc3_out_data), $signed(fc3_exp[i]));
            end
        end
        if (mismatches == 0) $display(" PASS: %0d/%0d", FC3_SIZE, FC3_SIZE);
        else                 $display(" FAIL: %0d mismatches", mismatches);

        // ══════════════ Argmax ══════════════
        $display("============================================");
        $display(" Argmax");
        $display("============================================");
        begin : argmax_block
            reg signed [15:0] best_val;
            reg [3:0] best_idx;
            best_val = -16'sd32768;
            best_idx = 0;
            for (i = 0; i < 10; i = i + 1) begin
                fc3_out_addr = i; #10;
                $display("   logit[%0d] = %0d", i, $signed(fc3_out_data));
                if ($signed(fc3_out_data) > $signed(best_val)) begin
                    best_val = fc3_out_data;
                    best_idx = i;
                end
            end
            $display(" ────────────────────────────────────────");
            $display(" PREDICTED CLASS: %0d", best_idx);
            $display("============================================");
        end

        #100;
        $finish;
    end

endmodule
