// ============================================================
// tb_avgpool.v — Testbench for Average Pooling (Pool1)
// ============================================================
// Feeds the verified Conv1 output into avgpool, then compares
// all 1176 outputs (6×14×14) against pool1_expected.hex.
// ============================================================

`timescale 1ns / 1ps

module tb_avgpool;

    // ────────────────────────────────────────
    // Clock and reset
    // ────────────────────────────────────────
    reg clk;
    reg rst;
    reg start;

    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    // ────────────────────────────────────────
    // Input memory (Conv1 output, loaded from hex)
    // ────────────────────────────────────────
    localparam IN_SIZE = 4704;  // 6 * 28 * 28
    reg signed [15:0] in_mem [0:IN_SIZE-1];

    initial begin
        $readmemh("conv1_expected.hex", in_mem);
    end

    // DUT reads from in_mem via address port
    wire [12:0] in_addr;
    wire [15:0] in_data = in_mem[in_addr];

    // ────────────────────────────────────────
    // DUT signals
    // ────────────────────────────────────────
    wire        done;
    reg  [12:0] out_addr;
    wire [15:0] out_data;

    // ────────────────────────────────────────
    // Instantiate DUT (Pool1 config)
    // ────────────────────────────────────────
    avgpool #(
        .IN_H  (28),
        .IN_W  (28),
        .N_CH  (6)
    ) dut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .done     (done),
        .in_addr  (in_addr),
        .in_data  (in_data),
        .out_addr (out_addr),
        .out_data (out_data)
    );

    // ────────────────────────────────────────
    // Golden expected output
    // ────────────────────────────────────────
    localparam OUT_SIZE = 1176;  // 6 * 14 * 14

    reg signed [15:0] expected [0:OUT_SIZE-1];

    initial begin
        $readmemh("pool1_expected.hex", expected);
    end

    // ────────────────────────────────────────
    // VCD dump
    // ────────────────────────────────────────
    initial begin
        $dumpfile("avgpool.vcd");
        $dumpvars(0, tb_avgpool);
    end

    // ────────────────────────────────────────
    // Timeout
    // ────────────────────────────────────────
    initial begin
        #500_000;
        $display("ERROR: Simulation timed out.");
        $finish;
    end

    // ────────────────────────────────────────
    // Main test sequence
    // ────────────────────────────────────────
    integer i;
    integer mismatches;

    initial begin
        rst      = 1;
        start    = 0;
        out_addr = 0;
        #100;
        rst = 0;
        #20;

        $display("--------------------------------------------");
        $display(" AvgPool (28x28x6 -> 14x14x6) — Starting...");
        $display("--------------------------------------------");
        start = 1;
        #10;
        start = 0;

        @(posedge done);
        $display(" Done at time %0t ns", $time);
        #20;

        $display("--------------------------------------------");
        $display(" Verifying %0d outputs...", OUT_SIZE);
        $display("--------------------------------------------");

        mismatches = 0;

        for (i = 0; i < OUT_SIZE; i = i + 1) begin
            out_addr = i;
            #10;

            if (out_data !== expected[i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 20) begin
                    $display(" MISMATCH addr=%0d: RTL=0x%04h (%0d)  Expected=0x%04h (%0d)",
                             i,
                             out_data, $signed(out_data),
                             expected[i], $signed(expected[i]));
                end
            end
        end

        $display("--------------------------------------------");
        if (mismatches == 0)
            $display(" PASS: %0d/%0d outputs match golden model", OUT_SIZE, OUT_SIZE);
        else begin
            $display(" FAIL: %0d mismatches out of %0d", mismatches, OUT_SIZE);
            if (mismatches > 20)
                $display("       (only first 20 shown)");
        end
        $display("--------------------------------------------");

        #100;
        $finish;
    end

endmodule
