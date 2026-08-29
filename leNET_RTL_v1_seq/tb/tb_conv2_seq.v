// ============================================================
// tb_conv2_seq.v — Testbench for Conv2 Sequential MAC Engine
// ============================================================

`timescale 1ns / 1ps

module tb_conv2_seq;

    reg clk;
    reg rst;
    reg start;

    initial clk = 0;
    always #5 clk = ~clk;

    // ────────────────────────────────────────
    // Input memory (Pool1 output)
    // ────────────────────────────────────────
    localparam IN_SIZE = 1176;  // 6 * 14 * 14
    reg signed [15:0] in_mem [0:IN_SIZE-1];

    initial begin
        $readmemh("pool1_expected.hex", in_mem);
    end

    wire [12:0] in_addr;
    wire [15:0] in_data = in_mem[in_addr];

    // ────────────────────────────────────────
    // DUT
    // ────────────────────────────────────────
    wire        done;
    reg  [12:0] out_addr;
    wire [15:0] out_data;

    conv2_seq dut (
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
    localparam OUT_SIZE = 1600;  // 16 * 10 * 10
    reg signed [15:0] expected [0:OUT_SIZE-1];

    initial begin
        $readmemh("conv2_expected.hex", expected);
    end

    // ────────────────────────────────────────
    // VCD + timeout
    // ────────────────────────────────────────
    initial begin
        $dumpfile("conv2_seq.vcd");
        $dumpvars(0, tb_conv2_seq);
    end

    initial begin
        #5_000_000;
        $display("ERROR: Simulation timed out.");
        $finish;
    end

    // ────────────────────────────────────────
    // Main test
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
        $display(" Conv2 Sequential (150 MACs/pixel) — Start");
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
