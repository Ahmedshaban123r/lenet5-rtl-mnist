// ============================================================
// tb_conv1_seq.v — Testbench for Conv1 Sequential MAC Engine
// ============================================================
// 1. Instantiates conv1_seq (which loads hex files internally)
// 2. Pulses 'start'
// 3. Waits for 'done'
// 4. Reads all 4704 outputs and compares vs conv1_expected.hex
// 5. Reports PASS/FAIL with mismatch details
// 6. Dumps VCD waveform to build/conv1_seq.vcd
// ============================================================

`timescale 1ns / 1ps

module tb_conv1_seq;

    // ────────────────────────────────────────
    // Clock and reset
    // ────────────────────────────────────────
    reg clk;
    reg rst;
    reg start;

    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz (10ns period)

    // ────────────────────────────────────────
    // DUT signals
    // ────────────────────────────────────────
    wire        done;
    reg  [12:0] out_addr;
    wire [15:0] out_data;

    // ────────────────────────────────────────
    // Instantiate DUT
    // ────────────────────────────────────────
    conv1_seq dut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .done     (done),
        .out_addr (out_addr),
        .out_data (out_data)
    );

    // ────────────────────────────────────────
    // Golden expected output
    // ────────────────────────────────────────
    localparam OUT_SIZE = 4704;  // 6 * 28 * 28

    reg signed [15:0] expected [0:OUT_SIZE-1];

    initial begin
        $readmemh("conv1_expected.hex", expected);
    end

    // ────────────────────────────────────────
    // VCD dump
    // ────────────────────────────────────────
    initial begin
        $dumpfile("conv1_seq.vcd");
        $dumpvars(0, tb_conv1_seq);
    end

    // ────────────────────────────────────────
    // Timeout watchdog
    // ────────────────────────────────────────
    initial begin
        // 117600 MAC + 4704 WRITE + margin ≈ 150k cycles × 10ns = 1.5ms
        #2_000_000;
        $display("ERROR: Simulation timed out — 'done' never asserted.");
        $finish;
    end

    // ────────────────────────────────────────
    // Main test sequence
    // ────────────────────────────────────────
    integer i;
    integer mismatches;
    integer total_checked;

    initial begin
        // ── Reset ──
        rst      = 1;
        start    = 0;
        out_addr = 0;
        #100;
        rst = 0;
        #20;

        // ── Start conv1 ──
        $display("--------------------------------------------");
        $display(" Conv1 Sequential — Starting inference...");
        $display("--------------------------------------------");
        start = 1;
        #10;
        start = 0;

        // ── Wait for done ──
        @(posedge done);
        $display(" Done asserted at time %0t ns", $time);
        $display(" Total cycles: %0d", ($time - 120) / 10);
        // 120ns = reset(100) + gap(20), 10ns = clock period
        #20;

        // ── Verify outputs ──
        $display("--------------------------------------------");
        $display(" Verifying %0d outputs...", OUT_SIZE);
        $display("--------------------------------------------");

        mismatches    = 0;
        total_checked = 0;

        for (i = 0; i < OUT_SIZE; i = i + 1) begin
            out_addr = i;
            #10;  // one clock cycle for read

            if (out_data !== expected[i]) begin
                mismatches = mismatches + 1;
                // Print first 20 mismatches to avoid flooding console
                if (mismatches <= 20) begin
                    $display(" MISMATCH addr=%0d: RTL=0x%04h (%0d)  Expected=0x%04h (%0d)",
                             i,
                             out_data, $signed(out_data),
                             expected[i], $signed(expected[i]));
                end
            end

            total_checked = total_checked + 1;
        end

        // ── Summary ──
        $display("--------------------------------------------");
        if (mismatches == 0) begin
            $display(" PASS: %0d/%0d outputs match golden model",
                     total_checked, OUT_SIZE);
        end
        else begin
            $display(" FAIL: %0d mismatches out of %0d outputs",
                     mismatches, OUT_SIZE);
            if (mismatches > 20)
                $display("       (only first 20 shown above)");
        end
        $display("--------------------------------------------");

        #100;
        $finish;
    end

endmodule