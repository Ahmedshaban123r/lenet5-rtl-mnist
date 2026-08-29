// ============================================================
// tb_lenet5_top.v — End-to-End LeNet-5 Testbench
// ============================================================
// Starts inference, waits for done, checks predicted digit.
// ============================================================

`timescale 1ns / 1ps

module tb_lenet5_top;

    reg  clk;
    reg  rst;
    reg  start;
    wire done;
    wire [3:0] digit;

    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    lenet5_top dut (
        .clk   (clk),
        .rst   (rst),
        .start (start),
        .done  (done),
        .digit (digit)
    );

    // VCD
    initial begin
        $dumpfile("lenet5_top.vcd");
        $dumpvars(0, tb_lenet5_top);
    end

    // Timeout (all layers ~4M cycles at 10ns = 40ms)
    initial begin
        #50_000_000;
        $display("ERROR: Simulation timed out.");
        $finish;
    end

    // Expected result
    localparam EXPECTED_DIGIT = 4'd7;

    initial begin
        rst   = 1;
        start = 0;
        #100;
        rst = 0;
        #20;

        $display("============================================");
        $display(" LeNet-5 End-to-End Inference");
        $display(" Image: MNIST test[0] (true label = 7)");
        $display("============================================");
        $display(" Starting at time %0t ns", $time);

        start = 1;
        #10;
        start = 0;

        @(posedge done);
        $display(" Done at time %0t ns", $time);
        $display(" Total cycles: %0d", ($time - 120) / 10);
        $display("============================================");
        $display(" PREDICTED DIGIT: %0d", digit);
        $display("============================================");

        if (digit == EXPECTED_DIGIT)
            $display(" PASS — correct classification!");
        else
            $display(" FAIL — expected %0d, got %0d", EXPECTED_DIGIT, digit);

        $display("============================================");
        #100;
        $finish;
    end

endmodule
