// ============================================================
// argmax.v — Find index of maximum among 10 values
// ============================================================
// Reads 10 Q7.9 values from an input port sequentially,
// outputs the index (0–9) of the largest value.
// ============================================================

module argmax (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,

    // Input read port (reads FC3 output)
    output reg  [12:0] in_addr,
    input  wire [15:0] in_data,

    // Result
    output reg  [3:0]  digit
);

    localparam N = 10;

    localparam S_IDLE  = 2'd0;
    localparam S_READ  = 2'd1;
    localparam S_COMP  = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0] state;
    reg [3:0] idx;
    reg signed [15:0] best_val;
    reg [3:0] best_idx;

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            idx      <= 0;
            best_val <= -16'sd32768;
            best_idx <= 0;
            in_addr  <= 0;
            digit    <= 0;
        end
        else begin
            case (state)

                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        idx      <= 0;
                        best_val <= -16'sd32768;
                        best_idx <= 0;
                        in_addr  <= 0;
                        state    <= S_READ;
                    end
                end

                // Wait one cycle for data to appear
                S_READ: begin
                    state <= S_COMP;
                end

                // Compare and advance
                S_COMP: begin
                    if ($signed(in_data) > $signed(best_val)) begin
                        best_val <= in_data;
                        best_idx <= idx;
                    end

                    if (idx == N - 1) begin
                        // Update digit — if current is better, use idx, else keep best_idx
                        if ($signed(in_data) > $signed(best_val))
                            digit <= idx;
                        else
                            digit <= best_idx;
                        state <= S_DONE;
                    end
                    else begin
                        idx     <= idx + 1;
                        in_addr <= idx + 1;
                        state   <= S_READ;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                end

            endcase
        end
    end

endmodule
