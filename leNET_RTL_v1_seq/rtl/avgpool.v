// ============================================================
// avgpool.v — 2×2 Average Pooling (Sequential)
// ============================================================
// Parameterized: works for any input size and channel count.
//   Pool1: IN_H=28, IN_W=28, N_CH=6  → 14×14×6
//   Pool2: IN_H=10, IN_W=10, N_CH=16 → 5×5×16
//
// Operation per output pixel:
//   avg = (a + b + c + d + 2) >> 2
// where a,b,c,d are the four Q7.9 values in a 2×2 window,
// and +2 is the rounding bias (0.5 LSB at the >>2 output).
//
// FSM: IDLE → COMPUTE → DONE
// One output pixel per clock cycle (no multiplier needed).
// ============================================================

module avgpool #(
    parameter IN_H  = 28,
    parameter IN_W  = 28,
    parameter N_CH  = 6
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,

    // Input read port (reads from upstream layer's output)
    output wire [12:0] in_addr,
    input  wire [15:0] in_data,

    // Output read port (downstream layer reads results)
    input  wire [12:0] out_addr,
    output wire [15:0] out_data
);

    // ────────────────────────────────────────
    // Derived parameters
    // ────────────────────────────────────────
    localparam OUT_H    = IN_H / 2;
    localparam OUT_W    = IN_W / 2;
    localparam OUT_SIZE = N_CH * OUT_H * OUT_W;

    // ────────────────────────────────────────
    // FSM states
    // ────────────────────────────────────────
    localparam S_IDLE    = 3'd0;
    localparam S_READ_A  = 3'd1;
    localparam S_READ_B  = 3'd2;
    localparam S_READ_C  = 3'd3;
    localparam S_READ_D  = 3'd4;
    localparam S_WRITE   = 3'd5;
    localparam S_DONE    = 3'd6;

    reg [2:0] state;

    // ────────────────────────────────────────
    // Output memory
    // ────────────────────────────────────────
    reg signed [15:0] out_mem [0:OUT_SIZE-1];

    assign out_data = out_mem[out_addr];

    // ────────────────────────────────────────
    // Loop counters
    // ────────────────────────────────────────
    reg [4:0] ch;      // channel
    reg [4:0] row;     // output row
    reg [4:0] col;     // output col

    // ────────────────────────────────────────
    // Datapath
    // ────────────────────────────────────────
    reg signed [17:0] acc;  // sum of 4 Q7.9 values, needs 18 bits

    // Input address generation
    // We read 4 pixels from the 2×2 window in sequence:
    //   A = (2*row,   2*col)
    //   B = (2*row,   2*col+1)
    //   C = (2*row+1, 2*col)
    //   D = (2*row+1, 2*col+1)
    // Base address in channel-major layout: ch * IN_H * IN_W + r * IN_W + c
    reg [12:0] rd_addr;
    assign in_addr = rd_addr;

    wire [4:0] base_r = row * 2;
    wire [4:0] base_c = col * 2;

    // Output address: ch * OUT_H * OUT_W + row * OUT_W + col
    wire [12:0] wr_addr = ch * (OUT_H * OUT_W) + row * OUT_W + col;

    // Average with rounding: (sum + 2) >> 2
    wire signed [15:0] avg_result = (acc + 18'sd2) >>> 2;

    // ────────────────────────────────────────
    // FSM
    // ────────────────────────────────────────
    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            done    <= 1'b0;
            ch      <= 0;
            row     <= 0;
            col     <= 0;
            acc     <= 0;
            rd_addr <= 0;
        end
        else begin
            case (state)

                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        ch  <= 0;
                        row <= 0;
                        col <= 0;
                        acc <= 0;
                        // Set up address for pixel A
                        rd_addr <= 0;  // ch=0, row=0, col=0 → addr 0
                        state   <= S_READ_A;
                    end
                end

                // Read pixel A: top-left of 2×2 window
                S_READ_A: begin
                    rd_addr <= ch * (IN_H * IN_W) + base_r * IN_W + base_c;
                    state   <= S_READ_B;
                end

                // Capture A, set up address for B: top-right
                S_READ_B: begin
                    acc     <= {{2{in_data[15]}}, in_data};  // sign-extend to 18 bits
                    rd_addr <= ch * (IN_H * IN_W) + base_r * IN_W + base_c + 1;
                    state   <= S_READ_C;
                end

                // Capture B, set up address for C: bottom-left
                S_READ_C: begin
                    acc     <= acc + {{2{in_data[15]}}, in_data};
                    rd_addr <= ch * (IN_H * IN_W) + (base_r + 1) * IN_W + base_c;
                    state   <= S_READ_D;
                end

                // Capture C, set up address for D: bottom-right
                S_READ_D: begin
                    acc     <= acc + {{2{in_data[15]}}, in_data};
                    rd_addr <= ch * (IN_H * IN_W) + (base_r + 1) * IN_W + base_c + 1;
                    state   <= S_WRITE;
                end

                // Capture D, compute average, store
                S_WRITE: begin
                    acc <= acc + {{2{in_data[15]}}, in_data};
                    // avg_result uses the updated acc — but acc updates next cycle
                    // so we compute inline here:
                    out_mem[wr_addr] <= ((acc + {{2{in_data[15]}}, in_data}) + 18'sd2) >>> 2;

                    // Advance counters
                    if (col == OUT_W - 1) begin
                        col <= 0;
                        if (row == OUT_H - 1) begin
                            row <= 0;
                            if (ch == N_CH - 1) begin
                                state <= S_DONE;
                            end
                            else begin
                                ch    <= ch + 1;
                                acc   <= 0;
                                state <= S_READ_A;
                            end
                        end
                        else begin
                            row   <= row + 1;
                            acc   <= 0;
                            state <= S_READ_A;
                        end
                    end
                    else begin
                        col   <= col + 1;
                        acc   <= 0;
                        state <= S_READ_A;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                end

            endcase
        end
    end

endmodule
