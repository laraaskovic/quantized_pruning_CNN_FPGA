// Zero-skipping MAC: accumulates only when valid is asserted.
// Designed to sit behind a compressed weight reader; no zero values are ever presented.
module zero_skip_mac #(
    parameter integer DATA_WIDTH = 16,
    parameter integer ACC_WIDTH  = 32
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        clear_acc,   // pulse to zero the accumulator for a new output channel
    input  wire                        valid,       // asserted when weight/activation pair is valid (non-zero weight)
    input  wire                        last,        // asserted with valid on the final element of the dot product
    input  wire signed [DATA_WIDTH-1:0] weight,
    input  wire signed [DATA_WIDTH-1:0] activation,
    output reg  [ACC_WIDTH-1:0]        acc,
    output reg                         acc_valid
);
    wire signed [ACC_WIDTH-1:0] product = weight * activation; // infers DSP block

    always @(posedge clk) begin
        if (rst) begin
            acc       <= {ACC_WIDTH{1'b0}};
            acc_valid <= 1'b0;
        end else begin
            acc_valid <= 1'b0;
            if (clear_acc)
                acc <= {ACC_WIDTH{1'b0}};
            if (valid)
                acc <= acc + product;
            if (valid && last)
                acc_valid <= 1'b1; // single-cycle pulse when accumulation for this channel is complete
        end
    end
endmodule
