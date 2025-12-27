// Sparse dot-product engine: streams compressed (index,value) pairs, fetches activations,
// and accumulates using the zero_skip_mac. One instance computes a single output channel.
// Assumes upstream weight store provides only non-zero weights, so MAC operations are never wasted.
module sparse_dot_product #(
    parameter integer INDEX_WIDTH   = 12,
    parameter integer DATA_WIDTH    = 16,
    parameter integer ACC_WIDTH     = 32,
    parameter integer ADDR_WIDTH    = 12,
    parameter integer CLK_FREQ_MHZ  = 200
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        start,
    input  wire [15:0]                 nnz_count,      // number of non-zero entries for this output channel
    output reg                         done,
    // weight store interface
    output reg  [ADDR_WIDTH-1:0]       weight_addr,
    input  wire [INDEX_WIDTH-1:0]      weight_idx,
    input  wire [DATA_WIDTH-1:0]       weight_val,
    input  wire                        weight_valid,
    // activation RAM interface
    output reg  [INDEX_WIDTH-1:0]      activation_addr,
    input  wire [DATA_WIDTH-1:0]       activation_data,
    // result
    output reg  [ACC_WIDTH-1:0]        acc_out,
    output reg                         acc_valid
);
    reg [15:0] count;
    reg clear_acc;
    reg busy;

    wire mac_acc_valid;
    wire [ACC_WIDTH-1:0] mac_acc;
    wire is_last = weight_valid && busy && (count == (nnz_count - 1'b1));

    // Stage: feed MAC
    always @(posedge clk) begin
        if (rst) begin
            weight_addr     <= {ADDR_WIDTH{1'b0}};
            activation_addr <= {INDEX_WIDTH{1'b0}};
            count           <= 16'd0;
            clear_acc       <= 1'b0;
            done            <= 1'b0;
            busy            <= 1'b0;
        end else begin
            done      <= 1'b0;
            clear_acc <= 1'b0;
            if (start) begin
                weight_addr     <= {ADDR_WIDTH{1'b0}};
                activation_addr <= {INDEX_WIDTH{1'b0}};
                count           <= 16'd0;
                clear_acc       <= 1'b1;
                busy            <= 1'b1;
            end else if (weight_valid && (count < nnz_count) && busy) begin
                activation_addr <= weight_idx; // use compressed index to select activation
                weight_addr     <= weight_addr + 1'b1;
                count           <= count + 1'b1;
                if (is_last)
                    busy <= 1'b0;
            end
            if (mac_acc_valid) begin
                acc_out   <= mac_acc;
                acc_valid <= 1'b1;
                if (is_last)
                    done <= 1'b1;
            end else begin
                acc_valid <= 1'b0;
            end
        end
    end

    zero_skip_mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) mac_inst (
        .clk       (clk),
        .rst       (rst),
        .clear_acc (clear_acc),
        .valid     (weight_valid && busy),
        .last      (is_last),
        .weight    (weight_val),
        .activation(activation_data),
        .acc       (mac_acc),
        .acc_valid (mac_acc_valid)
    );
endmodule
