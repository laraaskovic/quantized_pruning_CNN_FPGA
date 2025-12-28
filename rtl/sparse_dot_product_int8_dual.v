// Dual-lane sparse dot-product engine for int8 weights/activations.
// Streams two (index,value) pairs per cycle from a packed weight store and feeds
// a dual-lane int8 MAC that packs both multiplies into one DSP (SIMD TWO24).
// Requires a dual-port activation RAM (two read addresses per cycle).
module sparse_dot_product_int8_dual #(
    parameter integer INDEX_WIDTH   = 12,
    parameter integer DATA_WIDTH    = 8,
    parameter integer ACC_WIDTH     = 48,
    parameter integer ADDR_WIDTH    = 12,
    parameter integer SCALE_WIDTH   = 24
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        start,
    input  wire [15:0]                 nnz_pairs,     // number of pair-words (ceil(nnz/2))
    input  wire [INDEX_WIDTH-1:0]      tail_idx,      // valid only when nnz is odd; ignored otherwise
    output reg                         done,
    // weight store interface (packed pairs)
    output reg  [ADDR_WIDTH-1:0]       weight_addr,
    input  wire [INDEX_WIDTH-1:0]      weight_idx0,
    input  wire [DATA_WIDTH-1:0]       weight_val0,
    input  wire [INDEX_WIDTH-1:0]      weight_idx1,
    input  wire [DATA_WIDTH-1:0]       weight_val1,
    input  wire                        weight_valid,
    // activation RAM interface (dual-port read)
    output reg  [INDEX_WIDTH-1:0]      activation_addr0,
    output reg  [INDEX_WIDTH-1:0]      activation_addr1,
    input  wire [DATA_WIDTH-1:0]       activation_data0,
    input  wire [DATA_WIDTH-1:0]       activation_data1,
    // scale (Q8.16) for dequantization of the accumulator
    input  wire [SCALE_WIDTH-1:0]      dequant_scale,
    // result
    output reg  [ACC_WIDTH-1:0]        acc_out,
    output reg  [ACC_WIDTH-1:0]        acc_dequant,
    output reg                         acc_valid
);
    reg [15:0] pair_count;
    reg clear_acc;
    reg busy;

    wire mac_acc_valid;
    wire mac_dequant_valid;
    wire [ACC_WIDTH-1:0] mac_acc;
    wire [ACC_WIDTH-1:0] mac_acc_dequant;

    wire is_last_pair = weight_valid && busy && (pair_count == (nnz_pairs - 1'b1));

    // feed MAC + address generation
    always @(posedge clk) begin
        if (rst) begin
            weight_addr       <= {ADDR_WIDTH{1'b0}};
            activation_addr0  <= {INDEX_WIDTH{1'b0}};
            activation_addr1  <= {INDEX_WIDTH{1'b0}};
            pair_count        <= 16'd0;
            clear_acc         <= 1'b0;
            done              <= 1'b0;
            busy              <= 1'b0;
        end else begin
            done      <= 1'b0;
            clear_acc <= 1'b0;
            if (start) begin
                weight_addr      <= {ADDR_WIDTH{1'b0}};
                activation_addr0 <= {INDEX_WIDTH{1'b0}};
                activation_addr1 <= {INDEX_WIDTH{1'b0}};
                pair_count       <= 16'd0;
                clear_acc        <= 1'b1;
                busy             <= 1'b1;
            end else if (weight_valid && (pair_count < nnz_pairs) && busy) begin
                activation_addr0 <= weight_idx0;
                activation_addr1 <= weight_idx1;
                weight_addr      <= weight_addr + 1'b1;
                pair_count       <= pair_count + 1'b1;
                if (is_last_pair)
                    busy <= 1'b0;
            end
            if (mac_acc_valid) begin
                acc_out   <= mac_acc;
                acc_valid <= 1'b1;
                if (mac_dequant_valid)
                    acc_dequant <= mac_acc_dequant;
                if (is_last_pair)
                    done <= 1'b1;
            end else begin
                acc_valid <= 1'b0;
            end
        end
    end

    // tail handling: when nnz is odd, the final pair carries idx0/val0 valid, idx1/val1 zeroed.
    wire tail_active = is_last_pair && (tail_idx != {INDEX_WIDTH{1'b0}});

    int8_dual_zero_skip_mac #(
        .ACC_WIDTH   (ACC_WIDTH),
        .SCALE_WIDTH (SCALE_WIDTH),
        .USE_XIL_DSP (1)
    ) mac_inst (
        .clk                (clk),
        .rst                (rst),
        .clear_acc          (clear_acc),
        .valid0             (weight_valid && busy),
        .valid1             (weight_valid && busy && (!tail_active)), // suppress lane1 on tail when nnz is odd
        .last0              (is_last_pair),
        .last1              (is_last_pair),
        .weight0            (weight_val0),
        .weight1            (tail_active ? 8'sd0 : weight_val1),
        .act0               (activation_data0),
        .act1               (tail_active ? 8'sd0 : activation_data1),
        .dequant_scale      (dequant_scale),
        .acc                (mac_acc),
        .acc_valid          (mac_acc_valid),
        .acc_dequant        (mac_acc_dequant),
        .acc_dequant_valid  (mac_dequant_valid)
    );
endmodule
