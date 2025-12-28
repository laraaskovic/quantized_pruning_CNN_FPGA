// Dual-lane int8 zero-skipping MAC with optional DSP48E1 SIMD packing (TWO24).
// Packs two 8x8 multiplies into one DSP when targeting Xilinx; falls back to
// two parallel multiplies otherwise. Accumulator is signed; dequantization
// applies a Q8.16 scale (e.g., act_scale * weight_scale) to emit a real-valued
// fixed-point result.
module int8_dual_zero_skip_mac #(
    parameter integer ACC_WIDTH    = 48,
    parameter integer SCALE_WIDTH  = 24,  // Q8.16 dequant scale
    parameter         USE_XIL_DSP  = 1
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     clear_acc,
    input  wire                     valid0,
    input  wire                     valid1,
    input  wire                     last0,
    input  wire                     last1,
    input  wire signed [7:0]        weight0,
    input  wire signed [7:0]        weight1,
    input  wire signed [7:0]        act0,
    input  wire signed [7:0]        act1,
    input  wire [SCALE_WIDTH-1:0]   dequant_scale, // Q8.16
    output reg  [ACC_WIDTH-1:0]     acc,
    output reg                      acc_valid,
    output reg  [ACC_WIDTH-1:0]     acc_dequant,
    output reg                      acc_dequant_valid
);
    wire signed [7:0] w0_eff = valid0 ? weight0 : 8'sd0;
    wire signed [7:0] w1_eff = valid1 ? weight1 : 8'sd0;
    wire signed [7:0] a0_eff = valid0 ? act0    : 8'sd0;
    wire signed [7:0] a1_eff = valid1 ? act1    : 8'sd0;

    wire signed [17:0] pair_sum;

`ifdef XILINX_SIMULATOR
    localparam USE_DSP = 1;
`else
    localparam USE_DSP = USE_XIL_DSP;
`endif

    generate
        if (USE_DSP) begin : g_dsp48
            // Pack two 8-bit lanes into A[29:0] and B[17:0] and use SIMD TWO24.
            // A lane order: {sign-extended act1, sign-extended act0}
            // B lane order: {sign-extended w1, sign-extended w0}
            wire [29:0] dsp_a = { {6{a1_eff[7]}}, a1_eff, {6{a0_eff[7]}}, a0_eff };
            wire [17:0] dsp_b = { {2{w1_eff[7]}}, w1_eff, {2{w0_eff[7]}}, w0_eff };
            wire [47:0] dsp_p;

            DSP48E1 #(
                .USE_SIMD("TWO24"),
                .AUTORESET_PATDET("NO_RESET"),
                .MASK(48'h3fffffffffff),
                .PATTERN(48'h000000000000),
                .USE_MULT("MULTIPLY")
            ) dsp48e1_inst (
                .CLK   (clk),
                .A     (dsp_a),
                .B     (dsp_b),
                .C     (48'd0),
                .ALUMODE(4'b0000),
                .CARRYIN(1'b0),
                .CARRYINSEL(3'b000),
                .CEA1  (1'b1),
                .CEA2  (1'b1),
                .CEB1  (1'b1),
                .CEB2  (1'b1),
                .CEC   (1'b0),
                .CEM   (1'b1),
                .CEP   (1'b1),
                .INMODE(5'b00000),
                .OPMODE(7'b0000101), // P = A*B
                .P     (dsp_p),
                .PCIN  (48'd0),
                .RSTA  (1'b0),
                .RSTB  (1'b0),
                .RSTC  (1'b0),
                .RSTM  (1'b0),
                .RSTP  (1'b0)
            );
            // Two 24-bit results packed into dsp_p; sum them for the accumulator.
            assign pair_sum = $signed(dsp_p[23:0]) + $signed(dsp_p[47:24]);
        end else begin : g_soft
            wire signed [15:0] p0 = a0_eff * w0_eff;
            wire signed [15:0] p1 = a1_eff * w1_eff;
            assign pair_sum = $signed(p0) + $signed(p1);
        end
    endgenerate

    wire is_last = (valid0 && last0) || (valid1 && last1);

    always @(posedge clk) begin
        if (rst) begin
            acc               <= {ACC_WIDTH{1'b0}};
            acc_valid         <= 1'b0;
            acc_dequant       <= {ACC_WIDTH{1'b0}};
            acc_dequant_valid <= 1'b0;
        end else begin
            acc_valid         <= 1'b0;
            acc_dequant_valid <= 1'b0;
            if (clear_acc)
                acc <= {ACC_WIDTH{1'b0}};
            if (valid0 || valid1)
                acc <= acc + {{(ACC_WIDTH-18){pair_sum[17]}}, pair_sum}; // sign-extend
            if (is_last) begin
                acc_valid <= 1'b1;
                // Dequantize: (acc * scale)>>16, where scale is Q8.16
                acc_dequant <= ( ( $signed(acc) * $signed({1'b0, dequant_scale}) ) >>> 16 );
                acc_dequant_valid <= 1'b1;
            end
        end
    end
endmodule
