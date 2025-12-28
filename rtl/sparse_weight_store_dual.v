// Dual-entry compressed weight storage for sparse networks.
// Packs two (index,value) pairs into one word for higher throughput / SIMD MACs.
// Word format (LSB-first):
//   [19:0]  : {idx0[11:0], val0[7:0]}
//   [39:20] : {idx1[11:0], val1[7:0]}
module sparse_weight_store_dual #(
    parameter integer DEPTH_PAIRS  = 512,   // number of pair-words; total entries = 2*DEPTH_PAIRS
    parameter integer INDEX_WIDTH  = 12,
    parameter integer VALUE_WIDTH  = 8,
    parameter         INIT_FILE    = ""     // optional hex file with packed words
)(
    input  wire                          clk,
    input  wire                          en,
    input  wire [$clog2(DEPTH_PAIRS)-1:0] addr,
    output reg  [INDEX_WIDTH-1:0]        idx0_out,
    output reg  [VALUE_WIDTH-1:0]        val0_out,
    output reg  [INDEX_WIDTH-1:0]        idx1_out,
    output reg  [VALUE_WIDTH-1:0]        val1_out
);
    localparam integer WORD_WIDTH = 2 * (INDEX_WIDTH + VALUE_WIDTH); // 40 bits with defaults
    reg [WORD_WIDTH-1:0] mem [0:DEPTH_PAIRS-1];

    initial begin
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        if (en) begin
            {idx1_out, val1_out, idx0_out, val0_out} <= mem[addr];
        end
    end
endmodule
