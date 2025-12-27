// Compressed weight storage for sparse networks.
// Stores (index, value) pairs in a single ROM/BRAM word to minimize address traffic.
module sparse_weight_store #(
    parameter integer DEPTH        = 1024,
    parameter integer INDEX_WIDTH  = 12,
    parameter integer VALUE_WIDTH  = 16,
    parameter         INIT_FILE    = ""   // optional hex file with {index,value} packed
)(
    input  wire                        clk,
    input  wire                        en,
    input  wire [$clog2(DEPTH)-1:0]    addr,
    output reg  [INDEX_WIDTH-1:0]      idx_out,
    output reg  [VALUE_WIDTH-1:0]      val_out
);
    reg [INDEX_WIDTH+VALUE_WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        if (en) begin
            {idx_out, val_out} <= mem[addr];
        end
    end
endmodule
