#include "stream_src.h"

void stream_src(
    hls::stream<ap_axiu<32,0,0,0>> &out,
    int n)
{
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE s_axilite port=n      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        ap_axiu<32,0,0,0> w;
        w.data = (ap_uint<32>)i;   // counter payload — host verifies out[i] == i
        w.keep = (ap_uint<4>)-1;
        w.strb = (ap_uint<4>)-1;
        w.last = (i == n - 1) ? 1 : 0;
        out.write(w);
    }
}
