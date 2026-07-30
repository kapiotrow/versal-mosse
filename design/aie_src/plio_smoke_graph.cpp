#include "plio_smoke_graph.h"

// Graph instance name must match the string the host passes to xrt::graph().
PLIO_smoke_graph smoke_graph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void)
{
    smoke_graph.init();
    smoke_graph.run(1);
    smoke_graph.end();
    return 0;
}
#endif
