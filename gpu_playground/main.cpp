// ============================================================================
//
// Copyright (c) 2001-2008 Max-Planck-Institut Saarbruecken (Germany).
// All rights reserved.
//
// this file is not part of any library ;-)
//
// ----------------------------------------------------------------------------
//
// Library       : CUDA MP
//
// File          : 
//
// Author(s)     : Pavel Emeliyanenko <asm@mpi-sb.mpg.de>
//
// ============================================================================

#include "playground_host.h"
#include <memory>

int main(int argc, char** argv) try {

    auto pobj = std::make_unique< GPU_interpolator >();
    pobj->run();
    return 0;
}
catch(std::exception& ex) {
    fprintf(stderr, "Exception: %s", ex.what());
}
catch(...) {
    fprintf(stderr, "Unknown exception");
}
