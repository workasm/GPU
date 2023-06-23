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

// s - prologue
// t - code
// l - load ??
// x - epilogue

//#include "stk_stasm.h"

//#define ptxasm_t_(...)
//#define ptxasm_o_(...)
//#define ptxasm_i_(...)

#define ptxasm_t_code(...) #__VA_ARGS__
#define ptxasm_o_code(...)
#define ptxasm_i_code(...)

#define ptxasm_t_i(as, name, mode, ...) ptxasm_t_##mode(__VA_ARGS__)
#define ptxasm_o_i(as, name, mode, ...) ptxasm_o_##mode(__VA_ARGS__)

#define ptxasm_i_i(as, name, mode, ...) #as "(" #name ")", ptxasm_i_i(__VA_ARGS__)

#define ptxasm_t_o(as, name, mode, ...) ptxasm_t_##mode(__VA_ARGS__)
#define ptxasm_o_o(as, name, mode, ...) #as (name), ptxasm_o_##o(__VA_ARGS__)
#define ptxasm_i_o(as, name, mode, ...) ptxasm_i_##mode(__VA_ARGS__)

//! https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms#deferred-expression

#define ptxasm(mode,...)                           \
        ptxasm_t_##mode(__VA_ARGS__) "\noutputs:\n"          \
        ptxasm_o_##mode(__VA_ARGS__)  "\ninputs:\n"          \
        ptxasm_i_##mode(__VA_ARGS__)

# define EMPTY(...)
# define DEFER(...) __VA_ARGS__ EMPTY()
# define OBSTRUCT(...) __VA_ARGS__ DEFER(EMPTY)()
# define EXPAND(...) __VA_ARGS__

# define AA_id() AA
# define AA(a,...) (a+11, DEFER(AA_id)()(__VA_ARGS__))

int main(int argc, char** argv) try {

//    var(r,varA,r,varB) ->
//            [varA] r (varA), [varB] r (varB)

    //EXPAND(EXPAND(EXPAND(AA(1,2,3))))

    //const char *s[] = {ptxasm(i,r,varA,i,r,varB,code,uuu)};
    //const char *s[] = {ptxasm(i,r,varA,code,uuu)};


//    nBodySim(argc, (const char **)argv);

    //const char *ss = XX(qwef,wefw,we,weff,ewf,ewfwef,fwe,f);
    //printf(ss);

    //auto pobj = std::make_unique< GPU_interpolator >();
    auto pobj = std::make_unique< GPU_radixSort >();
    pobj->run();
    return 0;
}
catch(std::exception& ex) {
    fprintf(stderr, "Exception: %s\n", ex.what());
}
catch(...) {
    fprintf(stderr, "Unknown exception");
}
