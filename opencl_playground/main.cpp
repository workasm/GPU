#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include "include/playground_host.h"

static GPU_playground s_obj;

int main(int argc, char** argv) try {

    std::atexit([]() {
        system("pause");
    });

    s_obj.run(argc, argv);
    return 0;
}
catch (cl::Error& ex) {
    fprintf(stderr, "OpenCL exception: %s\n%s\n", ex.what(), 
        GPU_playground::getErrorStr(ex.err()));
}
catch(std::exception& ex) {
    fprintf(stderr, "Exception: %s\n", ex.what());
}
catch(...) {
    fprintf(stderr, "Unknown exception");
}
