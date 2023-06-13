
macro(X_CUDA_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if("x${arg}" STREQUAL "xOPTIONS")
      set( _found_options TRUE )
    elseif(
        "x${arg}" STREQUAL "xWIN32" OR
        "x${arg}" STREQUAL "xMACOSX_BUNDLE" OR
        "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
        "x${arg}" STREQUAL "xSTATIC" OR
        "x${arg}" STREQUAL "xSHARED" OR
        "x${arg}" STREQUAL "xMODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
        message(STATUS "--- file: ${arg}")
      endif()
    endif()
  endforeach()
endmacro()

macro(X_CUDA_INCLUDE_NVCC_DEPENDENCIES dependency_file)
  set(CUDA_NVCC_DEPEND)
  set(CUDA_NVCC_DEPEND_REGENERATE FALSE)
  # Include the dependency file.  Create it first if it doesn't exist .  The
  # INCLUDE puts a dependency that will force CMake to rerun and bring in the
  # new info when it changes.  DO NOT REMOVE THIS (as I did and spent a few
  # hours figuring out why it didn't work.
  if(NOT EXISTS ${dependency_file})
    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
  endif()
  # Always include this file to force CMake to run again next
  # invocation and rebuild the dependencies.
  #message("including dependency_file = ${dependency_file}")
  include(${dependency_file})

  # Now we need to verify the existence of all the included files
  # here.  If they aren't there we need to just blank this variable and
  # make the file regenerate again.
#   if(DEFINED CUDA_NVCC_DEPEND)
#     message("CUDA_NVCC_DEPEND set")
#   else()
#     message("CUDA_NVCC_DEPEND NOT set")
#   endif()
  if(CUDA_NVCC_DEPEND)
    #message("CUDA_NVCC_DEPEND found")
    foreach(f ${CUDA_NVCC_DEPEND})
      # message("searching for ${f}")
      if(NOT EXISTS ${f})
        #message("file ${f} not found")
        set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
      endif()
    endforeach()
  else()
    #message("CUDA_NVCC_DEPEND false")
    # No dependencies, so regenerate the file.
    set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
  endif()

  #message("CUDA_NVCC_DEPEND_REGENERATE = ${CUDA_NVCC_DEPEND_REGENERATE}")
  # No incoming dependencies, so we need to generate them.  Make the
  # output depend on the dependency file itself, which should cause the
  # rule to re-run.
  if(CUDA_NVCC_DEPEND_REGENERATE)
    set(CUDA_NVCC_DEPEND ${dependency_file})
    #message("Generating an empty dependency_file: ${dependency_file}")
    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
  endif()

endmacro()

macro(X_CUDA_PARSE_NVCC_OPTIONS _option_prefix)
  set( _found_config )
  foreach(arg ${ARGN})
    # Determine if we are dealing with a perconfiguration flag
    foreach(config ${CUDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      if (arg STREQUAL "${config_upper}")
        set( _found_config _${arg})
        # Set arg to nothing to keep it from being processed further
        set( arg )
      endif()
    endforeach()

    if ( arg )
      list(APPEND ${_option_prefix}${_found_config} "${arg}")
    endif()
  endforeach()
endmacro()


macro(NEW_CUDA_WRAP_SRCS cuda_target format generated_files)

    # Put optional arguments in list.
    set(_argn_list "${ARGN}")
    # If one of the given optional arguments is "PHONY", make a note of it, then
    # remove it from the list.
    set(_target_is_phony true)
     # Set up all the command line flags here, so that they can be overridden on a per target basis.

    if(CMAKE_GENERATOR MATCHES "Visual Studio")
       set(_CUDA_MSVC_HOST_COMPILER "$(VCInstallDir)Tools/MSVC/$(VCToolsVersion)/bin/Host$(Platform)/$(PlatformTarget)")
       if(MSVC_VERSION LESS 1910)
         set(_CUDA_MSVC_HOST_COMPILER "$(VCInstallDir)bin")
       endif()
    endif()

    #set(CUDA_HOST_COMPILER "${_CUDA_MSVC_HOST_COMPILER}" CACHE FILEPATH "Host side compiler used by NVCC")

    set(nvcc_flags "")
    set(generated_extension ${CMAKE_CXX_OUTPUT_EXTENSION})

    if(CUDA_64_BIT_DEVICE_CODE)
      set(nvcc_flags ${nvcc_flags} -m64)
    else()
      set(nvcc_flags ${nvcc_flags} -m32)
    endif()

    if(CUDA_TARGET_CPU_ARCH)
      set(nvcc_flags ${nvcc_flags} "--target-cpu-architecture=${CUDA_TARGET_CPU_ARCH}")
    endif()

    # This needs to be passed in at this stage, because VS needs to fill out the
    # various macros from within VS.  Note that CCBIN is only used if
    # -ccbin or --compiler-bindir isn't used and CUDA_HOST_COMPILER matches
    # _CUDA_MSVC_HOST_COMPILER
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
      set(ccbin_flags -D "\"CCBIN:PATH=${_CUDA_MSVC_HOST_COMPILER}\"" )
    else()
      set(ccbin_flags)
    endif()

    # Initialize our list of includes with the user ones followed by the CUDA system ones.
    set(CUDA_NVCC_INCLUDE_DIRS ${CUDA_NVCC_INCLUDE_DIRS_USER} "${CUDA_INCLUDE_DIRS}")
    if(_target_is_phony)
      # If the passed in target name isn't a real target (i.e., this is from a call to one of the
      # cuda_compile_* functions), need to query directory properties to get include directories
      # and compile definitions.
      get_directory_property(_dir_include_dirs INCLUDE_DIRECTORIES)
      get_directory_property(_dir_compile_defs COMPILE_DEFINITIONS)

      list(APPEND CUDA_NVCC_INCLUDE_DIRS "${_dir_include_dirs}")
      set(CUDA_NVCC_COMPILE_DEFINITIONS "${_dir_compile_defs}")
    else()
      # Append the include directories for this target via generator expression, which is
      # expanded by the FILE(GENERATE) call below.  This generator expression captures all
      # include dirs set by the user, whether via directory properties or target properties
      list(APPEND CUDA_NVCC_INCLUDE_DIRS "$<TARGET_PROPERTY:${cuda_target},INCLUDE_DIRECTORIES>")

      # Do the same thing with compile definitions
      set(CUDA_NVCC_COMPILE_DEFINITIONS "$<TARGET_PROPERTY:${cuda_target},COMPILE_DEFINITIONS>")
    endif()

    set(CUDA_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
    list(REMOVE_DUPLICATES CUDA_configuration_types)

    # Reset these variables
    set(CUDA_WRAP_OPTION_NVCC_FLAGS)

    # Figure out if we are building a shared library.  BUILD_SHARED_LIBS is
    # respected in CUDA_ADD_LIBRARY.
    set(_cuda_build_shared_libs FALSE)

    # Loop over all the configuration types to generate appropriate flags for run_nvcc.cmake
    foreach(config ${CUDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      # CMAKE_FLAGS are strings and not lists.  By not putting quotes around CMAKE_FLAGS
      # we convert the strings to lists (like we want).
      if(CUDA_PROPAGATE_HOST_FLAGS)
        set(_cuda_C_FLAGS "${CMAKE_CXX_FLAGS_${config_upper}}")
        set(CMAKE_HOST_FLAGS_${config_upper} ${_cuda_C_FLAGS})
      endif()

    endforeach()

    # Process the C++11 flag.  If the host sets the flag, we need to add it to nvcc and
    # remove it from the host. This is because -Xcompile -std=c++ will choke nvcc (it uses
    # the C preprocessor).  In order to get this to work correctly, we need to use nvcc's
    # specific c++11 flag. 
    if( "${_cuda_host_flags}" MATCHES "-std=c\\+\\+11")
      # Add the c++11 flag to nvcc if it isn't already present.  Note that we only look at
      # the main flag instead of the configuration specific flags.
      if( NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND nvcc_flags --std c++11)
      endif()
      string(REGEX REPLACE "[-]+std=c\\+\\+11" "" _cuda_host_flags "${_cuda_host_flags}")
    endif()

    set(nvcc_host_compiler_flags "")
    string(TOUPPER ${CMAKE_BUILD_TYPE} build_type_upper)
	
    set(COMMON_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_HOST_FLAGS_${build_type_upper}}")
    string(REPLACE " " ";" COMMON_FLAGS ${COMMON_FLAGS})

    foreach(flag ${COMMON_FLAGS})
      # Extra quotes are added around each flag to help nvcc parse out flags with spaces.
      message(STATUS "Xflag: ${flag}")
      #string(APPEND nvcc_host_compiler_flags ",\"${flag}\"")
    endforeach()
    set(nvcc_host_compiler_flags "${COMMON_FLAGS}")

    #message("nvcc_host_compiler_flags = \"${nvcc_host_compiler_flags}\"")
    # Add the build specific configuration flags
    list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${build_type_upper}})

  # Reset the output variable
  set(_cuda_wrap_generated_files "")

  # Put optional arguments in list.
  foreach(file ${_argn_list})
        message(STATUS "XCUDA_FILE ${file}")

        cuda_compute_build_path("${file}" cuda_build_path)
        set(cuda_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${cuda_target}.dir")

        set(main_dep MAIN_DEPENDENCY ${file})

        get_filename_component( basename ${file} NAME )
        set(generated_file_path "${cuda_compile_output_dir}/")
        set(generated_file_basename "${cuda_target}_generated_${basename}.obj")
        # Set all of our file names.  Make sure that whatever filenames that have
        # generated_file_path in them get passed in through as a command line
        # argument, so that the ${CMAKE_CFG_INTDIR} gets expanded at run time
        # instead of configure time.
        set(generated_file "${generated_file_path}/${generated_file_basename}")

        #add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/Generated.hpp"
        #    COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/GenerateHeader.py" --argument
        #    DEPENDS some_target)

        set(CUDA_NVCC_INCLUDE_ARGS)
        foreach(dir ${CUDA_NVCC_INCLUDE_DIRS})
          # Extra quotes are added around each flag to help nvcc parse out flags with spaces.
          list(APPEND CUDA_NVCC_INCLUDE_ARGS "-I${dir}")
        endforeach()

		add_custom_command(
          OUTPUT ${generated_file}
          # These output files depend on the source_file and the contents of cmake_dependency_file
          ${main_dep}
          #DEPENDS ${CUDA_NVCC_DEPEND}
          #DEPENDS ${custom_target_script}
          # Make sure the output directory exists before trying to write to it.
          #COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
          COMMAND ${CMAKE_CXX_COMPILER} ARGS
            -D__CUDACC__
            ${file}
            ${cuda_language_flag}
            -c -o "${generated_file}"
            ${nvcc_flags}
            ${CUDA_NVCC_FLAGS}
			${nvcc_host_compiler_flags}
            --cuda-path="${CUDA_TOOLKIT_ROOT_DIR}"
            ${CUDA_NVCC_INCLUDE_ARGS}
          #WORKING_DIRECTORY "${cuda_compile_intermediate_directory}"
          #COMMENT "${cuda_build_comment_string}"
          #${_verbatim}
          )

      #-ccbin "C:/Program Files (x86)/Microsoft Visual Studio/2017/Professional/VC/Tools/MSVC/14.16.27023/bin/HostX64/x64"
	  
	  # clang invocation:
	  # ${CMAKE_CXX_COMPILER} --cuda-path="${CUDA_TOOLKIT_ROOT_DIR}" 
	  #"C:\Program Files\LLVM\bin\clang.exe" --cuda-path="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4" -c -o bla.obj --cuda-gpu-arch=sm_35 -Xcuda-ptxas -v interpolate_gpu.cu
	  # use option clang bla.cu -S in order to generate assembly output	

      message(STATUS "nvcc_flags = ${nvcc_flags}")
      message(STATUS "nvcc_host_compiler_flags = ${nvcc_host_compiler_flags}")
      message(STATUS "CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")
      message(STATUS "CUDA_NVCC_INCLUDE_ARGS = ${CUDA_NVCC_INCLUDE_ARGS}")
      message(STATUS "CCBIN = ${CCBIN}")

        set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)
        list(APPEND _cuda_wrap_generated_files ${generated_file})
    endforeach()
    set(${generated_files} ${_cuda_wrap_generated_files})
endmacro()

# special CUDA compile for OBJ
macro(x_cuda_compile_base cuda_target format generated_files)
  # Update a counter in this directory, to keep phony target names unique.
  set(_cuda_target "${cuda_target}")
  get_property(_counter DIRECTORY PROPERTY _cuda_internal_phony_counter)
  if(_counter)
    math(EXPR _counter "${_counter} + 1")
  else() 
    set(_counter 1)
  endif()
  string(APPEND _cuda_target "_${_counter}")
  set_property(DIRECTORY PROPERTY _cuda_internal_phony_counter ${_counter})

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options )

  # Create custom commands and targets for each file.
  NEW_CUDA_WRAP_SRCS( ${_cuda_target} ${format} _generated_files ${ARGN})

  set( ${generated_files} ${_generated_files})

endmacro()

macro(X_CUDA_COMPILE generated_files)
  #message(STATUS "exec ----${CUDA_NVCC_EXECUTABLE} - ${CUDA_TOOLKIT_ROOT_DIR}") 
  x_cuda_compile_base(cuda_compile OBJ ${generated_files} ${ARGN})
endmacro()

