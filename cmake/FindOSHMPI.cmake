# Finds OSHMPI and sets:
#
#   OSHMPI_FOUND
#   OSHMPI_INCLUDE_DIRS
#   OSHMPI_LIBRARIES
#   OSHMPI_LIBRARY_DIRS
#   OSHMPI_OSHCC

set(OSHMPI_HINT_PATHS
  ${OSHMPI_DIR}
  ${OSHMPI_ROOT}
  $ENV{OSHMPI_DIR}
  $ENV{OSHMPI_ROOT}
  $ENV{OSHMPI_HOME}
)
list(REMOVE_ITEM OSHMPI_HINT_PATHS "")

find_path(OSHMPI_INCLUDE_DIR
  NAMES shmem.h
  HINTS ${OSHMPI_HINT_PATHS}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

find_path(OSHMPI_SHMEMX_INCLUDE_DIR
  NAMES shmemx.h
  HINTS ${OSHMPI_HINT_PATHS}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

find_library(OSHMPI_LIBRARY
  NAMES oshmpi liboshmpi
  HINTS ${OSHMPI_HINT_PATHS}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
)

find_program(OSHMPI_OSHCC
  NAMES oshcc
  HINTS ${OSHMPI_HINT_PATHS}
  PATH_SUFFIXES bin
  NO_DEFAULT_PATH
)

if(OSHMPI_LIBRARY)
  get_filename_component(OSHMPI_LIBRARY_DIR "${OSHMPI_LIBRARY}" DIRECTORY)
endif()

include(FindPackageHandleStandardArgs)
# Naming the searched paths matters here: five different variables feed the hints, so
# "not found" is otherwise indistinguishable from "found, but you meant another one".
if(OSHMPI_HINT_PATHS)
  string(REPLACE ";" "\n  " _oshmpi_searched "${OSHMPI_HINT_PATHS}")
  set(_oshmpi_reason "OSHMPI not found under:\n  ${_oshmpi_searched}\nBuild it with cluster/leonardo/bootstrap.sh oneccl-oshmpi, or set OSHMPI_HOME to an existing install.")
else()
  set(_oshmpi_reason "No OSHMPI location given. Set OSHMPI_HOME, OSHMPI_ROOT, or OSHMPI_DIR.")
endif()

find_package_handle_standard_args(OSHMPI
  REQUIRED_VARS OSHMPI_LIBRARY OSHMPI_INCLUDE_DIR OSHMPI_SHMEMX_INCLUDE_DIR
  REASON_FAILURE_MESSAGE "${_oshmpi_reason}")

mark_as_advanced(OSHMPI_INCLUDE_DIR OSHMPI_SHMEMX_INCLUDE_DIR OSHMPI_LIBRARY OSHMPI_LIBRARY_DIR OSHMPI_OSHCC)

set(OSHMPI_INCLUDE_DIRS ${OSHMPI_INCLUDE_DIR})
if(OSHMPI_SHMEMX_INCLUDE_DIR AND NOT OSHMPI_SHMEMX_INCLUDE_DIR STREQUAL OSHMPI_INCLUDE_DIR)
  list(APPEND OSHMPI_INCLUDE_DIRS ${OSHMPI_SHMEMX_INCLUDE_DIR})
endif()
set(OSHMPI_LIBRARIES ${OSHMPI_LIBRARY})
set(OSHMPI_LIBRARY_DIRS ${OSHMPI_LIBRARY_DIR})
