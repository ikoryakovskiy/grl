# Setup build environment
set(TARGET pybind11)

message("-- Building included Pybind11 library")

execute_process(
  COMMAND rm -rf externals/pybind11
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xf ${SRC}/../share/pybind11-a303c6fc479662fd53eaa8990dbc65b7de9b7deb.zip
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

execute_process(
  COMMAND mkdir -p externals
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

execute_process(
  COMMAND mv pybind11-a303c6fc479662fd53eaa8990dbc65b7de9b7deb externals/pybind11
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

execute_process(
  COMMAND mkdir externals/pybind11/build
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_subdirectory(${CMAKE_BINARY_DIR}/externals/pybind11 ${CMAKE_BINARY_DIR}/externals/pybind11/build)

