# Select Python version which contains openai gym and baseline packages. If no such version found, select any available
foreach(python_version IN ITEMS 3 2)
  message("-- Looking for Python version ${python_version}")
  FIND_PACKAGE(Python ${python_version} EXACT)
  find_python_module(gym)
  find_python_module(baselines)
  if (PY_GYM AND PY_BASELINES)
    set(OPENAI_FOUND 1)
    break()
  else()
    unset(PYTHON_FOUND CACHE)
    unset(PYTHON_EXEC CACHE)
    unset(PYTHON_LIBRARIES CACHE)
    unset(PYTHON_INCLUDE_DIRS CACHE)
    unset(PYTHON_SITE_MODULES CACHE)
    unset(PYTHON_ARCH CACHE)
    unset(PYTHON_VERSION CACHE)
    unset(PYTHON_VERSION_MAJOR CACHE)
    unset(PYTHON_VERSION_MAJOR CACHE)
    unset(PYTHON_VERSION_MICRO CACHE)
  endif()
endforeach(python_version)

if (NOT OPENAI_FOUND)
  message("   Could not find Openai, use any python version")
  FIND_PACKAGE(Python)
endif()

if (PYTHON_FOUND)
  # Version of Eigen library
  file(READ "${EIGEN3_INCLUDE_DIRS}/Eigen/src/Core/util/Macros.h" _eigen_version_header)
  string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen_world_version_match "${_eigen_version_header}")
  set(EIGEN_WORLD_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen_major_version_match "${_eigen_version_header}")
  set(EIGEN_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen_minor_version_match "${_eigen_version_header}")
  set(EIGEN_MINOR_VERSION "${CMAKE_MATCH_1}")
  set(EIGEN_VERSION_NUMBER ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})

  if (EIGEN_VERSION_NUMBER LESS 3.2.6)
    message("   Found Eigen ${EIGEN_VERSION_NUMBER}")
    message("   Eigen support in pybind11 requires Eigen >= 3.2.7")
  else()
    message("-- Building Python addon (using Python ${PYTHON_VERSION} from ${PYTHON_EXEC})")
    pybind11_add_module(py_env NO_EXTRAS ${SRC}/py_env.cpp)
    target_link_libraries(py_env PRIVATE grl pthread dl yaml-cpp)
    # TODO Find a proper way to install .so library as python package (+resolve possible permission denied issue)
    set(py_site_modules ${PYTHON_PREFIX2}/${PYTHON_SITE_MODULES})
    string(FIND "${py_site_modules}" "/home/" pos)
    if (${pos} EQUAL 0)
      add_custom_command(
              TARGET py_env
              POST_BUILD
              COMMAND ln -s -f ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/py_env* ${PYTHON_PREFIX2}/${PYTHON_SITE_MODULES})
      message("   the link to py_env will created from ${PYTHON_PREFIX2}/${PYTHON_SITE_MODULES} location")
    endif()
  endif()
else()
  message("   Python not found")
endif()
