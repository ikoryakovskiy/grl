# Setup build environment
set(TARGET addon_gl)

find_package(OpenGL)
if (OPENGL_FOUND)
  message("-- Building OpenGL addon")

  # Build library
  add_library(${TARGET} SHARED
              ${SRC}/sample.cpp
              ${SRC}/random_sample.cpp
              ${SRC}/field.cpp
              ${SRC}/value.cpp
              ${SRC}/policy.cpp
              ${SRC}/state.cpp
             )

  # Add dependencies
  target_link_libraries(${TARGET} ${OPENGL_LIBRARIES})
  grl_link_libraries(${TARGET} base externals/ics)
  install(TARGETS ${TARGET} DESTINATION ${GRL_LIB_DESTINATION})
  install(DIRECTORY ${SRC}/../include/grl DESTINATION ${GRL_INCLUDE_DESTINATION} FILES_MATCHING PATTERN "*.h")
endif()
