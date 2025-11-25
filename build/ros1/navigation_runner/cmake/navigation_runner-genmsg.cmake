# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "navigation_runner: 0 messages, 3 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(navigation_runner_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv" NAME_WE)
add_custom_target(_navigation_runner_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "navigation_runner" "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv" "geometry_msgs/Vector3:geometry_msgs/Point"
)

get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv" NAME_WE)
add_custom_target(_navigation_runner_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "navigation_runner" "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv" ""
)

get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv" NAME_WE)
add_custom_target(_navigation_runner_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "navigation_runner" "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv" "geometry_msgs/Vector3:geometry_msgs/Point"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/navigation_runner
)
_generate_srv_cpp(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/navigation_runner
)
_generate_srv_cpp(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/navigation_runner
)

### Generating Module File
_generate_module_cpp(navigation_runner
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/navigation_runner
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(navigation_runner_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(navigation_runner_generate_messages navigation_runner_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_cpp _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_cpp _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_cpp _navigation_runner_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(navigation_runner_gencpp)
add_dependencies(navigation_runner_gencpp navigation_runner_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS navigation_runner_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/navigation_runner
)
_generate_srv_eus(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/navigation_runner
)
_generate_srv_eus(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/navigation_runner
)

### Generating Module File
_generate_module_eus(navigation_runner
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/navigation_runner
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(navigation_runner_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(navigation_runner_generate_messages navigation_runner_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_eus _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_eus _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_eus _navigation_runner_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(navigation_runner_geneus)
add_dependencies(navigation_runner_geneus navigation_runner_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS navigation_runner_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/navigation_runner
)
_generate_srv_lisp(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/navigation_runner
)
_generate_srv_lisp(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/navigation_runner
)

### Generating Module File
_generate_module_lisp(navigation_runner
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/navigation_runner
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(navigation_runner_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(navigation_runner_generate_messages navigation_runner_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_lisp _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_lisp _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_lisp _navigation_runner_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(navigation_runner_genlisp)
add_dependencies(navigation_runner_genlisp navigation_runner_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS navigation_runner_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/navigation_runner
)
_generate_srv_nodejs(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/navigation_runner
)
_generate_srv_nodejs(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/navigation_runner
)

### Generating Module File
_generate_module_nodejs(navigation_runner
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/navigation_runner
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(navigation_runner_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(navigation_runner_generate_messages navigation_runner_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_nodejs _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_nodejs _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_nodejs _navigation_runner_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(navigation_runner_gennodejs)
add_dependencies(navigation_runner_gennodejs navigation_runner_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS navigation_runner_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/navigation_runner
)
_generate_srv_py(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/navigation_runner
)
_generate_srv_py(navigation_runner
  "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/navigation_runner
)

### Generating Module File
_generate_module_py(navigation_runner
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/navigation_runner
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(navigation_runner_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(navigation_runner_generate_messages navigation_runner_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeAction.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_py _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetPolicyInference.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_py _navigation_runner_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/navigation_runner/srv/GetSafeActionMap.srv" NAME_WE)
add_dependencies(navigation_runner_generate_messages_py _navigation_runner_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(navigation_runner_genpy)
add_dependencies(navigation_runner_genpy navigation_runner_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS navigation_runner_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/navigation_runner)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/navigation_runner
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(navigation_runner_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(navigation_runner_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/navigation_runner)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/navigation_runner
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(navigation_runner_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(navigation_runner_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/navigation_runner)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/navigation_runner
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(navigation_runner_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(navigation_runner_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/navigation_runner)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/navigation_runner
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(navigation_runner_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(navigation_runner_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/navigation_runner)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/navigation_runner\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/navigation_runner
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(navigation_runner_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(navigation_runner_generate_messages_py geometry_msgs_generate_messages_py)
endif()
