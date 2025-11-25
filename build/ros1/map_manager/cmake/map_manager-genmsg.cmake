# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "map_manager: 0 messages, 3 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(map_manager_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv" NAME_WE)
add_custom_target(_map_manager_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "map_manager" "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv" ""
)

get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv" NAME_WE)
add_custom_target(_map_manager_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "map_manager" "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv" "geometry_msgs/Point"
)

get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv" NAME_WE)
add_custom_target(_map_manager_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "map_manager" "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv" "geometry_msgs/Vector3"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/map_manager
)
_generate_srv_cpp(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/map_manager
)
_generate_srv_cpp(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/map_manager
)

### Generating Module File
_generate_module_cpp(map_manager
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/map_manager
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(map_manager_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(map_manager_generate_messages map_manager_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_cpp _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_cpp _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_cpp _map_manager_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(map_manager_gencpp)
add_dependencies(map_manager_gencpp map_manager_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS map_manager_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/map_manager
)
_generate_srv_eus(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/map_manager
)
_generate_srv_eus(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/map_manager
)

### Generating Module File
_generate_module_eus(map_manager
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/map_manager
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(map_manager_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(map_manager_generate_messages map_manager_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_eus _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_eus _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_eus _map_manager_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(map_manager_geneus)
add_dependencies(map_manager_geneus map_manager_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS map_manager_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/map_manager
)
_generate_srv_lisp(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/map_manager
)
_generate_srv_lisp(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/map_manager
)

### Generating Module File
_generate_module_lisp(map_manager
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/map_manager
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(map_manager_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(map_manager_generate_messages map_manager_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_lisp _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_lisp _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_lisp _map_manager_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(map_manager_genlisp)
add_dependencies(map_manager_genlisp map_manager_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS map_manager_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/map_manager
)
_generate_srv_nodejs(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/map_manager
)
_generate_srv_nodejs(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/map_manager
)

### Generating Module File
_generate_module_nodejs(map_manager
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/map_manager
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(map_manager_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(map_manager_generate_messages map_manager_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_nodejs _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_nodejs _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_nodejs _map_manager_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(map_manager_gennodejs)
add_dependencies(map_manager_gennodejs map_manager_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS map_manager_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/map_manager
)
_generate_srv_py(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/map_manager
)
_generate_srv_py(map_manager
  "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/map_manager
)

### Generating Module File
_generate_module_py(map_manager
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/map_manager
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(map_manager_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(map_manager_generate_messages map_manager_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/CheckPosCollision.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_py _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/RayCast.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_py _map_manager_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tiko/navrl_ws/src/ros1/map_manager/srv/GetStaticObstacles.srv" NAME_WE)
add_dependencies(map_manager_generate_messages_py _map_manager_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(map_manager_genpy)
add_dependencies(map_manager_genpy map_manager_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS map_manager_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/map_manager)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/map_manager
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(map_manager_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(map_manager_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/map_manager)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/map_manager
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(map_manager_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(map_manager_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/map_manager)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/map_manager
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(map_manager_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(map_manager_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/map_manager)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/map_manager
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(map_manager_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(map_manager_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/map_manager)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/map_manager\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/map_manager
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(map_manager_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(map_manager_generate_messages_py geometry_msgs_generate_messages_py)
endif()
