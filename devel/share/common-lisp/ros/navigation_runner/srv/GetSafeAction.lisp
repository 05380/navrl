; Auto-generated. Do not edit!


(cl:in-package navigation_runner-srv)


;//! \htmlinclude GetSafeAction-request.msg.html

(cl:defclass <GetSafeAction-request> (roslisp-msg-protocol:ros-message)
  ((agent_position
    :reader agent_position
    :initarg :agent_position
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point))
   (agent_velocity
    :reader agent_velocity
    :initarg :agent_velocity
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (agent_size
    :reader agent_size
    :initarg :agent_size
    :type cl:float
    :initform 0.0)
   (obs_position
    :reader obs_position
    :initarg :obs_position
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (obs_velocity
    :reader obs_velocity
    :initarg :obs_velocity
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (obs_size
    :reader obs_size
    :initarg :obs_size
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (laser_points
    :reader laser_points
    :initarg :laser_points
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (laser_range
    :reader laser_range
    :initarg :laser_range
    :type cl:float
    :initform 0.0)
   (laser_res
    :reader laser_res
    :initarg :laser_res
    :type cl:float
    :initform 0.0)
   (max_velocity
    :reader max_velocity
    :initarg :max_velocity
    :type cl:float
    :initform 0.0)
   (rl_velocity
    :reader rl_velocity
    :initarg :rl_velocity
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass GetSafeAction-request (<GetSafeAction-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSafeAction-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSafeAction-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name navigation_runner-srv:<GetSafeAction-request> is deprecated: use navigation_runner-srv:GetSafeAction-request instead.")))

(cl:ensure-generic-function 'agent_position-val :lambda-list '(m))
(cl:defmethod agent_position-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:agent_position-val is deprecated.  Use navigation_runner-srv:agent_position instead.")
  (agent_position m))

(cl:ensure-generic-function 'agent_velocity-val :lambda-list '(m))
(cl:defmethod agent_velocity-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:agent_velocity-val is deprecated.  Use navigation_runner-srv:agent_velocity instead.")
  (agent_velocity m))

(cl:ensure-generic-function 'agent_size-val :lambda-list '(m))
(cl:defmethod agent_size-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:agent_size-val is deprecated.  Use navigation_runner-srv:agent_size instead.")
  (agent_size m))

(cl:ensure-generic-function 'obs_position-val :lambda-list '(m))
(cl:defmethod obs_position-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:obs_position-val is deprecated.  Use navigation_runner-srv:obs_position instead.")
  (obs_position m))

(cl:ensure-generic-function 'obs_velocity-val :lambda-list '(m))
(cl:defmethod obs_velocity-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:obs_velocity-val is deprecated.  Use navigation_runner-srv:obs_velocity instead.")
  (obs_velocity m))

(cl:ensure-generic-function 'obs_size-val :lambda-list '(m))
(cl:defmethod obs_size-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:obs_size-val is deprecated.  Use navigation_runner-srv:obs_size instead.")
  (obs_size m))

(cl:ensure-generic-function 'laser_points-val :lambda-list '(m))
(cl:defmethod laser_points-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:laser_points-val is deprecated.  Use navigation_runner-srv:laser_points instead.")
  (laser_points m))

(cl:ensure-generic-function 'laser_range-val :lambda-list '(m))
(cl:defmethod laser_range-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:laser_range-val is deprecated.  Use navigation_runner-srv:laser_range instead.")
  (laser_range m))

(cl:ensure-generic-function 'laser_res-val :lambda-list '(m))
(cl:defmethod laser_res-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:laser_res-val is deprecated.  Use navigation_runner-srv:laser_res instead.")
  (laser_res m))

(cl:ensure-generic-function 'max_velocity-val :lambda-list '(m))
(cl:defmethod max_velocity-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:max_velocity-val is deprecated.  Use navigation_runner-srv:max_velocity instead.")
  (max_velocity m))

(cl:ensure-generic-function 'rl_velocity-val :lambda-list '(m))
(cl:defmethod rl_velocity-val ((m <GetSafeAction-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:rl_velocity-val is deprecated.  Use navigation_runner-srv:rl_velocity instead.")
  (rl_velocity m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSafeAction-request>) ostream)
  "Serializes a message object of type '<GetSafeAction-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'agent_position) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'agent_velocity) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'agent_size))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'obs_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'obs_position))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'obs_velocity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'obs_velocity))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'obs_size))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'obs_size))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'laser_points))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'laser_points))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'laser_range))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'laser_res))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'max_velocity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rl_velocity) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSafeAction-request>) istream)
  "Deserializes a message object of type '<GetSafeAction-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'agent_position) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'agent_velocity) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'agent_size) (roslisp-utils:decode-double-float-bits bits)))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'obs_position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'obs_position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'obs_velocity) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'obs_velocity)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'obs_size) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'obs_size)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'laser_points) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'laser_points)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits))))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'laser_range) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'laser_res) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'max_velocity) (roslisp-utils:decode-double-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rl_velocity) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSafeAction-request>)))
  "Returns string type for a service object of type '<GetSafeAction-request>"
  "navigation_runner/GetSafeActionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSafeAction-request)))
  "Returns string type for a service object of type 'GetSafeAction-request"
  "navigation_runner/GetSafeActionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSafeAction-request>)))
  "Returns md5sum for a message object of type '<GetSafeAction-request>"
  "54fa8c288cd542a68cd6fb86a87e59f8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSafeAction-request)))
  "Returns md5sum for a message object of type 'GetSafeAction-request"
  "54fa8c288cd542a68cd6fb86a87e59f8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSafeAction-request>)))
  "Returns full string definition for message of type '<GetSafeAction-request>"
  (cl:format cl:nil "geometry_msgs/Point agent_position~%geometry_msgs/Vector3 agent_velocity~%float64 agent_size~%~%# dynamic obstacles~%geometry_msgs/Vector3[] obs_position~%geometry_msgs/Vector3[] obs_velocity~%geometry_msgs/Vector3[] obs_size~%~%# static obstacles~%float64[] laser_points~%float64 laser_range~%float64 laser_res # use the larger one among horizonal and vertical~%~%float64 max_velocity~%geometry_msgs/Vector3 rl_velocity~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSafeAction-request)))
  "Returns full string definition for message of type 'GetSafeAction-request"
  (cl:format cl:nil "geometry_msgs/Point agent_position~%geometry_msgs/Vector3 agent_velocity~%float64 agent_size~%~%# dynamic obstacles~%geometry_msgs/Vector3[] obs_position~%geometry_msgs/Vector3[] obs_velocity~%geometry_msgs/Vector3[] obs_size~%~%# static obstacles~%float64[] laser_points~%float64 laser_range~%float64 laser_res # use the larger one among horizonal and vertical~%~%float64 max_velocity~%geometry_msgs/Vector3 rl_velocity~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSafeAction-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'agent_position))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'agent_velocity))
     8
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'obs_position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'obs_velocity) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'obs_size) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'laser_points) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     8
     8
     8
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rl_velocity))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSafeAction-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSafeAction-request
    (cl:cons ':agent_position (agent_position msg))
    (cl:cons ':agent_velocity (agent_velocity msg))
    (cl:cons ':agent_size (agent_size msg))
    (cl:cons ':obs_position (obs_position msg))
    (cl:cons ':obs_velocity (obs_velocity msg))
    (cl:cons ':obs_size (obs_size msg))
    (cl:cons ':laser_points (laser_points msg))
    (cl:cons ':laser_range (laser_range msg))
    (cl:cons ':laser_res (laser_res msg))
    (cl:cons ':max_velocity (max_velocity msg))
    (cl:cons ':rl_velocity (rl_velocity msg))
))
;//! \htmlinclude GetSafeAction-response.msg.html

(cl:defclass <GetSafeAction-response> (roslisp-msg-protocol:ros-message)
  ((safe_action
    :reader safe_action
    :initarg :safe_action
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass GetSafeAction-response (<GetSafeAction-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSafeAction-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSafeAction-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name navigation_runner-srv:<GetSafeAction-response> is deprecated: use navigation_runner-srv:GetSafeAction-response instead.")))

(cl:ensure-generic-function 'safe_action-val :lambda-list '(m))
(cl:defmethod safe_action-val ((m <GetSafeAction-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:safe_action-val is deprecated.  Use navigation_runner-srv:safe_action instead.")
  (safe_action m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSafeAction-response>) ostream)
  "Serializes a message object of type '<GetSafeAction-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'safe_action) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSafeAction-response>) istream)
  "Deserializes a message object of type '<GetSafeAction-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'safe_action) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSafeAction-response>)))
  "Returns string type for a service object of type '<GetSafeAction-response>"
  "navigation_runner/GetSafeActionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSafeAction-response)))
  "Returns string type for a service object of type 'GetSafeAction-response"
  "navigation_runner/GetSafeActionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSafeAction-response>)))
  "Returns md5sum for a message object of type '<GetSafeAction-response>"
  "54fa8c288cd542a68cd6fb86a87e59f8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSafeAction-response)))
  "Returns md5sum for a message object of type 'GetSafeAction-response"
  "54fa8c288cd542a68cd6fb86a87e59f8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSafeAction-response>)))
  "Returns full string definition for message of type '<GetSafeAction-response>"
  (cl:format cl:nil "geometry_msgs/Vector3 safe_action~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSafeAction-response)))
  "Returns full string definition for message of type 'GetSafeAction-response"
  (cl:format cl:nil "geometry_msgs/Vector3 safe_action~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSafeAction-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'safe_action))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSafeAction-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSafeAction-response
    (cl:cons ':safe_action (safe_action msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetSafeAction)))
  'GetSafeAction-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetSafeAction)))
  'GetSafeAction-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSafeAction)))
  "Returns string type for a service object of type '<GetSafeAction>"
  "navigation_runner/GetSafeAction")