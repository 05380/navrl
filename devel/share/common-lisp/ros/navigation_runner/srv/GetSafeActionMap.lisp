; Auto-generated. Do not edit!


(cl:in-package navigation_runner-srv)


;//! \htmlinclude GetSafeActionMap-request.msg.html

(cl:defclass <GetSafeActionMap-request> (roslisp-msg-protocol:ros-message)
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
   (dynamic_obs_position
    :reader dynamic_obs_position
    :initarg :dynamic_obs_position
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (dynamic_obs_velocity
    :reader dynamic_obs_velocity
    :initarg :dynamic_obs_velocity
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (dynamic_obs_size
    :reader dynamic_obs_size
    :initarg :dynamic_obs_size
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (static_obs_position
    :reader static_obs_position
    :initarg :static_obs_position
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (static_obs_size
    :reader static_obs_size
    :initarg :static_obs_size
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (static_obs_angle
    :reader static_obs_angle
    :initarg :static_obs_angle
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
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

(cl:defclass GetSafeActionMap-request (<GetSafeActionMap-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSafeActionMap-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSafeActionMap-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name navigation_runner-srv:<GetSafeActionMap-request> is deprecated: use navigation_runner-srv:GetSafeActionMap-request instead.")))

(cl:ensure-generic-function 'agent_position-val :lambda-list '(m))
(cl:defmethod agent_position-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:agent_position-val is deprecated.  Use navigation_runner-srv:agent_position instead.")
  (agent_position m))

(cl:ensure-generic-function 'agent_velocity-val :lambda-list '(m))
(cl:defmethod agent_velocity-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:agent_velocity-val is deprecated.  Use navigation_runner-srv:agent_velocity instead.")
  (agent_velocity m))

(cl:ensure-generic-function 'agent_size-val :lambda-list '(m))
(cl:defmethod agent_size-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:agent_size-val is deprecated.  Use navigation_runner-srv:agent_size instead.")
  (agent_size m))

(cl:ensure-generic-function 'dynamic_obs_position-val :lambda-list '(m))
(cl:defmethod dynamic_obs_position-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:dynamic_obs_position-val is deprecated.  Use navigation_runner-srv:dynamic_obs_position instead.")
  (dynamic_obs_position m))

(cl:ensure-generic-function 'dynamic_obs_velocity-val :lambda-list '(m))
(cl:defmethod dynamic_obs_velocity-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:dynamic_obs_velocity-val is deprecated.  Use navigation_runner-srv:dynamic_obs_velocity instead.")
  (dynamic_obs_velocity m))

(cl:ensure-generic-function 'dynamic_obs_size-val :lambda-list '(m))
(cl:defmethod dynamic_obs_size-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:dynamic_obs_size-val is deprecated.  Use navigation_runner-srv:dynamic_obs_size instead.")
  (dynamic_obs_size m))

(cl:ensure-generic-function 'static_obs_position-val :lambda-list '(m))
(cl:defmethod static_obs_position-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:static_obs_position-val is deprecated.  Use navigation_runner-srv:static_obs_position instead.")
  (static_obs_position m))

(cl:ensure-generic-function 'static_obs_size-val :lambda-list '(m))
(cl:defmethod static_obs_size-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:static_obs_size-val is deprecated.  Use navigation_runner-srv:static_obs_size instead.")
  (static_obs_size m))

(cl:ensure-generic-function 'static_obs_angle-val :lambda-list '(m))
(cl:defmethod static_obs_angle-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:static_obs_angle-val is deprecated.  Use navigation_runner-srv:static_obs_angle instead.")
  (static_obs_angle m))

(cl:ensure-generic-function 'max_velocity-val :lambda-list '(m))
(cl:defmethod max_velocity-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:max_velocity-val is deprecated.  Use navigation_runner-srv:max_velocity instead.")
  (max_velocity m))

(cl:ensure-generic-function 'rl_velocity-val :lambda-list '(m))
(cl:defmethod rl_velocity-val ((m <GetSafeActionMap-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:rl_velocity-val is deprecated.  Use navigation_runner-srv:rl_velocity instead.")
  (rl_velocity m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSafeActionMap-request>) ostream)
  "Serializes a message object of type '<GetSafeActionMap-request>"
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
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'dynamic_obs_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'dynamic_obs_position))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'dynamic_obs_velocity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'dynamic_obs_velocity))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'dynamic_obs_size))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'dynamic_obs_size))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'static_obs_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'static_obs_position))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'static_obs_size))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'static_obs_size))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'static_obs_angle))))
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
   (cl:slot-value msg 'static_obs_angle))
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSafeActionMap-request>) istream)
  "Deserializes a message object of type '<GetSafeActionMap-request>"
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
  (cl:setf (cl:slot-value msg 'dynamic_obs_position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'dynamic_obs_position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'dynamic_obs_velocity) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'dynamic_obs_velocity)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'dynamic_obs_size) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'dynamic_obs_size)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'static_obs_position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'static_obs_position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'static_obs_size) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'static_obs_size)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'static_obs_angle) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'static_obs_angle)))
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
    (cl:setf (cl:slot-value msg 'max_velocity) (roslisp-utils:decode-double-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rl_velocity) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSafeActionMap-request>)))
  "Returns string type for a service object of type '<GetSafeActionMap-request>"
  "navigation_runner/GetSafeActionMapRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSafeActionMap-request)))
  "Returns string type for a service object of type 'GetSafeActionMap-request"
  "navigation_runner/GetSafeActionMapRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSafeActionMap-request>)))
  "Returns md5sum for a message object of type '<GetSafeActionMap-request>"
  "4635b39c982fd59eaaa757b70b204f4c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSafeActionMap-request)))
  "Returns md5sum for a message object of type 'GetSafeActionMap-request"
  "4635b39c982fd59eaaa757b70b204f4c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSafeActionMap-request>)))
  "Returns full string definition for message of type '<GetSafeActionMap-request>"
  (cl:format cl:nil "geometry_msgs/Point agent_position~%geometry_msgs/Vector3 agent_velocity~%float64 agent_size~%~%# dynamic obstacles~%geometry_msgs/Vector3[] dynamic_obs_position~%geometry_msgs/Vector3[] dynamic_obs_velocity~%geometry_msgs/Vector3[] dynamic_obs_size~%~%# static obstacles~%geometry_msgs/Vector3[] static_obs_position~%geometry_msgs/Vector3[] static_obs_size~%float64[] static_obs_angle~%~%float64 max_velocity~%geometry_msgs/Vector3 rl_velocity~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSafeActionMap-request)))
  "Returns full string definition for message of type 'GetSafeActionMap-request"
  (cl:format cl:nil "geometry_msgs/Point agent_position~%geometry_msgs/Vector3 agent_velocity~%float64 agent_size~%~%# dynamic obstacles~%geometry_msgs/Vector3[] dynamic_obs_position~%geometry_msgs/Vector3[] dynamic_obs_velocity~%geometry_msgs/Vector3[] dynamic_obs_size~%~%# static obstacles~%geometry_msgs/Vector3[] static_obs_position~%geometry_msgs/Vector3[] static_obs_size~%float64[] static_obs_angle~%~%float64 max_velocity~%geometry_msgs/Vector3 rl_velocity~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSafeActionMap-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'agent_position))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'agent_velocity))
     8
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'dynamic_obs_position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'dynamic_obs_velocity) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'dynamic_obs_size) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'static_obs_position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'static_obs_size) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'static_obs_angle) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     8
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rl_velocity))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSafeActionMap-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSafeActionMap-request
    (cl:cons ':agent_position (agent_position msg))
    (cl:cons ':agent_velocity (agent_velocity msg))
    (cl:cons ':agent_size (agent_size msg))
    (cl:cons ':dynamic_obs_position (dynamic_obs_position msg))
    (cl:cons ':dynamic_obs_velocity (dynamic_obs_velocity msg))
    (cl:cons ':dynamic_obs_size (dynamic_obs_size msg))
    (cl:cons ':static_obs_position (static_obs_position msg))
    (cl:cons ':static_obs_size (static_obs_size msg))
    (cl:cons ':static_obs_angle (static_obs_angle msg))
    (cl:cons ':max_velocity (max_velocity msg))
    (cl:cons ':rl_velocity (rl_velocity msg))
))
;//! \htmlinclude GetSafeActionMap-response.msg.html

(cl:defclass <GetSafeActionMap-response> (roslisp-msg-protocol:ros-message)
  ((safe_action
    :reader safe_action
    :initarg :safe_action
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass GetSafeActionMap-response (<GetSafeActionMap-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSafeActionMap-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSafeActionMap-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name navigation_runner-srv:<GetSafeActionMap-response> is deprecated: use navigation_runner-srv:GetSafeActionMap-response instead.")))

(cl:ensure-generic-function 'safe_action-val :lambda-list '(m))
(cl:defmethod safe_action-val ((m <GetSafeActionMap-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader navigation_runner-srv:safe_action-val is deprecated.  Use navigation_runner-srv:safe_action instead.")
  (safe_action m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSafeActionMap-response>) ostream)
  "Serializes a message object of type '<GetSafeActionMap-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'safe_action) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSafeActionMap-response>) istream)
  "Deserializes a message object of type '<GetSafeActionMap-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'safe_action) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSafeActionMap-response>)))
  "Returns string type for a service object of type '<GetSafeActionMap-response>"
  "navigation_runner/GetSafeActionMapResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSafeActionMap-response)))
  "Returns string type for a service object of type 'GetSafeActionMap-response"
  "navigation_runner/GetSafeActionMapResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSafeActionMap-response>)))
  "Returns md5sum for a message object of type '<GetSafeActionMap-response>"
  "4635b39c982fd59eaaa757b70b204f4c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSafeActionMap-response)))
  "Returns md5sum for a message object of type 'GetSafeActionMap-response"
  "4635b39c982fd59eaaa757b70b204f4c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSafeActionMap-response>)))
  "Returns full string definition for message of type '<GetSafeActionMap-response>"
  (cl:format cl:nil "geometry_msgs/Vector3 safe_action~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSafeActionMap-response)))
  "Returns full string definition for message of type 'GetSafeActionMap-response"
  (cl:format cl:nil "geometry_msgs/Vector3 safe_action~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSafeActionMap-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'safe_action))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSafeActionMap-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSafeActionMap-response
    (cl:cons ':safe_action (safe_action msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetSafeActionMap)))
  'GetSafeActionMap-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetSafeActionMap)))
  'GetSafeActionMap-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSafeActionMap)))
  "Returns string type for a service object of type '<GetSafeActionMap>"
  "navigation_runner/GetSafeActionMap")