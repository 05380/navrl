; Auto-generated. Do not edit!


(cl:in-package map_manager-srv)


;//! \htmlinclude GetStaticObstacles-request.msg.html

(cl:defclass <GetStaticObstacles-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GetStaticObstacles-request (<GetStaticObstacles-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetStaticObstacles-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetStaticObstacles-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name map_manager-srv:<GetStaticObstacles-request> is deprecated: use map_manager-srv:GetStaticObstacles-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetStaticObstacles-request>) ostream)
  "Serializes a message object of type '<GetStaticObstacles-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetStaticObstacles-request>) istream)
  "Deserializes a message object of type '<GetStaticObstacles-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetStaticObstacles-request>)))
  "Returns string type for a service object of type '<GetStaticObstacles-request>"
  "map_manager/GetStaticObstaclesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetStaticObstacles-request)))
  "Returns string type for a service object of type 'GetStaticObstacles-request"
  "map_manager/GetStaticObstaclesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetStaticObstacles-request>)))
  "Returns md5sum for a message object of type '<GetStaticObstacles-request>"
  "10afeb58fe82a6ef3c941519f4670020")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetStaticObstacles-request)))
  "Returns md5sum for a message object of type 'GetStaticObstacles-request"
  "10afeb58fe82a6ef3c941519f4670020")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetStaticObstacles-request>)))
  "Returns full string definition for message of type '<GetStaticObstacles-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetStaticObstacles-request)))
  "Returns full string definition for message of type 'GetStaticObstacles-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetStaticObstacles-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetStaticObstacles-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetStaticObstacles-request
))
;//! \htmlinclude GetStaticObstacles-response.msg.html

(cl:defclass <GetStaticObstacles-response> (roslisp-msg-protocol:ros-message)
  ((position
    :reader position
    :initarg :position
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (size
    :reader size
    :initarg :size
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (angle
    :reader angle
    :initarg :angle
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass GetStaticObstacles-response (<GetStaticObstacles-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetStaticObstacles-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetStaticObstacles-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name map_manager-srv:<GetStaticObstacles-response> is deprecated: use map_manager-srv:GetStaticObstacles-response instead.")))

(cl:ensure-generic-function 'position-val :lambda-list '(m))
(cl:defmethod position-val ((m <GetStaticObstacles-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:position-val is deprecated.  Use map_manager-srv:position instead.")
  (position m))

(cl:ensure-generic-function 'size-val :lambda-list '(m))
(cl:defmethod size-val ((m <GetStaticObstacles-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:size-val is deprecated.  Use map_manager-srv:size instead.")
  (size m))

(cl:ensure-generic-function 'angle-val :lambda-list '(m))
(cl:defmethod angle-val ((m <GetStaticObstacles-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:angle-val is deprecated.  Use map_manager-srv:angle instead.")
  (angle m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetStaticObstacles-response>) ostream)
  "Serializes a message object of type '<GetStaticObstacles-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'position))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'size))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'size))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'angle))))
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
   (cl:slot-value msg 'angle))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetStaticObstacles-response>) istream)
  "Deserializes a message object of type '<GetStaticObstacles-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'size) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'size)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'angle) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'angle)))
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetStaticObstacles-response>)))
  "Returns string type for a service object of type '<GetStaticObstacles-response>"
  "map_manager/GetStaticObstaclesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetStaticObstacles-response)))
  "Returns string type for a service object of type 'GetStaticObstacles-response"
  "map_manager/GetStaticObstaclesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetStaticObstacles-response>)))
  "Returns md5sum for a message object of type '<GetStaticObstacles-response>"
  "10afeb58fe82a6ef3c941519f4670020")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetStaticObstacles-response)))
  "Returns md5sum for a message object of type 'GetStaticObstacles-response"
  "10afeb58fe82a6ef3c941519f4670020")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetStaticObstacles-response>)))
  "Returns full string definition for message of type '<GetStaticObstacles-response>"
  (cl:format cl:nil "geometry_msgs/Vector3[] position~%geometry_msgs/Vector3[] size~%float64[] angle~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetStaticObstacles-response)))
  "Returns full string definition for message of type 'GetStaticObstacles-response"
  (cl:format cl:nil "geometry_msgs/Vector3[] position~%geometry_msgs/Vector3[] size~%float64[] angle~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetStaticObstacles-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'size) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'angle) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetStaticObstacles-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetStaticObstacles-response
    (cl:cons ':position (position msg))
    (cl:cons ':size (size msg))
    (cl:cons ':angle (angle msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetStaticObstacles)))
  'GetStaticObstacles-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetStaticObstacles)))
  'GetStaticObstacles-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetStaticObstacles)))
  "Returns string type for a service object of type '<GetStaticObstacles>"
  "map_manager/GetStaticObstacles")