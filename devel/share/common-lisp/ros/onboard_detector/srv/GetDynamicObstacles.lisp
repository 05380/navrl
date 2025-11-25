; Auto-generated. Do not edit!


(cl:in-package onboard_detector-srv)


;//! \htmlinclude GetDynamicObstacles-request.msg.html

(cl:defclass <GetDynamicObstacles-request> (roslisp-msg-protocol:ros-message)
  ((current_position
    :reader current_position
    :initarg :current_position
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point))
   (range
    :reader range
    :initarg :range
    :type cl:float
    :initform 0.0))
)

(cl:defclass GetDynamicObstacles-request (<GetDynamicObstacles-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetDynamicObstacles-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetDynamicObstacles-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name onboard_detector-srv:<GetDynamicObstacles-request> is deprecated: use onboard_detector-srv:GetDynamicObstacles-request instead.")))

(cl:ensure-generic-function 'current_position-val :lambda-list '(m))
(cl:defmethod current_position-val ((m <GetDynamicObstacles-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader onboard_detector-srv:current_position-val is deprecated.  Use onboard_detector-srv:current_position instead.")
  (current_position m))

(cl:ensure-generic-function 'range-val :lambda-list '(m))
(cl:defmethod range-val ((m <GetDynamicObstacles-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader onboard_detector-srv:range-val is deprecated.  Use onboard_detector-srv:range instead.")
  (range m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetDynamicObstacles-request>) ostream)
  "Serializes a message object of type '<GetDynamicObstacles-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'current_position) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'range))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetDynamicObstacles-request>) istream)
  "Deserializes a message object of type '<GetDynamicObstacles-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'current_position) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'range) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetDynamicObstacles-request>)))
  "Returns string type for a service object of type '<GetDynamicObstacles-request>"
  "onboard_detector/GetDynamicObstaclesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetDynamicObstacles-request)))
  "Returns string type for a service object of type 'GetDynamicObstacles-request"
  "onboard_detector/GetDynamicObstaclesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetDynamicObstacles-request>)))
  "Returns md5sum for a message object of type '<GetDynamicObstacles-request>"
  "a508c5cda827e9832476a1de6a1e2f79")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetDynamicObstacles-request)))
  "Returns md5sum for a message object of type 'GetDynamicObstacles-request"
  "a508c5cda827e9832476a1de6a1e2f79")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetDynamicObstacles-request>)))
  "Returns full string definition for message of type '<GetDynamicObstacles-request>"
  (cl:format cl:nil "geometry_msgs/Point current_position~%float64 range~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetDynamicObstacles-request)))
  "Returns full string definition for message of type 'GetDynamicObstacles-request"
  (cl:format cl:nil "geometry_msgs/Point current_position~%float64 range~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetDynamicObstacles-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'current_position))
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetDynamicObstacles-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetDynamicObstacles-request
    (cl:cons ':current_position (current_position msg))
    (cl:cons ':range (range msg))
))
;//! \htmlinclude GetDynamicObstacles-response.msg.html

(cl:defclass <GetDynamicObstacles-response> (roslisp-msg-protocol:ros-message)
  ((position
    :reader position
    :initarg :position
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (velocity
    :reader velocity
    :initarg :velocity
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3)))
   (size
    :reader size
    :initarg :size
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3))))
)

(cl:defclass GetDynamicObstacles-response (<GetDynamicObstacles-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetDynamicObstacles-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetDynamicObstacles-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name onboard_detector-srv:<GetDynamicObstacles-response> is deprecated: use onboard_detector-srv:GetDynamicObstacles-response instead.")))

(cl:ensure-generic-function 'position-val :lambda-list '(m))
(cl:defmethod position-val ((m <GetDynamicObstacles-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader onboard_detector-srv:position-val is deprecated.  Use onboard_detector-srv:position instead.")
  (position m))

(cl:ensure-generic-function 'velocity-val :lambda-list '(m))
(cl:defmethod velocity-val ((m <GetDynamicObstacles-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader onboard_detector-srv:velocity-val is deprecated.  Use onboard_detector-srv:velocity instead.")
  (velocity m))

(cl:ensure-generic-function 'size-val :lambda-list '(m))
(cl:defmethod size-val ((m <GetDynamicObstacles-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader onboard_detector-srv:size-val is deprecated.  Use onboard_detector-srv:size instead.")
  (size m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetDynamicObstacles-response>) ostream)
  "Serializes a message object of type '<GetDynamicObstacles-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'position))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'velocity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'velocity))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'size))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'size))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetDynamicObstacles-response>) istream)
  "Deserializes a message object of type '<GetDynamicObstacles-response>"
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
  (cl:setf (cl:slot-value msg 'velocity) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'velocity)))
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetDynamicObstacles-response>)))
  "Returns string type for a service object of type '<GetDynamicObstacles-response>"
  "onboard_detector/GetDynamicObstaclesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetDynamicObstacles-response)))
  "Returns string type for a service object of type 'GetDynamicObstacles-response"
  "onboard_detector/GetDynamicObstaclesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetDynamicObstacles-response>)))
  "Returns md5sum for a message object of type '<GetDynamicObstacles-response>"
  "a508c5cda827e9832476a1de6a1e2f79")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetDynamicObstacles-response)))
  "Returns md5sum for a message object of type 'GetDynamicObstacles-response"
  "a508c5cda827e9832476a1de6a1e2f79")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetDynamicObstacles-response>)))
  "Returns full string definition for message of type '<GetDynamicObstacles-response>"
  (cl:format cl:nil "geometry_msgs/Vector3[] position~%geometry_msgs/Vector3[] velocity~%geometry_msgs/Vector3[] size~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetDynamicObstacles-response)))
  "Returns full string definition for message of type 'GetDynamicObstacles-response"
  (cl:format cl:nil "geometry_msgs/Vector3[] position~%geometry_msgs/Vector3[] velocity~%geometry_msgs/Vector3[] size~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetDynamicObstacles-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'velocity) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'size) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetDynamicObstacles-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetDynamicObstacles-response
    (cl:cons ':position (position msg))
    (cl:cons ':velocity (velocity msg))
    (cl:cons ':size (size msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetDynamicObstacles)))
  'GetDynamicObstacles-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetDynamicObstacles)))
  'GetDynamicObstacles-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetDynamicObstacles)))
  "Returns string type for a service object of type '<GetDynamicObstacles>"
  "onboard_detector/GetDynamicObstacles")