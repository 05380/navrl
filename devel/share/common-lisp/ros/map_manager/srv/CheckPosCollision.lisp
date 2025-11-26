; Auto-generated. Do not edit!


(cl:in-package map_manager-srv)


;//! \htmlinclude CheckPosCollision-request.msg.html

(cl:defclass <CheckPosCollision-request> (roslisp-msg-protocol:ros-message)
  ((x
    :reader x
    :initarg :x
    :type cl:float
    :initform 0.0)
   (y
    :reader y
    :initarg :y
    :type cl:float
    :initform 0.0)
   (z
    :reader z
    :initarg :z
    :type cl:float
    :initform 0.0)
   (inflated
    :reader inflated
    :initarg :inflated
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass CheckPosCollision-request (<CheckPosCollision-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CheckPosCollision-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CheckPosCollision-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name map_manager-srv:<CheckPosCollision-request> is deprecated: use map_manager-srv:CheckPosCollision-request instead.")))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <CheckPosCollision-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:x-val is deprecated.  Use map_manager-srv:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <CheckPosCollision-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:y-val is deprecated.  Use map_manager-srv:y instead.")
  (y m))

(cl:ensure-generic-function 'z-val :lambda-list '(m))
(cl:defmethod z-val ((m <CheckPosCollision-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:z-val is deprecated.  Use map_manager-srv:z instead.")
  (z m))

(cl:ensure-generic-function 'inflated-val :lambda-list '(m))
(cl:defmethod inflated-val ((m <CheckPosCollision-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:inflated-val is deprecated.  Use map_manager-srv:inflated instead.")
  (inflated m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CheckPosCollision-request>) ostream)
  "Serializes a message object of type '<CheckPosCollision-request>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'inflated) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CheckPosCollision-request>) istream)
  "Deserializes a message object of type '<CheckPosCollision-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'z) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'inflated) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CheckPosCollision-request>)))
  "Returns string type for a service object of type '<CheckPosCollision-request>"
  "map_manager/CheckPosCollisionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckPosCollision-request)))
  "Returns string type for a service object of type 'CheckPosCollision-request"
  "map_manager/CheckPosCollisionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CheckPosCollision-request>)))
  "Returns md5sum for a message object of type '<CheckPosCollision-request>"
  "5b2617caaa5faa9b1c910b8c4d0cf2ea")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CheckPosCollision-request)))
  "Returns md5sum for a message object of type 'CheckPosCollision-request"
  "5b2617caaa5faa9b1c910b8c4d0cf2ea")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CheckPosCollision-request>)))
  "Returns full string definition for message of type '<CheckPosCollision-request>"
  (cl:format cl:nil "float64 x~%float64 y~%float64 z~%bool inflated~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CheckPosCollision-request)))
  "Returns full string definition for message of type 'CheckPosCollision-request"
  (cl:format cl:nil "float64 x~%float64 y~%float64 z~%bool inflated~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CheckPosCollision-request>))
  (cl:+ 0
     8
     8
     8
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CheckPosCollision-request>))
  "Converts a ROS message object to a list"
  (cl:list 'CheckPosCollision-request
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
    (cl:cons ':z (z msg))
    (cl:cons ':inflated (inflated msg))
))
;//! \htmlinclude CheckPosCollision-response.msg.html

(cl:defclass <CheckPosCollision-response> (roslisp-msg-protocol:ros-message)
  ((occupied
    :reader occupied
    :initarg :occupied
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass CheckPosCollision-response (<CheckPosCollision-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CheckPosCollision-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CheckPosCollision-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name map_manager-srv:<CheckPosCollision-response> is deprecated: use map_manager-srv:CheckPosCollision-response instead.")))

(cl:ensure-generic-function 'occupied-val :lambda-list '(m))
(cl:defmethod occupied-val ((m <CheckPosCollision-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader map_manager-srv:occupied-val is deprecated.  Use map_manager-srv:occupied instead.")
  (occupied m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CheckPosCollision-response>) ostream)
  "Serializes a message object of type '<CheckPosCollision-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'occupied) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CheckPosCollision-response>) istream)
  "Deserializes a message object of type '<CheckPosCollision-response>"
    (cl:setf (cl:slot-value msg 'occupied) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CheckPosCollision-response>)))
  "Returns string type for a service object of type '<CheckPosCollision-response>"
  "map_manager/CheckPosCollisionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckPosCollision-response)))
  "Returns string type for a service object of type 'CheckPosCollision-response"
  "map_manager/CheckPosCollisionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CheckPosCollision-response>)))
  "Returns md5sum for a message object of type '<CheckPosCollision-response>"
  "5b2617caaa5faa9b1c910b8c4d0cf2ea")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CheckPosCollision-response)))
  "Returns md5sum for a message object of type 'CheckPosCollision-response"
  "5b2617caaa5faa9b1c910b8c4d0cf2ea")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CheckPosCollision-response>)))
  "Returns full string definition for message of type '<CheckPosCollision-response>"
  (cl:format cl:nil "bool occupied~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CheckPosCollision-response)))
  "Returns full string definition for message of type 'CheckPosCollision-response"
  (cl:format cl:nil "bool occupied~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CheckPosCollision-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CheckPosCollision-response>))
  "Converts a ROS message object to a list"
  (cl:list 'CheckPosCollision-response
    (cl:cons ':occupied (occupied msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'CheckPosCollision)))
  'CheckPosCollision-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'CheckPosCollision)))
  'CheckPosCollision-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CheckPosCollision)))
  "Returns string type for a service object of type '<CheckPosCollision>"
  "map_manager/CheckPosCollision")