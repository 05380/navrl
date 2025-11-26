;; Auto-generated. Do not edit!


(when (boundp 'onboard_detector::GetDynamicObstacles)
  (if (not (find-package "ONBOARD_DETECTOR"))
    (make-package "ONBOARD_DETECTOR"))
  (shadow 'GetDynamicObstacles (find-package "ONBOARD_DETECTOR")))
(unless (find-package "ONBOARD_DETECTOR::GETDYNAMICOBSTACLES")
  (make-package "ONBOARD_DETECTOR::GETDYNAMICOBSTACLES"))
(unless (find-package "ONBOARD_DETECTOR::GETDYNAMICOBSTACLESREQUEST")
  (make-package "ONBOARD_DETECTOR::GETDYNAMICOBSTACLESREQUEST"))
(unless (find-package "ONBOARD_DETECTOR::GETDYNAMICOBSTACLESRESPONSE")
  (make-package "ONBOARD_DETECTOR::GETDYNAMICOBSTACLESRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass onboard_detector::GetDynamicObstaclesRequest
  :super ros::object
  :slots (_current_position _range ))

(defmethod onboard_detector::GetDynamicObstaclesRequest
  (:init
   (&key
    ((:current_position __current_position) (instance geometry_msgs::Point :init))
    ((:range __range) 0.0)
    )
   (send-super :init)
   (setq _current_position __current_position)
   (setq _range (float __range))
   self)
  (:current_position
   (&rest __current_position)
   (if (keywordp (car __current_position))
       (send* _current_position __current_position)
     (progn
       (if __current_position (setq _current_position (car __current_position)))
       _current_position)))
  (:range
   (&optional __range)
   (if __range (setq _range __range)) _range)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Point _current_position
    (send _current_position :serialization-length)
    ;; float64 _range
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Point _current_position
       (send _current_position :serialize s)
     ;; float64 _range
       (sys::poke _range (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Point _current_position
     (send _current_position :deserialize buf ptr-) (incf ptr- (send _current_position :serialization-length))
   ;; float64 _range
     (setq _range (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass onboard_detector::GetDynamicObstaclesResponse
  :super ros::object
  :slots (_position _velocity _size ))

(defmethod onboard_detector::GetDynamicObstaclesResponse
  (:init
   (&key
    ((:position __position) ())
    ((:velocity __velocity) ())
    ((:size __size) ())
    )
   (send-super :init)
   (setq _position __position)
   (setq _velocity __velocity)
   (setq _size __size)
   self)
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:velocity
   (&rest __velocity)
   (if (keywordp (car __velocity))
       (send* _velocity __velocity)
     (progn
       (if __velocity (setq _velocity (car __velocity)))
       _velocity)))
  (:size
   (&rest __size)
   (if (keywordp (car __size))
       (send* _size __size)
     (progn
       (if __size (setq _size (car __size)))
       _size)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Vector3[] _position
    (apply #'+ (send-all _position :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _velocity
    (apply #'+ (send-all _velocity :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _size
    (apply #'+ (send-all _size :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Vector3[] _position
     (write-long (length _position) s)
     (dolist (elem _position)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _velocity
     (write-long (length _velocity) s)
     (dolist (elem _velocity)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _size
     (write-long (length _size) s)
     (dolist (elem _size)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Vector3[] _position
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _position (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _position)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _velocity
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _velocity (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _velocity)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _size
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _size (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _size)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass onboard_detector::GetDynamicObstacles
  :super ros::object
  :slots ())

(setf (get onboard_detector::GetDynamicObstacles :md5sum-) "a508c5cda827e9832476a1de6a1e2f79")
(setf (get onboard_detector::GetDynamicObstacles :datatype-) "onboard_detector/GetDynamicObstacles")
(setf (get onboard_detector::GetDynamicObstacles :request) onboard_detector::GetDynamicObstaclesRequest)
(setf (get onboard_detector::GetDynamicObstacles :response) onboard_detector::GetDynamicObstaclesResponse)

(defmethod onboard_detector::GetDynamicObstaclesRequest
  (:response () (instance onboard_detector::GetDynamicObstaclesResponse :init)))

(setf (get onboard_detector::GetDynamicObstaclesRequest :md5sum-) "a508c5cda827e9832476a1de6a1e2f79")
(setf (get onboard_detector::GetDynamicObstaclesRequest :datatype-) "onboard_detector/GetDynamicObstaclesRequest")
(setf (get onboard_detector::GetDynamicObstaclesRequest :definition-)
      "geometry_msgs/Point current_position
float64 range

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---
geometry_msgs/Vector3[] position
geometry_msgs/Vector3[] velocity
geometry_msgs/Vector3[] size

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
")

(setf (get onboard_detector::GetDynamicObstaclesResponse :md5sum-) "a508c5cda827e9832476a1de6a1e2f79")
(setf (get onboard_detector::GetDynamicObstaclesResponse :datatype-) "onboard_detector/GetDynamicObstaclesResponse")
(setf (get onboard_detector::GetDynamicObstaclesResponse :definition-)
      "geometry_msgs/Point current_position
float64 range

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---
geometry_msgs/Vector3[] position
geometry_msgs/Vector3[] velocity
geometry_msgs/Vector3[] size

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
")



(provide :onboard_detector/GetDynamicObstacles "a508c5cda827e9832476a1de6a1e2f79")


