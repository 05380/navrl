;; Auto-generated. Do not edit!


(when (boundp 'map_manager::GetStaticObstacles)
  (if (not (find-package "MAP_MANAGER"))
    (make-package "MAP_MANAGER"))
  (shadow 'GetStaticObstacles (find-package "MAP_MANAGER")))
(unless (find-package "MAP_MANAGER::GETSTATICOBSTACLES")
  (make-package "MAP_MANAGER::GETSTATICOBSTACLES"))
(unless (find-package "MAP_MANAGER::GETSTATICOBSTACLESREQUEST")
  (make-package "MAP_MANAGER::GETSTATICOBSTACLESREQUEST"))
(unless (find-package "MAP_MANAGER::GETSTATICOBSTACLESRESPONSE")
  (make-package "MAP_MANAGER::GETSTATICOBSTACLESRESPONSE"))

(in-package "ROS")



(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass map_manager::GetStaticObstaclesRequest
  :super ros::object
  :slots ())

(defmethod map_manager::GetStaticObstaclesRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass map_manager::GetStaticObstaclesResponse
  :super ros::object
  :slots (_position _size _angle ))

(defmethod map_manager::GetStaticObstaclesResponse
  (:init
   (&key
    ((:position __position) ())
    ((:size __size) ())
    ((:angle __angle) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _position __position)
   (setq _size __size)
   (setq _angle __angle)
   self)
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:size
   (&rest __size)
   (if (keywordp (car __size))
       (send* _size __size)
     (progn
       (if __size (setq _size (car __size)))
       _size)))
  (:angle
   (&optional __angle)
   (if __angle (setq _angle __angle)) _angle)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Vector3[] _position
    (apply #'+ (send-all _position :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _size
    (apply #'+ (send-all _size :serialization-length)) 4
    ;; float64[] _angle
    (* 8    (length _angle)) 4
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
     ;; geometry_msgs/Vector3[] _size
     (write-long (length _size) s)
     (dolist (elem _size)
       (send elem :serialize s)
       )
     ;; float64[] _angle
     (write-long (length _angle) s)
     (dotimes (i (length _angle))
       (sys::poke (elt _angle i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
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
   ;; geometry_msgs/Vector3[] _size
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _size (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _size)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; float64[] _angle
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _angle (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _angle i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(defclass map_manager::GetStaticObstacles
  :super ros::object
  :slots ())

(setf (get map_manager::GetStaticObstacles :md5sum-) "10afeb58fe82a6ef3c941519f4670020")
(setf (get map_manager::GetStaticObstacles :datatype-) "map_manager/GetStaticObstacles")
(setf (get map_manager::GetStaticObstacles :request) map_manager::GetStaticObstaclesRequest)
(setf (get map_manager::GetStaticObstacles :response) map_manager::GetStaticObstaclesResponse)

(defmethod map_manager::GetStaticObstaclesRequest
  (:response () (instance map_manager::GetStaticObstaclesResponse :init)))

(setf (get map_manager::GetStaticObstaclesRequest :md5sum-) "10afeb58fe82a6ef3c941519f4670020")
(setf (get map_manager::GetStaticObstaclesRequest :datatype-) "map_manager/GetStaticObstaclesRequest")
(setf (get map_manager::GetStaticObstaclesRequest :definition-)
      "---
geometry_msgs/Vector3[] position
geometry_msgs/Vector3[] size
float64[] angle

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

(setf (get map_manager::GetStaticObstaclesResponse :md5sum-) "10afeb58fe82a6ef3c941519f4670020")
(setf (get map_manager::GetStaticObstaclesResponse :datatype-) "map_manager/GetStaticObstaclesResponse")
(setf (get map_manager::GetStaticObstaclesResponse :definition-)
      "---
geometry_msgs/Vector3[] position
geometry_msgs/Vector3[] size
float64[] angle

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



(provide :map_manager/GetStaticObstacles "10afeb58fe82a6ef3c941519f4670020")


