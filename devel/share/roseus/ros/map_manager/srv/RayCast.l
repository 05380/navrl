;; Auto-generated. Do not edit!


(when (boundp 'map_manager::RayCast)
  (if (not (find-package "MAP_MANAGER"))
    (make-package "MAP_MANAGER"))
  (shadow 'RayCast (find-package "MAP_MANAGER")))
(unless (find-package "MAP_MANAGER::RAYCAST")
  (make-package "MAP_MANAGER::RAYCAST"))
(unless (find-package "MAP_MANAGER::RAYCASTREQUEST")
  (make-package "MAP_MANAGER::RAYCASTREQUEST"))
(unless (find-package "MAP_MANAGER::RAYCASTRESPONSE")
  (make-package "MAP_MANAGER::RAYCASTRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))




(defclass map_manager::RayCastRequest
  :super ros::object
  :slots (_position _startAngle _range _vfov_min _vfov_max _vbeams _hres ))

(defmethod map_manager::RayCastRequest
  (:init
   (&key
    ((:position __position) (instance geometry_msgs::Point :init))
    ((:startAngle __startAngle) 0.0)
    ((:range __range) 0.0)
    ((:vfov_min __vfov_min) 0.0)
    ((:vfov_max __vfov_max) 0.0)
    ((:vbeams __vbeams) 0)
    ((:hres __hres) 0.0)
    )
   (send-super :init)
   (setq _position __position)
   (setq _startAngle (float __startAngle))
   (setq _range (float __range))
   (setq _vfov_min (float __vfov_min))
   (setq _vfov_max (float __vfov_max))
   (setq _vbeams (round __vbeams))
   (setq _hres (float __hres))
   self)
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:startAngle
   (&optional __startAngle)
   (if __startAngle (setq _startAngle __startAngle)) _startAngle)
  (:range
   (&optional __range)
   (if __range (setq _range __range)) _range)
  (:vfov_min
   (&optional __vfov_min)
   (if __vfov_min (setq _vfov_min __vfov_min)) _vfov_min)
  (:vfov_max
   (&optional __vfov_max)
   (if __vfov_max (setq _vfov_max __vfov_max)) _vfov_max)
  (:vbeams
   (&optional __vbeams)
   (if __vbeams (setq _vbeams __vbeams)) _vbeams)
  (:hres
   (&optional __hres)
   (if __hres (setq _hres __hres)) _hres)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Point _position
    (send _position :serialization-length)
    ;; float64 _startAngle
    8
    ;; float64 _range
    8
    ;; float64 _vfov_min
    8
    ;; float64 _vfov_max
    8
    ;; int32 _vbeams
    4
    ;; float64 _hres
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Point _position
       (send _position :serialize s)
     ;; float64 _startAngle
       (sys::poke _startAngle (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _range
       (sys::poke _range (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _vfov_min
       (sys::poke _vfov_min (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _vfov_max
       (sys::poke _vfov_max (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; int32 _vbeams
       (write-long _vbeams s)
     ;; float64 _hres
       (sys::poke _hres (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Point _position
     (send _position :deserialize buf ptr-) (incf ptr- (send _position :serialization-length))
   ;; float64 _startAngle
     (setq _startAngle (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _range
     (setq _range (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _vfov_min
     (setq _vfov_min (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _vfov_max
     (setq _vfov_max (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; int32 _vbeams
     (setq _vbeams (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float64 _hres
     (setq _hres (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass map_manager::RayCastResponse
  :super ros::object
  :slots (_points ))

(defmethod map_manager::RayCastResponse
  (:init
   (&key
    ((:points __points) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _points __points)
   self)
  (:points
   (&optional __points)
   (if __points (setq _points __points)) _points)
  (:serialization-length
   ()
   (+
    ;; float64[] _points
    (* 8    (length _points)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _points
     (write-long (length _points) s)
     (dotimes (i (length _points))
       (sys::poke (elt _points i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _points
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _points (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _points i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(defclass map_manager::RayCast
  :super ros::object
  :slots ())

(setf (get map_manager::RayCast :md5sum-) "0c76c296d09a6c11167d8e932a6ebc50")
(setf (get map_manager::RayCast :datatype-) "map_manager/RayCast")
(setf (get map_manager::RayCast :request) map_manager::RayCastRequest)
(setf (get map_manager::RayCast :response) map_manager::RayCastResponse)

(defmethod map_manager::RayCastRequest
  (:response () (instance map_manager::RayCastResponse :init)))

(setf (get map_manager::RayCastRequest :md5sum-) "0c76c296d09a6c11167d8e932a6ebc50")
(setf (get map_manager::RayCastRequest :datatype-) "map_manager/RayCastRequest")
(setf (get map_manager::RayCastRequest :definition-)
      "geometry_msgs/Point position
float64 startAngle
float64 range
float64 vfov_min
float64 vfov_max
int32 vbeams
float64 hres

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---
float64[] points
")

(setf (get map_manager::RayCastResponse :md5sum-) "0c76c296d09a6c11167d8e932a6ebc50")
(setf (get map_manager::RayCastResponse :datatype-) "map_manager/RayCastResponse")
(setf (get map_manager::RayCastResponse :definition-)
      "geometry_msgs/Point position
float64 startAngle
float64 range
float64 vfov_min
float64 vfov_max
int32 vbeams
float64 hres

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---
float64[] points
")



(provide :map_manager/RayCast "0c76c296d09a6c11167d8e932a6ebc50")


