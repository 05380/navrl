;; Auto-generated. Do not edit!


(when (boundp 'navigation_runner::GetSafeActionMap)
  (if (not (find-package "NAVIGATION_RUNNER"))
    (make-package "NAVIGATION_RUNNER"))
  (shadow 'GetSafeActionMap (find-package "NAVIGATION_RUNNER")))
(unless (find-package "NAVIGATION_RUNNER::GETSAFEACTIONMAP")
  (make-package "NAVIGATION_RUNNER::GETSAFEACTIONMAP"))
(unless (find-package "NAVIGATION_RUNNER::GETSAFEACTIONMAPREQUEST")
  (make-package "NAVIGATION_RUNNER::GETSAFEACTIONMAPREQUEST"))
(unless (find-package "NAVIGATION_RUNNER::GETSAFEACTIONMAPRESPONSE")
  (make-package "NAVIGATION_RUNNER::GETSAFEACTIONMAPRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass navigation_runner::GetSafeActionMapRequest
  :super ros::object
  :slots (_agent_position _agent_velocity _agent_size _dynamic_obs_position _dynamic_obs_velocity _dynamic_obs_size _static_obs_position _static_obs_size _static_obs_angle _max_velocity _rl_velocity ))

(defmethod navigation_runner::GetSafeActionMapRequest
  (:init
   (&key
    ((:agent_position __agent_position) (instance geometry_msgs::Point :init))
    ((:agent_velocity __agent_velocity) (instance geometry_msgs::Vector3 :init))
    ((:agent_size __agent_size) 0.0)
    ((:dynamic_obs_position __dynamic_obs_position) ())
    ((:dynamic_obs_velocity __dynamic_obs_velocity) ())
    ((:dynamic_obs_size __dynamic_obs_size) ())
    ((:static_obs_position __static_obs_position) ())
    ((:static_obs_size __static_obs_size) ())
    ((:static_obs_angle __static_obs_angle) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:max_velocity __max_velocity) 0.0)
    ((:rl_velocity __rl_velocity) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _agent_position __agent_position)
   (setq _agent_velocity __agent_velocity)
   (setq _agent_size (float __agent_size))
   (setq _dynamic_obs_position __dynamic_obs_position)
   (setq _dynamic_obs_velocity __dynamic_obs_velocity)
   (setq _dynamic_obs_size __dynamic_obs_size)
   (setq _static_obs_position __static_obs_position)
   (setq _static_obs_size __static_obs_size)
   (setq _static_obs_angle __static_obs_angle)
   (setq _max_velocity (float __max_velocity))
   (setq _rl_velocity __rl_velocity)
   self)
  (:agent_position
   (&rest __agent_position)
   (if (keywordp (car __agent_position))
       (send* _agent_position __agent_position)
     (progn
       (if __agent_position (setq _agent_position (car __agent_position)))
       _agent_position)))
  (:agent_velocity
   (&rest __agent_velocity)
   (if (keywordp (car __agent_velocity))
       (send* _agent_velocity __agent_velocity)
     (progn
       (if __agent_velocity (setq _agent_velocity (car __agent_velocity)))
       _agent_velocity)))
  (:agent_size
   (&optional __agent_size)
   (if __agent_size (setq _agent_size __agent_size)) _agent_size)
  (:dynamic_obs_position
   (&rest __dynamic_obs_position)
   (if (keywordp (car __dynamic_obs_position))
       (send* _dynamic_obs_position __dynamic_obs_position)
     (progn
       (if __dynamic_obs_position (setq _dynamic_obs_position (car __dynamic_obs_position)))
       _dynamic_obs_position)))
  (:dynamic_obs_velocity
   (&rest __dynamic_obs_velocity)
   (if (keywordp (car __dynamic_obs_velocity))
       (send* _dynamic_obs_velocity __dynamic_obs_velocity)
     (progn
       (if __dynamic_obs_velocity (setq _dynamic_obs_velocity (car __dynamic_obs_velocity)))
       _dynamic_obs_velocity)))
  (:dynamic_obs_size
   (&rest __dynamic_obs_size)
   (if (keywordp (car __dynamic_obs_size))
       (send* _dynamic_obs_size __dynamic_obs_size)
     (progn
       (if __dynamic_obs_size (setq _dynamic_obs_size (car __dynamic_obs_size)))
       _dynamic_obs_size)))
  (:static_obs_position
   (&rest __static_obs_position)
   (if (keywordp (car __static_obs_position))
       (send* _static_obs_position __static_obs_position)
     (progn
       (if __static_obs_position (setq _static_obs_position (car __static_obs_position)))
       _static_obs_position)))
  (:static_obs_size
   (&rest __static_obs_size)
   (if (keywordp (car __static_obs_size))
       (send* _static_obs_size __static_obs_size)
     (progn
       (if __static_obs_size (setq _static_obs_size (car __static_obs_size)))
       _static_obs_size)))
  (:static_obs_angle
   (&optional __static_obs_angle)
   (if __static_obs_angle (setq _static_obs_angle __static_obs_angle)) _static_obs_angle)
  (:max_velocity
   (&optional __max_velocity)
   (if __max_velocity (setq _max_velocity __max_velocity)) _max_velocity)
  (:rl_velocity
   (&rest __rl_velocity)
   (if (keywordp (car __rl_velocity))
       (send* _rl_velocity __rl_velocity)
     (progn
       (if __rl_velocity (setq _rl_velocity (car __rl_velocity)))
       _rl_velocity)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Point _agent_position
    (send _agent_position :serialization-length)
    ;; geometry_msgs/Vector3 _agent_velocity
    (send _agent_velocity :serialization-length)
    ;; float64 _agent_size
    8
    ;; geometry_msgs/Vector3[] _dynamic_obs_position
    (apply #'+ (send-all _dynamic_obs_position :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _dynamic_obs_velocity
    (apply #'+ (send-all _dynamic_obs_velocity :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _dynamic_obs_size
    (apply #'+ (send-all _dynamic_obs_size :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _static_obs_position
    (apply #'+ (send-all _static_obs_position :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _static_obs_size
    (apply #'+ (send-all _static_obs_size :serialization-length)) 4
    ;; float64[] _static_obs_angle
    (* 8    (length _static_obs_angle)) 4
    ;; float64 _max_velocity
    8
    ;; geometry_msgs/Vector3 _rl_velocity
    (send _rl_velocity :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Point _agent_position
       (send _agent_position :serialize s)
     ;; geometry_msgs/Vector3 _agent_velocity
       (send _agent_velocity :serialize s)
     ;; float64 _agent_size
       (sys::poke _agent_size (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; geometry_msgs/Vector3[] _dynamic_obs_position
     (write-long (length _dynamic_obs_position) s)
     (dolist (elem _dynamic_obs_position)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _dynamic_obs_velocity
     (write-long (length _dynamic_obs_velocity) s)
     (dolist (elem _dynamic_obs_velocity)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _dynamic_obs_size
     (write-long (length _dynamic_obs_size) s)
     (dolist (elem _dynamic_obs_size)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _static_obs_position
     (write-long (length _static_obs_position) s)
     (dolist (elem _static_obs_position)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _static_obs_size
     (write-long (length _static_obs_size) s)
     (dolist (elem _static_obs_size)
       (send elem :serialize s)
       )
     ;; float64[] _static_obs_angle
     (write-long (length _static_obs_angle) s)
     (dotimes (i (length _static_obs_angle))
       (sys::poke (elt _static_obs_angle i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64 _max_velocity
       (sys::poke _max_velocity (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; geometry_msgs/Vector3 _rl_velocity
       (send _rl_velocity :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Point _agent_position
     (send _agent_position :deserialize buf ptr-) (incf ptr- (send _agent_position :serialization-length))
   ;; geometry_msgs/Vector3 _agent_velocity
     (send _agent_velocity :deserialize buf ptr-) (incf ptr- (send _agent_velocity :serialization-length))
   ;; float64 _agent_size
     (setq _agent_size (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; geometry_msgs/Vector3[] _dynamic_obs_position
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _dynamic_obs_position (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _dynamic_obs_position)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _dynamic_obs_velocity
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _dynamic_obs_velocity (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _dynamic_obs_velocity)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _dynamic_obs_size
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _dynamic_obs_size (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _dynamic_obs_size)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _static_obs_position
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _static_obs_position (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _static_obs_position)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _static_obs_size
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _static_obs_size (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _static_obs_size)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; float64[] _static_obs_angle
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _static_obs_angle (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _static_obs_angle i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64 _max_velocity
     (setq _max_velocity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; geometry_msgs/Vector3 _rl_velocity
     (send _rl_velocity :deserialize buf ptr-) (incf ptr- (send _rl_velocity :serialization-length))
   ;;
   self)
  )

(defclass navigation_runner::GetSafeActionMapResponse
  :super ros::object
  :slots (_safe_action ))

(defmethod navigation_runner::GetSafeActionMapResponse
  (:init
   (&key
    ((:safe_action __safe_action) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _safe_action __safe_action)
   self)
  (:safe_action
   (&rest __safe_action)
   (if (keywordp (car __safe_action))
       (send* _safe_action __safe_action)
     (progn
       (if __safe_action (setq _safe_action (car __safe_action)))
       _safe_action)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Vector3 _safe_action
    (send _safe_action :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Vector3 _safe_action
       (send _safe_action :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Vector3 _safe_action
     (send _safe_action :deserialize buf ptr-) (incf ptr- (send _safe_action :serialization-length))
   ;;
   self)
  )

(defclass navigation_runner::GetSafeActionMap
  :super ros::object
  :slots ())

(setf (get navigation_runner::GetSafeActionMap :md5sum-) "4635b39c982fd59eaaa757b70b204f4c")
(setf (get navigation_runner::GetSafeActionMap :datatype-) "navigation_runner/GetSafeActionMap")
(setf (get navigation_runner::GetSafeActionMap :request) navigation_runner::GetSafeActionMapRequest)
(setf (get navigation_runner::GetSafeActionMap :response) navigation_runner::GetSafeActionMapResponse)

(defmethod navigation_runner::GetSafeActionMapRequest
  (:response () (instance navigation_runner::GetSafeActionMapResponse :init)))

(setf (get navigation_runner::GetSafeActionMapRequest :md5sum-) "4635b39c982fd59eaaa757b70b204f4c")
(setf (get navigation_runner::GetSafeActionMapRequest :datatype-) "navigation_runner/GetSafeActionMapRequest")
(setf (get navigation_runner::GetSafeActionMapRequest :definition-)
      "geometry_msgs/Point agent_position
geometry_msgs/Vector3 agent_velocity
float64 agent_size

# dynamic obstacles
geometry_msgs/Vector3[] dynamic_obs_position
geometry_msgs/Vector3[] dynamic_obs_velocity
geometry_msgs/Vector3[] dynamic_obs_size

# static obstacles
geometry_msgs/Vector3[] static_obs_position
geometry_msgs/Vector3[] static_obs_size
float64[] static_obs_angle

float64 max_velocity
geometry_msgs/Vector3 rl_velocity

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

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
---
geometry_msgs/Vector3 safe_action

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

(setf (get navigation_runner::GetSafeActionMapResponse :md5sum-) "4635b39c982fd59eaaa757b70b204f4c")
(setf (get navigation_runner::GetSafeActionMapResponse :datatype-) "navigation_runner/GetSafeActionMapResponse")
(setf (get navigation_runner::GetSafeActionMapResponse :definition-)
      "geometry_msgs/Point agent_position
geometry_msgs/Vector3 agent_velocity
float64 agent_size

# dynamic obstacles
geometry_msgs/Vector3[] dynamic_obs_position
geometry_msgs/Vector3[] dynamic_obs_velocity
geometry_msgs/Vector3[] dynamic_obs_size

# static obstacles
geometry_msgs/Vector3[] static_obs_position
geometry_msgs/Vector3[] static_obs_size
float64[] static_obs_angle

float64 max_velocity
geometry_msgs/Vector3 rl_velocity

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

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
---
geometry_msgs/Vector3 safe_action

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



(provide :navigation_runner/GetSafeActionMap "4635b39c982fd59eaaa757b70b204f4c")


