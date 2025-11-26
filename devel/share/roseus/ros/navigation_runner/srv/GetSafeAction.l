;; Auto-generated. Do not edit!


(when (boundp 'navigation_runner::GetSafeAction)
  (if (not (find-package "NAVIGATION_RUNNER"))
    (make-package "NAVIGATION_RUNNER"))
  (shadow 'GetSafeAction (find-package "NAVIGATION_RUNNER")))
(unless (find-package "NAVIGATION_RUNNER::GETSAFEACTION")
  (make-package "NAVIGATION_RUNNER::GETSAFEACTION"))
(unless (find-package "NAVIGATION_RUNNER::GETSAFEACTIONREQUEST")
  (make-package "NAVIGATION_RUNNER::GETSAFEACTIONREQUEST"))
(unless (find-package "NAVIGATION_RUNNER::GETSAFEACTIONRESPONSE")
  (make-package "NAVIGATION_RUNNER::GETSAFEACTIONRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass navigation_runner::GetSafeActionRequest
  :super ros::object
  :slots (_agent_position _agent_velocity _agent_size _obs_position _obs_velocity _obs_size _laser_points _laser_range _laser_res _max_velocity _rl_velocity ))

(defmethod navigation_runner::GetSafeActionRequest
  (:init
   (&key
    ((:agent_position __agent_position) (instance geometry_msgs::Point :init))
    ((:agent_velocity __agent_velocity) (instance geometry_msgs::Vector3 :init))
    ((:agent_size __agent_size) 0.0)
    ((:obs_position __obs_position) ())
    ((:obs_velocity __obs_velocity) ())
    ((:obs_size __obs_size) ())
    ((:laser_points __laser_points) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:laser_range __laser_range) 0.0)
    ((:laser_res __laser_res) 0.0)
    ((:max_velocity __max_velocity) 0.0)
    ((:rl_velocity __rl_velocity) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _agent_position __agent_position)
   (setq _agent_velocity __agent_velocity)
   (setq _agent_size (float __agent_size))
   (setq _obs_position __obs_position)
   (setq _obs_velocity __obs_velocity)
   (setq _obs_size __obs_size)
   (setq _laser_points __laser_points)
   (setq _laser_range (float __laser_range))
   (setq _laser_res (float __laser_res))
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
  (:obs_position
   (&rest __obs_position)
   (if (keywordp (car __obs_position))
       (send* _obs_position __obs_position)
     (progn
       (if __obs_position (setq _obs_position (car __obs_position)))
       _obs_position)))
  (:obs_velocity
   (&rest __obs_velocity)
   (if (keywordp (car __obs_velocity))
       (send* _obs_velocity __obs_velocity)
     (progn
       (if __obs_velocity (setq _obs_velocity (car __obs_velocity)))
       _obs_velocity)))
  (:obs_size
   (&rest __obs_size)
   (if (keywordp (car __obs_size))
       (send* _obs_size __obs_size)
     (progn
       (if __obs_size (setq _obs_size (car __obs_size)))
       _obs_size)))
  (:laser_points
   (&optional __laser_points)
   (if __laser_points (setq _laser_points __laser_points)) _laser_points)
  (:laser_range
   (&optional __laser_range)
   (if __laser_range (setq _laser_range __laser_range)) _laser_range)
  (:laser_res
   (&optional __laser_res)
   (if __laser_res (setq _laser_res __laser_res)) _laser_res)
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
    ;; geometry_msgs/Vector3[] _obs_position
    (apply #'+ (send-all _obs_position :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _obs_velocity
    (apply #'+ (send-all _obs_velocity :serialization-length)) 4
    ;; geometry_msgs/Vector3[] _obs_size
    (apply #'+ (send-all _obs_size :serialization-length)) 4
    ;; float64[] _laser_points
    (* 8    (length _laser_points)) 4
    ;; float64 _laser_range
    8
    ;; float64 _laser_res
    8
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
     ;; geometry_msgs/Vector3[] _obs_position
     (write-long (length _obs_position) s)
     (dolist (elem _obs_position)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _obs_velocity
     (write-long (length _obs_velocity) s)
     (dolist (elem _obs_velocity)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Vector3[] _obs_size
     (write-long (length _obs_size) s)
     (dolist (elem _obs_size)
       (send elem :serialize s)
       )
     ;; float64[] _laser_points
     (write-long (length _laser_points) s)
     (dotimes (i (length _laser_points))
       (sys::poke (elt _laser_points i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64 _laser_range
       (sys::poke _laser_range (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _laser_res
       (sys::poke _laser_res (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
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
   ;; geometry_msgs/Vector3[] _obs_position
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _obs_position (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _obs_position)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _obs_velocity
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _obs_velocity (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _obs_velocity)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Vector3[] _obs_size
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _obs_size (let (r) (dotimes (i n) (push (instance geometry_msgs::Vector3 :init) r)) r))
     (dolist (elem- _obs_size)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; float64[] _laser_points
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _laser_points (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _laser_points i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64 _laser_range
     (setq _laser_range (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _laser_res
     (setq _laser_res (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _max_velocity
     (setq _max_velocity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; geometry_msgs/Vector3 _rl_velocity
     (send _rl_velocity :deserialize buf ptr-) (incf ptr- (send _rl_velocity :serialization-length))
   ;;
   self)
  )

(defclass navigation_runner::GetSafeActionResponse
  :super ros::object
  :slots (_safe_action ))

(defmethod navigation_runner::GetSafeActionResponse
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

(defclass navigation_runner::GetSafeAction
  :super ros::object
  :slots ())

(setf (get navigation_runner::GetSafeAction :md5sum-) "54fa8c288cd542a68cd6fb86a87e59f8")
(setf (get navigation_runner::GetSafeAction :datatype-) "navigation_runner/GetSafeAction")
(setf (get navigation_runner::GetSafeAction :request) navigation_runner::GetSafeActionRequest)
(setf (get navigation_runner::GetSafeAction :response) navigation_runner::GetSafeActionResponse)

(defmethod navigation_runner::GetSafeActionRequest
  (:response () (instance navigation_runner::GetSafeActionResponse :init)))

(setf (get navigation_runner::GetSafeActionRequest :md5sum-) "54fa8c288cd542a68cd6fb86a87e59f8")
(setf (get navigation_runner::GetSafeActionRequest :datatype-) "navigation_runner/GetSafeActionRequest")
(setf (get navigation_runner::GetSafeActionRequest :definition-)
      "geometry_msgs/Point agent_position
geometry_msgs/Vector3 agent_velocity
float64 agent_size

# dynamic obstacles
geometry_msgs/Vector3[] obs_position
geometry_msgs/Vector3[] obs_velocity
geometry_msgs/Vector3[] obs_size

# static obstacles
float64[] laser_points
float64 laser_range
float64 laser_res # use the larger one among horizonal and vertical

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

(setf (get navigation_runner::GetSafeActionResponse :md5sum-) "54fa8c288cd542a68cd6fb86a87e59f8")
(setf (get navigation_runner::GetSafeActionResponse :datatype-) "navigation_runner/GetSafeActionResponse")
(setf (get navigation_runner::GetSafeActionResponse :definition-)
      "geometry_msgs/Point agent_position
geometry_msgs/Vector3 agent_velocity
float64 agent_size

# dynamic obstacles
geometry_msgs/Vector3[] obs_position
geometry_msgs/Vector3[] obs_velocity
geometry_msgs/Vector3[] obs_size

# static obstacles
float64[] laser_points
float64 laser_range
float64 laser_res # use the larger one among horizonal and vertical

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



(provide :navigation_runner/GetSafeAction "54fa8c288cd542a68cd6fb86a87e59f8")


