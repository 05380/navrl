;; Auto-generated. Do not edit!


(when (boundp 'navigation_runner::GetPolicyInference)
  (if (not (find-package "NAVIGATION_RUNNER"))
    (make-package "NAVIGATION_RUNNER"))
  (shadow 'GetPolicyInference (find-package "NAVIGATION_RUNNER")))
(unless (find-package "NAVIGATION_RUNNER::GETPOLICYINFERENCE")
  (make-package "NAVIGATION_RUNNER::GETPOLICYINFERENCE"))
(unless (find-package "NAVIGATION_RUNNER::GETPOLICYINFERENCEREQUEST")
  (make-package "NAVIGATION_RUNNER::GETPOLICYINFERENCEREQUEST"))
(unless (find-package "NAVIGATION_RUNNER::GETPOLICYINFERENCERESPONSE")
  (make-package "NAVIGATION_RUNNER::GETPOLICYINFERENCERESPONSE"))

(in-package "ROS")





(defclass navigation_runner::GetPolicyInferenceRequest
  :super ros::object
  :slots (_state _state_shape _lidar _lidar_shape _direction _direction_shape _dynamic_obstacle _dynamic_obstacle_shape ))

(defmethod navigation_runner::GetPolicyInferenceRequest
  (:init
   (&key
    ((:state __state) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:state_shape __state_shape) (make-array 0 :initial-element 0 :element-type :integer))
    ((:lidar __lidar) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:lidar_shape __lidar_shape) (make-array 0 :initial-element 0 :element-type :integer))
    ((:direction __direction) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:direction_shape __direction_shape) (make-array 0 :initial-element 0 :element-type :integer))
    ((:dynamic_obstacle __dynamic_obstacle) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:dynamic_obstacle_shape __dynamic_obstacle_shape) (make-array 0 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _state __state)
   (setq _state_shape __state_shape)
   (setq _lidar __lidar)
   (setq _lidar_shape __lidar_shape)
   (setq _direction __direction)
   (setq _direction_shape __direction_shape)
   (setq _dynamic_obstacle __dynamic_obstacle)
   (setq _dynamic_obstacle_shape __dynamic_obstacle_shape)
   self)
  (:state
   (&optional __state)
   (if __state (setq _state __state)) _state)
  (:state_shape
   (&optional __state_shape)
   (if __state_shape (setq _state_shape __state_shape)) _state_shape)
  (:lidar
   (&optional __lidar)
   (if __lidar (setq _lidar __lidar)) _lidar)
  (:lidar_shape
   (&optional __lidar_shape)
   (if __lidar_shape (setq _lidar_shape __lidar_shape)) _lidar_shape)
  (:direction
   (&optional __direction)
   (if __direction (setq _direction __direction)) _direction)
  (:direction_shape
   (&optional __direction_shape)
   (if __direction_shape (setq _direction_shape __direction_shape)) _direction_shape)
  (:dynamic_obstacle
   (&optional __dynamic_obstacle)
   (if __dynamic_obstacle (setq _dynamic_obstacle __dynamic_obstacle)) _dynamic_obstacle)
  (:dynamic_obstacle_shape
   (&optional __dynamic_obstacle_shape)
   (if __dynamic_obstacle_shape (setq _dynamic_obstacle_shape __dynamic_obstacle_shape)) _dynamic_obstacle_shape)
  (:serialization-length
   ()
   (+
    ;; float64[] _state
    (* 8    (length _state)) 4
    ;; int32[] _state_shape
    (* 4    (length _state_shape)) 4
    ;; float64[] _lidar
    (* 8    (length _lidar)) 4
    ;; int32[] _lidar_shape
    (* 4    (length _lidar_shape)) 4
    ;; float64[] _direction
    (* 8    (length _direction)) 4
    ;; int32[] _direction_shape
    (* 4    (length _direction_shape)) 4
    ;; float64[] _dynamic_obstacle
    (* 8    (length _dynamic_obstacle)) 4
    ;; int32[] _dynamic_obstacle_shape
    (* 4    (length _dynamic_obstacle_shape)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _state
     (write-long (length _state) s)
     (dotimes (i (length _state))
       (sys::poke (elt _state i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; int32[] _state_shape
     (write-long (length _state_shape) s)
     (dotimes (i (length _state_shape))
       (write-long (elt _state_shape i) s)
       )
     ;; float64[] _lidar
     (write-long (length _lidar) s)
     (dotimes (i (length _lidar))
       (sys::poke (elt _lidar i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; int32[] _lidar_shape
     (write-long (length _lidar_shape) s)
     (dotimes (i (length _lidar_shape))
       (write-long (elt _lidar_shape i) s)
       )
     ;; float64[] _direction
     (write-long (length _direction) s)
     (dotimes (i (length _direction))
       (sys::poke (elt _direction i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; int32[] _direction_shape
     (write-long (length _direction_shape) s)
     (dotimes (i (length _direction_shape))
       (write-long (elt _direction_shape i) s)
       )
     ;; float64[] _dynamic_obstacle
     (write-long (length _dynamic_obstacle) s)
     (dotimes (i (length _dynamic_obstacle))
       (sys::poke (elt _dynamic_obstacle i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; int32[] _dynamic_obstacle_shape
     (write-long (length _dynamic_obstacle_shape) s)
     (dotimes (i (length _dynamic_obstacle_shape))
       (write-long (elt _dynamic_obstacle_shape i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _state
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _state (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _state i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; int32[] _state_shape
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _state_shape (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _state_shape i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     ))
   ;; float64[] _lidar
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _lidar (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _lidar i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; int32[] _lidar_shape
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _lidar_shape (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _lidar_shape i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     ))
   ;; float64[] _direction
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _direction (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _direction i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; int32[] _direction_shape
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _direction_shape (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _direction_shape i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     ))
   ;; float64[] _dynamic_obstacle
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _dynamic_obstacle (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _dynamic_obstacle i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; int32[] _dynamic_obstacle_shape
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _dynamic_obstacle_shape (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _dynamic_obstacle_shape i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass navigation_runner::GetPolicyInferenceResponse
  :super ros::object
  :slots (_action ))

(defmethod navigation_runner::GetPolicyInferenceResponse
  (:init
   (&key
    ((:action __action) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _action __action)
   self)
  (:action
   (&optional __action)
   (if __action (setq _action __action)) _action)
  (:serialization-length
   ()
   (+
    ;; float64[] _action
    (* 8    (length _action)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _action
     (write-long (length _action) s)
     (dotimes (i (length _action))
       (sys::poke (elt _action i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _action
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _action (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _action i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(defclass navigation_runner::GetPolicyInference
  :super ros::object
  :slots ())

(setf (get navigation_runner::GetPolicyInference :md5sum-) "a4b6b43418ed234e9de75a5ccb110333")
(setf (get navigation_runner::GetPolicyInference :datatype-) "navigation_runner/GetPolicyInference")
(setf (get navigation_runner::GetPolicyInference :request) navigation_runner::GetPolicyInferenceRequest)
(setf (get navigation_runner::GetPolicyInference :response) navigation_runner::GetPolicyInferenceResponse)

(defmethod navigation_runner::GetPolicyInferenceRequest
  (:response () (instance navigation_runner::GetPolicyInferenceResponse :init)))

(setf (get navigation_runner::GetPolicyInferenceRequest :md5sum-) "a4b6b43418ed234e9de75a5ccb110333")
(setf (get navigation_runner::GetPolicyInferenceRequest :datatype-) "navigation_runner/GetPolicyInferenceRequest")
(setf (get navigation_runner::GetPolicyInferenceRequest :definition-)
      "float64[] state
int32[] state_shape
float64[] lidar
int32[] lidar_shape
float64[] direction
int32[] direction_shape
float64[] dynamic_obstacle
int32[] dynamic_obstacle_shape
---
float64[] action
")

(setf (get navigation_runner::GetPolicyInferenceResponse :md5sum-) "a4b6b43418ed234e9de75a5ccb110333")
(setf (get navigation_runner::GetPolicyInferenceResponse :datatype-) "navigation_runner/GetPolicyInferenceResponse")
(setf (get navigation_runner::GetPolicyInferenceResponse :definition-)
      "float64[] state
int32[] state_shape
float64[] lidar
int32[] lidar_shape
float64[] direction
int32[] direction_shape
float64[] dynamic_obstacle
int32[] dynamic_obstacle_shape
---
float64[] action
")



(provide :navigation_runner/GetPolicyInference "a4b6b43418ed234e9de75a5ccb110333")


