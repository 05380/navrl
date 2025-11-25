;; Auto-generated. Do not edit!


(when (boundp 'map_manager::CheckPosCollision)
  (if (not (find-package "MAP_MANAGER"))
    (make-package "MAP_MANAGER"))
  (shadow 'CheckPosCollision (find-package "MAP_MANAGER")))
(unless (find-package "MAP_MANAGER::CHECKPOSCOLLISION")
  (make-package "MAP_MANAGER::CHECKPOSCOLLISION"))
(unless (find-package "MAP_MANAGER::CHECKPOSCOLLISIONREQUEST")
  (make-package "MAP_MANAGER::CHECKPOSCOLLISIONREQUEST"))
(unless (find-package "MAP_MANAGER::CHECKPOSCOLLISIONRESPONSE")
  (make-package "MAP_MANAGER::CHECKPOSCOLLISIONRESPONSE"))

(in-package "ROS")





(defclass map_manager::CheckPosCollisionRequest
  :super ros::object
  :slots (_x _y _z _inflated ))

(defmethod map_manager::CheckPosCollisionRequest
  (:init
   (&key
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    ((:z __z) 0.0)
    ((:inflated __inflated) nil)
    )
   (send-super :init)
   (setq _x (float __x))
   (setq _y (float __y))
   (setq _z (float __z))
   (setq _inflated __inflated)
   self)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:z
   (&optional __z)
   (if __z (setq _z __z)) _z)
  (:inflated
   (&optional (__inflated :null))
   (if (not (eq __inflated :null)) (setq _inflated __inflated)) _inflated)
  (:serialization-length
   ()
   (+
    ;; float64 _x
    8
    ;; float64 _y
    8
    ;; float64 _z
    8
    ;; bool _inflated
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _x
       (sys::poke _x (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _y
       (sys::poke _y (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _z
       (sys::poke _z (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _inflated
       (if _inflated (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _x
     (setq _x (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _y
     (setq _y (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _z
     (setq _z (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _inflated
     (setq _inflated (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass map_manager::CheckPosCollisionResponse
  :super ros::object
  :slots (_occupied ))

(defmethod map_manager::CheckPosCollisionResponse
  (:init
   (&key
    ((:occupied __occupied) nil)
    )
   (send-super :init)
   (setq _occupied __occupied)
   self)
  (:occupied
   (&optional (__occupied :null))
   (if (not (eq __occupied :null)) (setq _occupied __occupied)) _occupied)
  (:serialization-length
   ()
   (+
    ;; bool _occupied
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _occupied
       (if _occupied (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _occupied
     (setq _occupied (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass map_manager::CheckPosCollision
  :super ros::object
  :slots ())

(setf (get map_manager::CheckPosCollision :md5sum-) "5b2617caaa5faa9b1c910b8c4d0cf2ea")
(setf (get map_manager::CheckPosCollision :datatype-) "map_manager/CheckPosCollision")
(setf (get map_manager::CheckPosCollision :request) map_manager::CheckPosCollisionRequest)
(setf (get map_manager::CheckPosCollision :response) map_manager::CheckPosCollisionResponse)

(defmethod map_manager::CheckPosCollisionRequest
  (:response () (instance map_manager::CheckPosCollisionResponse :init)))

(setf (get map_manager::CheckPosCollisionRequest :md5sum-) "5b2617caaa5faa9b1c910b8c4d0cf2ea")
(setf (get map_manager::CheckPosCollisionRequest :datatype-) "map_manager/CheckPosCollisionRequest")
(setf (get map_manager::CheckPosCollisionRequest :definition-)
      "float64 x
float64 y
float64 z
bool inflated
---
bool occupied
")

(setf (get map_manager::CheckPosCollisionResponse :md5sum-) "5b2617caaa5faa9b1c910b8c4d0cf2ea")
(setf (get map_manager::CheckPosCollisionResponse :datatype-) "map_manager/CheckPosCollisionResponse")
(setf (get map_manager::CheckPosCollisionResponse :definition-)
      "float64 x
float64 y
float64 z
bool inflated
---
bool occupied
")



(provide :map_manager/CheckPosCollision "5b2617caaa5faa9b1c910b8c4d0cf2ea")


