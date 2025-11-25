
(cl:in-package :asdf)

(defsystem "onboard_detector-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "GetDynamicObstacles" :depends-on ("_package_GetDynamicObstacles"))
    (:file "_package_GetDynamicObstacles" :depends-on ("_package"))
  ))