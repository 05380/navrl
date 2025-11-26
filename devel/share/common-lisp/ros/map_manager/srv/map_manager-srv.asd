
(cl:in-package :asdf)

(defsystem "map_manager-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "CheckPosCollision" :depends-on ("_package_CheckPosCollision"))
    (:file "_package_CheckPosCollision" :depends-on ("_package"))
    (:file "GetStaticObstacles" :depends-on ("_package_GetStaticObstacles"))
    (:file "_package_GetStaticObstacles" :depends-on ("_package"))
    (:file "RayCast" :depends-on ("_package_RayCast"))
    (:file "_package_RayCast" :depends-on ("_package"))
  ))