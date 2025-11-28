#!/usr/bin/env python3
"""
只是一个入口脚本 / 启动节点：
- 用 hydra 读取 cfg/train.yaml 配置；
- 调用 rospy.init_node("navigation_node") 初始化 ROS 节点；
- nav = Navigation(cfg)：创建 Navigation 对象；
- nav.run()：启动导航主循环；
- rospy.spin()：进入 ROS 事件循环。
"""
import rospy
#明 navigation_node.py 依赖 navigation.py 提供的 Navigation 类。
from navigation import Navigation
import hydra
import os

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts/cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    rospy.init_node("navigation_node", anonymous=True)
    nav = Navigation(cfg)
    nav.run()
    rospy.spin()


if __name__ == "__main__":
    main()