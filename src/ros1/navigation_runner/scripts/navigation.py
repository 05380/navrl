import rospy
import numpy as np
import torch
from map_manager.srv import RayCast
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Quaternion, Vector3
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from navigation_runner.srv import GetSafeAction, GetSafeActionMap
from onboard_detector.srv import GetDynamicObstacles
from map_manager.srv import GetStaticObstacles
from ppo import PPO
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from navigation_runner.srv import GetPolicyInference
from utils import vec_to_new_frame
import math
from std_srvs.srv import Empty
import tf.transformations
import time
import threading
import os

"""
 定义核心类 Navigation，里面实现了整个导航逻辑
- ROS 订阅/发布（odom、目标点、控制指令、可视化等）
- 调用地图服务、检测服务、安全盾服务；
- 加载 PPO 强化学习策略网络；
- 起飞流程、主循环 run()、安全检查线程等。
"""
class Navigation:
    def __init__(self, cfg):
        # 配合参数初始化
        self.cfg = cfg
        self.lidar_hbeams = int(360/self.cfg.sensor.lidar_hres)#雷达水平线束数
        self.raypoints = []
        self.dynamic_obstacles = []
        self.robot_size = 0.3 # radius
        #从配置计算出涉嫌角度分辨率
        self.raycast_vres = ((self.cfg.sensor.lidar_vfov[1] - self.cfg.sensor.lidar_vfov[0]))/(self.cfg.sensor.lidar_vbeams - 1) * np.pi/180.0
        self.raycast_hres = self.cfg.sensor.lidar_hres * np.pi/180.0
        # 状态变量初始化
        self.goal = None
        self.goal_received = False
        self.target_dir = None
        self.stable_times = 0
        self.has_action = False
        self.laser_points_msg = None

        # 控制模式配置
        self.height_control = False 
        self.px4_control = rospy.get_param('rl/use_px4', True)

        self.use_policy_server = False

        self.odom_received = False

        # ROS通信接口配置
         #PX4控制模式
        if (self.px4_control):
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)
            self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)

            self.action_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)#发布速度与高度
            self.pose_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)#发布位置

            self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
            self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)#解锁电机

            self.mavros_state = None
            self.offb_set_mode = SetModeRequest()
            self.offb_set_mode.custom_mode = 'OFFBOARD'
            self.arm_cmd = CommandBoolRequest()
            self.arm_cmd.value = True
         # 仿真模式
        else:
            self.odom_sub = rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, self.odom_callback)

            self.action_pub = rospy.Publisher("/CERLAB/quadcopter/cmd_vel", TwistStamped, queue_size=10)
            self.pose_pub = rospy.Publisher("/CERLAB/quadcopter/setpoint_pose", PoseStamped, queue_size=10)

        # 目标点订阅与可视化发布器（仿真和px4模式都需要做的）
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)#RViz 里 2D Nav Goal 按钮发的目标

        self.raycast_vis_pub = rospy.Publisher("/rl_navigation/raycast", MarkerArray, queue_size=10)#激光射线可视化
        self.cmd_vis_pub = rospy.Publisher("/rl_navigation/cmd", MarkerArray, queue_size=10)#动作箭头可视化
        self.goal_vis_pub = rospy.Publisher("rl_navigation/goal", MarkerArray, queue_size=10)#目标可视化
        self.rollout_traj_pub = rospy.Publisher("/rollout_traj", Path, queue_size=10)#预测轨迹可视化
        self.dynamic_obstacle_vis_pub = rospy.Publisher("/rl_navigation/in_range_dynamic_obstacles", MarkerArray, queue_size=10)#动态障碍可视化
        # 强化学习模型加载
        if (not self.use_policy_server):
            self.policy = self.init_model()
            self.policy.eval()

        # 安全线程启动
        # safety thread
        self.safety_stop = False
        safety_thread = threading.Thread(target = self.safety_check)
        safety_thread.start()
        #起飞程序
        self.takeoff()
  
    #构造观测/动作 Spec & 加载 PPO
    def init_model(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.cfg.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams), device=self.cfg.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.cfg.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.cfg.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.cfg.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.cfg.device), 
            })
        }).expand(1, action_dim).to(self.cfg.device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.cfg.device)

        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
        checkpoint = "navrl_checkpoint.pt"

        """
        os.path.join(file_dir, checkpoint): 拼接完整文件路径
        例如：/home/tiko/navrl_ws/src/ros1/navigation_runner/scripts/ckpts/navrl_checkpoint.pt
        
        torch.load(): PyTorch的模型加载函数
        读取.pt文件中的模型权重数据
        
        map_location=self.cfg.device: 将模型映射到指定设备（CPU或GPU）
        这确保模型能在不同设备间正确加载
        
        policy.load_state_dict(): 将加载的权重应用到网络中
        替换网络中随机初始化的权重
        使网络具有训练好的参数
        """
        policy.load_state_dict(torch.load(os.path.join(file_dir, checkpoint), map_location=self.cfg.device))
        return policy

    def takeoff(self):
        takeoff_height = 1.0
        r = rospy.Rate(10)
        while (not rospy.is_shutdown() and self.odom_received == False):
            print("[nav-ros]: Wait for robot odom...")
            r.sleep()

        takeoff_pose = PoseStamped()
        takeoff_pose.pose.position.x = self.odom.pose.pose.position.x
        takeoff_pose.pose.position.y = self.odom.pose.pose.position.y
        takeoff_pose.pose.position.z = takeoff_height
        takeoff_pose.pose.orientation = self.odom.pose.pose.orientation
        self.takeoff_pose = takeoff_pose
        if (self.px4_control):
            pose = PoseStamped()

            pose.pose.position.x = 0
            pose.pose.position.y = 0
            pose.pose.position.z = 2
            rate = rospy.Rate(20)
            # Send a few setpoints before starting
            for i in range(100):
                if(rospy.is_shutdown()):
                    break
                self.pose_pub.publish(pose)
                rate.sleep()
            last_req = rospy.Time.now()
        while (not rospy.is_shutdown() and not (np.abs(self.odom.pose.pose.position.z - takeoff_height) <= 0.2)):
            if (self.px4_control):
                if (self.mavros_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                    if(self.set_mode_client.call(self.offb_set_mode).mode_sent == True):
                        print("[nav-ros]: OFFBOARD enabled.")

                    last_req = rospy.Time.now()
                else:
                    if(not self.mavros_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                        if(self.arming_client.call(self.arm_cmd).success == True):
                            print("[nav-ros]: Vehicle armed.")

                        last_req = rospy.Time.now()
            self.pose_pub.publish(takeoff_pose)
            r.sleep()
        print("[nav-ros]: take off completed at height: ", takeoff_height)


    def safety_check(self):
        while not rospy.is_shutdown():
            if (self.safety_stop == False):
                input("[nav-ros]: Press Enter to STOP motion!\n")
                self.safety_stop = True
                self.stop_pose = PoseStamped()
                self.stop_pose.pose = self.odom.pose.pose
            else:
                input("[nav-ros]: Press Enter to CONTINUE motion!\n")
                self.safety_stop = False

    """
    调用地图节点的 occupancy_map/raycast 服务，从当前位置 pos 出发
    按给定起始水平角 start_angle 做 3D 射线扫描，得到一圈激光“打到的点”列表，
    并缓存原始点数组到 self.laser_points_msg。

    输入：当前位置 pos 和起始角度 start_angle。
   输出：raypoints: List[List[float]]，每个元素是 [x,y,z]，表示某条射线的击中点坐标（或最大量程端点）。
    """
    def get_raycast(self, pos: np.array , start_angle: float):
        raypoints = []
        try:
            raycast = rospy.ServiceProxy("occupancy_map/raycast", RayCast)#创建指向地图节点 occupancyMap.cpp 中 getRayCast 的客户端；
            pos_msg = Point()
            pos_msg.x = pos[0]
            pos_msg.y = pos[1]
            pos_msg.z = pos[2]

            response = raycast(pos_msg,
                               start_angle,
                               self.cfg.sensor.lidar_range, 
                               self.cfg.sensor.lidar_vfov[0], 
                               self.cfg.sensor.lidar_vfov[1], 
                               self.cfg.sensor.lidar_vbeams, 
                               self.cfg.sensor.lidar_hres
                               )
            
            num_points = int(len(response.points)/3)#response.points 是一个扁平的 float 数组 [x0,y0,z0,x1,y1,z1,...]。每 3 个数是一点，所以总点数是 len(points)/3。
            self.laser_points_msg = response.points

            for i in range(num_points):
                p = [response.points[3*i+0], response.points[3*i+1], response.points[3*i+2]]
                raypoints.append(p)
        except rospy.service.ServiceException as e:
            print("[nav-ros]: raycast func err!")        
        return raypoints

    def get_dynamic_obstacles(self, pos: np.array):
        dynamic_obstacle_pos = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.cfg.device)
        dynamic_obstacle_vel = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.cfg.device)
        dynamic_obstacle_size = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.cfg.device)
        try:
            "从 on-board detector 获取周围 distance_range=4m 内的动态障碍。"
            distance_range = 4.0
            pos_msg = Point()
            pos_msg.x = pos[0]
            pos_msg.y = pos[1]
            pos_msg.z = pos[2]

            get_obstacle = rospy.ServiceProxy("onboard_detector/get_dynamic_obstacles", GetDynamicObstacles)
            response = get_obstacle(pos_msg, distance_range)#请求当前位置与半径

            total_obs_num = len(response.position)#实际返回的障碍物数量
            #为每个障碍物填充到固定长度的张量：
            for i in range(self.cfg.algo.feature_extractor.dyn_obs_num):
                if (i < total_obs_num):
                    pos_vec = response.position[i]
                    vel_vec = response.velocity[i]
                    size_vec = response.size[i]
                    dynamic_obstacle_pos[i] = torch.tensor([pos_vec.x, pos_vec.y, pos_vec.z], dtype=torch.float, device=self.cfg.device)
                    dynamic_obstacle_vel[i] = torch.tensor([vel_vec.x, vel_vec.y, vel_vec.z], dtype=torch.float, device=self.cfg.device)
                    dynamic_obstacle_size[i] = torch.tensor([size_vec.x, size_vec.y, size_vec.z], dtype=torch.float, device=self.cfg.device)
        except rospy.service.ServiceException as e:
            print("[nav-ros]: dynamic obstacle func err!")   
        return dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size
    #获取静态障碍物的位置、尺寸、角度（可能是长方体障碍物）。
    def get_static_obstacles(self):
        static_obstacle_pos = []
        static_obstacle_size = []
        static_obstacle_angle = []
        try:
            get_static_obstacles_server = rospy.ServiceProxy("occupancy_map/get_static_obstacles", GetStaticObstacles)
            static_obstacle_response = get_static_obstacles_server()#返回静态障碍物的位置、尺寸、角度列表

            static_obstacle_pos = static_obstacle_response.position
            static_obstacle_size = static_obstacle_response.size
            static_obstacle_angle = static_obstacle_response.angle
        except rospy.service.ServiceException as e:
            print("[nav-ros]: static obstacle func err!")
        return static_obstacle_pos, static_obstacle_size, static_obstacle_angle
    
    #定期更新激光雷达扫描数据，为强化学习决策提供环境感知信息。
    def raycast_callback(self, event):
        if not self.odom_received or not self.goal_received:
            return
        pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        start_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        self.raypoints = self.get_raycast(pos, start_angle)

    def dynamic_obstacle_callback(self, event):
        if not self.odom_received:
            return
        pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size = self.get_dynamic_obstacles(pos)
        self.dynamic_obstacles = (dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size)

    def odom_callback(self, odom):
        self.odom = odom
        self.odom_received = True
    
    def state_callback(self, state):
        self.mavros_state = state
    
    def goal_callback(self, goal):
        if not self.odom_received:
            return

        self.goal = goal
        self.goal.pose.position.z = self.takeoff_pose.pose.position.z
        dir_x = self.goal.pose.position.x - self.odom.pose.pose.position.x
        dir_y = self.goal.pose.position.y - self.odom.pose.pose.position.y
        dir_z = self.goal.pose.position.z - self.odom.pose.pose.position.z
        self.target_dir = torch.tensor([dir_x, dir_y, dir_z], device=self.cfg.device) 

        self.goal_received = True
        self.stable_times = 0

    #用来把机体系速度转换到世界坐标系
    def quaternion_to_rotation_matrix(self, quaternion):
        # w, x, y, z = quaternion
        w = quaternion.w
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        xx, xy, xz = x**2, x*y, x*z
        yy, yz = y**2, y*z
        zz = z**2
        wx, wy, wz = w*x, w*y, w*z
        
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])        
    
    def check_obstacle(self, lidar_scan, dyn_obs_states):
        # return true if there is obstacles in the range
        # has_static = not torch.all(lidar_scan == 0.)
        # has_static = not torch.all(lidar_scan[..., 1:] < 0.2) # hardcode to tune

        quarter_size = lidar_scan.shape[2] // 4#每个扇区的beam数
        first_quarter_check, last_quarter_check = torch.all(lidar_scan[:, :, :quarter_size, 1:] < 0.2), torch.all(lidar_scan[:, :, -quarter_size:, 1:] < 0.2)

        has_static = (not first_quarter_check) or (not last_quarter_check)
        has_dynamic = not torch.all(dyn_obs_states == 0.)
        return has_static or has_dynamic

    def get_safe_action(self, vel_world, action_vel_world):#action_vel_world为调用ppo获得的预测的期望速度
        """
        （2）调用 get_action 得到 RL 策略的期望速度：
        python cmd_vel_world = self.get_action(pos, vel_world, goal).squeeze(...).numpy()
        （3）再调用 get_safe_action 做安全裁剪！！！，其实就是调用get_safe_action服务结合当前位置与ppo生成的速度进行安全裁剪。
        python safe_cmd_vel_world = self.get_safe_action(vel_world, cmd_vel_world)
        """ 
        safe_action = np.zeros(3)
        try:
            
            pos_msg = Point(x=self.odom.pose.pose.position.x, y=self.odom.pose.pose.position.y, z=self.odom.pose.pose.position.z)#从self.odom中获取当前位置信息
            get_safe_action = rospy.ServiceProxy("rl_navigation/get_safe_action", GetSafeAction) #get_safe_action 现在是一个可调用对象，后面可以 get_safe_action(...) 来发请求。
            vel_msg = Vector3(x=vel_world[0].item(), y=vel_world[1].item(), z=vel_world[2].item()) #ppo 算法输出的期望速度
            action_vel_msg = Vector3(x=action_vel_world[0], y=action_vel_world[1], z=action_vel_world[2])
            max_vel = np.sqrt(3. * self.cfg.algo.actor.action_limit**2)
            #初始化动态障碍物列表
            obstacle_pos_list = []
            obstacle_vel_list = []
            obstacle_size_list = []
            #填充动态障碍物列表
            for i in range(len(self.dynamic_obstacles[0])):
                if (self.dynamic_obstacles[2][i][0] != 0):
                    obs_pos = Vector3(x=self.dynamic_obstacles[0][i][0].item(), y=self.dynamic_obstacles[0][i][1].item(), z=self.dynamic_obstacles[0][i][2].item())
                    obs_vel = Vector3(x=self.dynamic_obstacles[1][i][0].item(), y=self.dynamic_obstacles[1][i][1].item(), z=self.dynamic_obstacles[1][i][2].item())
                    obs_size = Vector3(x=self.dynamic_obstacles[2][i][0].item(), y=self.dynamic_obstacles[2][i][1].item(), z=self.dynamic_obstacles[2][i][2].item())

                    obstacle_pos_list.append(obs_pos)
                    obstacle_vel_list.append(obs_vel)
                    obstacle_size_list.append(obs_size)
            response = get_safe_action(pos_msg, vel_msg, self.robot_size, obstacle_pos_list, obstacle_vel_list,\
                                    obstacle_size_list, self.laser_points_msg, self.cfg.sensor.lidar_range,\
                                    max(self.raycast_vres, self.raycast_hres), max_vel, action_vel_msg)
            safe_action = np.array([response.safe_action.x, response.safe_action.y, response.safe_action.z])#取出 x, y, z 三个分量，封装为 numpy 数组 [vx, vy, vz]，覆盖之前的 safe_action = np.zeros(3)，得到真正安全速度。
            return safe_action
        except rospy.service.ServiceException as e:
            # print("[nav-ros]: no safety running!")
            return action_vel_world   




    def get_safe_action_map(self, vel_world, action_vel_world):
        safe_action = np.zeros(3)#应当写成safe_action_map
        try:
            pos_msg = Point(x=self.odom.pose.pose.position.x, y=self.odom.pose.pose.position.y, z=self.odom.pose.pose.position.z)
            get_safe_action = rospy.ServiceProxy("rl_navigation/get_safe_action_map", GetSafeActionMap)#应该写成get_safe_action_map
            vel_msg = Vector3(x=vel_world[0].item(), y=vel_world[1].item(), z=vel_world[2].item())
            action_vel_msg = Vector3(x=action_vel_world[0], y=action_vel_world[1], z=action_vel_world[2])
            max_vel = np.sqrt(3. * self.cfg.algo.actor.action_limit**2)
            
            # Dynamic Obstacles
            obstacle_pos_list = []
            obstacle_vel_list = []
            obstacle_size_list = []
            for i in range(len(self.dynamic_obstacles[0])):
                if (self.dynamic_obstacles[2][i][0] != 0):
                    obs_pos = Vector3(x=self.dynamic_obstacles[0][i][0].item(), y=self.dynamic_obstacles[0][i][1].item(), z=self.dynamic_obstacles[0][i][2].item())
                    obs_vel = Vector3(x=self.dynamic_obstacles[1][i][0].item(), y=self.dynamic_obstacles[1][i][1].item(), z=self.dynamic_obstacles[1][i][2].item())
                    obs_size = Vector3(x=self.dynamic_obstacles[2][i][0].item(), y=self.dynamic_obstacles[2][i][1].item(), z=self.dynamic_obstacles[2][i][2].item())
                    obstacle_pos_list.append(obs_pos)
                    obstacle_vel_list.append(obs_vel)
                    obstacle_size_list.append(obs_size)
            
            # Static Obstacles
            static_obstacle_pos, static_obstacle_size, static_obstacle_angle = self.get_static_obstacles()

            response = get_safe_action(pos_msg, vel_msg, self.robot_size, obstacle_pos_list, obstacle_vel_list,\
                                    obstacle_size_list, static_obstacle_pos, static_obstacle_size,\
                                    static_obstacle_angle, max_vel, action_vel_msg)
            safe_action = np.array([response.safe_action.x, response.safe_action.y, response.safe_action.z])
            return safe_action
        except rospy.service.ServiceException as e:
            # print("[nav-ros]: no safety running!")
            return action_vel_world   
    
    #输入当前位姿、当前速度、目标位置，输出速度
    def get_action(self, pos: torch.Tensor, vel: torch.Tensor, goal: torch.Tensor): # use world velocity
        #计算相对目标位置与距离
        rpos = goal - pos
        distance = rpos.norm(dim=-1, keepdim=True)#三维距离
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)#水平距离
        distance_z = rpos[..., 2].unsqueeze(-1)#高度差

        #构造基于目标方向的局部坐标系
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[2] = 0.

        rpos_clipped = rpos / distance.clamp(1e-6) # start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d).squeeze(0).squeeze(0)

        # "relative" velocity
        vel_g = vec_to_new_frame(vel, target_dir_2d).squeeze(0).squeeze(0) # goal velocity

        # 将以上的机器人状态进行拼接
        #  drone_state = torch.cat([rpos_clipped, orientation, vel_g], dim=-1).squeeze(1)
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).unsqueeze(0)

        # Lidar States
        lidar_scan = torch.tensor(self.raypoints, device=self.cfg.device)
        lidar_scan = (lidar_scan - pos).norm(dim=-1).clamp_max(self.cfg.sensor.lidar_range).reshape(1, 1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams)
        lidar_scan = self.cfg.sensor.lidar_range - lidar_scan


        # 动态障碍物特征dynamic obstacle states
        dynamic_obstacle_pos = self.dynamic_obstacles[0].clone()
        dynamic_obstacle_vel = self.dynamic_obstacles[1].clone()
        dynamic_obstacle_size = self.dynamic_obstacles[2].clone()
        closest_dyn_obs_rpos = dynamic_obstacle_pos - pos
        closest_dyn_obs_rpos[dynamic_obstacle_size[:, 2] == 0] = 0.
        closest_dyn_obs_rpos[:, 2][dynamic_obstacle_size[:, 2] > 1] = 0.
        closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos.unsqueeze(0), target_dir_2d).squeeze(0)
        closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
        closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
        closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
        closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)


        closest_dyn_obs_vel_g = vec_to_new_frame(dynamic_obstacle_vel.unsqueeze(0), target_dir_2d).squeeze(0)
        
        obs_res = 0.25
        closest_dyn_obs_width = torch.max(dynamic_obstacle_size[:, 0], dynamic_obstacle_size[:, 1])
        closest_dyn_obs_width += self.robot_size * 2.
        closest_dyn_obs_width = torch.clamp(torch.ceil(closest_dyn_obs_width / 0.25) - 1, min=0, max=1./obs_res - 1)
        closest_dyn_obs_width[dynamic_obstacle_size[:, 2] == 0] = 0.
        closest_dyn_obs_height = dynamic_obstacle_size[:, 2]
        closest_dyn_obs_height[(closest_dyn_obs_height <= 1) & (closest_dyn_obs_height != 0)] = 1.
        closest_dyn_obs_height[closest_dyn_obs_height > 1] = 0.
        # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, \
        #                             closest_dyn_obs_width.unsqueeze(1), closest_dyn_obs_height.unsqueeze(1)], dim=-1).unsqueeze(0).unsqueeze(0)
        dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, \
                                    closest_dyn_obs_width.unsqueeze(1), closest_dyn_obs_height.unsqueeze(1)], dim=-1).unsqueeze(0).unsqueeze(0)
        # 组合成tensordict观测 states
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state,
                    "lidar": lidar_scan,
                    "direction": target_dir_2d,
                    "dynamic_obstacle": dyn_obs_states
                })
            })
        })
        #判断是否使用RL
        """

        check_obstacle 逻辑：
        看前方 1/4 和后方 1/4 的 Lidar 区域，有无障碍物
        若全部距离都很大（>0.2）就认为没有静态障碍。
        再看动态障碍特征是否全 0。
        只要有一种存在，就返回 True
        """
        has_obstacle_in_range = self.check_obstacle(lidar_scan, dyn_obs_states)
        
        # if (False):
        if (has_obstacle_in_range):
            if (not self.use_policy_server):#使用本地PPO策略
                with set_exploration_type(ExplorationType.MEAN):
                    output = self.policy(obs)
                vel_world = output["agents", "action"]
            else:#使用远程PPO策略
                try:
                    get_policy_inference = rospy.ServiceProxy("rl_navigation/GetPolicyInference", GetPolicyInference)

                    response = get_policy_inference(obs["agents"]["observation"]["state"].cpu().numpy().flatten().tolist(),
                                                    obs["agents"]["observation"]["state"].size(),
                                                    obs["agents"]["observation"]["lidar"].cpu().numpy().flatten().tolist(),
                                                    obs["agents"]["observation"]["lidar"].size(), 
                                                    obs["agents"]["observation"]["direction"].cpu().numpy().flatten().tolist(),
                                                    obs["agents"]["observation"]["direction"].size(),
                                                    obs["agents"]["observation"]["dynamic_obstacle"].cpu().numpy().flatten().tolist(),
                                                    obs["agents"]["observation"]["dynamic_obstacle"].size())
                    vel_world = torch.tensor(response.action, device=self.cfg.device, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                except rospy.service.ServiceException as e:
                    print("[nav-ros]: Policy server err!")
                    vel_world = torch.tensor([0., 0., 0.], device=self.cfg.device).unsqueeze(0).unsqueeze(0)#把返回的长度 3 动作转为形状 [1,1,3] 的张量，放回世界系速度。
        else:
            vel_world =  (goal - pos)/torch.norm(goal - pos) * self.cfg.algo.actor.action_limit#无障碍：直接 (goal - pos)/norm * action_limit，沿目标方向的最大速度
        return vel_world

    #作用：在当前环境状态下，反复调用 get_action 做前向“滚动预测”，得到未来一段时间内的轨迹点，用来可视化或分析策略行为。
    def get_rollout_traj(self, pos: torch.Tensor, vel: torch.Tensor, goal: torch.Tensor, dt=0.1, horizon=3.0):
        """
       输入：
       （1）pos：当前世界系位置（torch.Tensor，形如 [3] 或 [1,1,3]）。
        （2）vel：当前世界系速度。
        （3）goal：目标位置。
        （4）dt：时间步长，每次前向积分的时间间隔（默认 0.1s）。
        （5）horizon：预测总时长（默认 3.0s）。
        输出：traj：numpy 数组，形状约为 [N, 3]，表示从当前到 horizon 内的离散位置序列。
        """
       
        traj = [pos.cpu().detach().numpy()] #（1）把初始位置 pos 拷到 CPU、detach，转成 numpy；（2）作为轨迹的第一个点放入列表 traj。
        t = 0.
        while (t < horizon):
            vel_curr_world = self.get_action(pos, vel, goal)
            t += dt
            pos = (pos + dt * vel_curr_world).squeeze(0).squeeze(0)
            vel = vel_curr_world.squeeze(0).squeeze(0)
            traj.append(pos.cpu().detach().numpy())
        return np.array(traj)#把 Python 列表 [step0_pos, step1_pos, ...] 转成 numpy 数组返回。

    def control_callback(self, event):
        #前置检查阶段
        if (not self.odom_received):
            return

        if (not self.goal_received or len(self.raypoints) == 0 or len(self.dynamic_obstacles) == 0):
            self.pose_pub.publish(self.takeoff_pose)
            return

        if (self.safety_stop):#安全急停
            self.pose_pub.publish(self.stop_pose)
            return
        #朝向对齐阶段（先转向再平移）
        start_time = time.time()
        
        goal_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        _, _, curr_angle = tf.transformations.euler_from_quaternion([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w])
        angle_diff = np.abs(goal_angle - curr_angle)
        if (angle_diff > math.pi):
            angle_diff = np.abs(angle_diff - math.pi * 2)#（1）先算两者差值绝对值。（2）如果大于 π，则通过减去 2π 做 wrap，使角度差在 [0, π] 之间。
        if (angle_diff >= 0.1):
            """
            （1）若差值大于约 0.1 rad（约 5.7°），则：
                1）保持当前位置不变；
                2）仅修改姿态，使 yaw 对准目标方向 goal_angle；
                3）通过 pose_pub 发布姿态 setpoint，让飞控先转向。
            （2）然后 return，本周期不做平移控制。
            """
            pose_msg = PoseStamped()
            pose_msg.pose = self.odom.pose.pose
            quaternion = tf.transformations.quaternion_from_euler(0, 0, goal_angle)
            pose_msg.pose.orientation.w = quaternion[3]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            self.pose_pub.publish(pose_msg)
            return
        else:
            self.stable_times += 1
            if (self.stable_times <= 10):
                return
        #构造当前状态（pos / vel / goal）
        pos = torch.tensor([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z], device=self.cfg.device)
        goal = torch.tensor([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z], device=self.cfg.device)
        orientation = torch.tensor([self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z], device=self.cfg.device)
        rot = self.quaternion_to_rotation_matrix(self.odom.pose.pose.orientation)
        vel_body = np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_world = torch.tensor(rot @ vel_body, device=self.cfg.device, dtype=torch.float) # world vel
        
        # RL 策略动作与安全裁剪
        cmd_vel_world = self.get_action(pos, vel_world, goal).squeeze(0).squeeze(0).detach().cpu().numpy()        
        self.cmd_vel_world = cmd_vel_world.copy()

       
        safe_cmd_vel_world = self.get_safe_action(vel_world, cmd_vel_world) # 调用安全层对get_action函数获取的动作做裁剪
        # safe_cmd_vel_world = self.get_safe_action_map(vel_world, cmd_vel_world)
        # safe_cmd_vel_world[2] = 0
        self.safe_cmd_vel_world = safe_cmd_vel_world.copy()

        #转换到“无倾斜”的机体系局部速度
        quat_no_tilt = tf.transformations.quaternion_from_euler(0, 0, curr_angle)
        quat_msg = Quaternion()
        quat_msg.w = quat_no_tilt[3]
        quat_msg.x = quat_no_tilt[0]
        quat_msg.y = quat_no_tilt[1]
        quat_msg.z = quat_no_tilt[2]
        rot_no_tilt = self.quaternion_to_rotation_matrix(quat_msg)

        safe_cmd_vel_local = np.linalg.inv(rot_no_tilt) @ safe_cmd_vel_world#世界速度 → 局部速度

        # 根据与目标的距离进行相关的速度调节
        distance = (pos - goal).norm() 
        if (distance <= 3. and distance > 0.3):
            if (np.linalg.norm(safe_cmd_vel_local) != 0):
                safe_cmd_vel_local = 0.5 * safe_cmd_vel_local/np.linalg.norm(safe_cmd_vel_local)
                safe_cmd_vel_world = 0.5 * safe_cmd_vel_world/np.linalg.norm(safe_cmd_vel_world)
        elif (distance <= 1.0):
            safe_cmd_vel_local *= 0.
            safe_cmd_vel_world *= 0.

        # 根据控制模式构造最终控制消息
        if (self.px4_control):
            final_cmd_vel = PositionTarget()
            final_cmd_vel.coordinate_frame = final_cmd_vel.FRAME_LOCAL_NED
            final_cmd_vel.header.stamp = rospy.Time.now()
            final_cmd_vel.header.frame_id = "map"
            if (self.height_control):
                final_cmd_vel.velocity.x = safe_cmd_vel_world[0]
                final_cmd_vel.velocity.y = safe_cmd_vel_world[1]
                final_cmd_vel.velocity.z = safe_cmd_vel_world[2]
                final_cmd_vel.yaw = goal_angle
                final_cmd_vel.type_mask = final_cmd_vel.IGNORE_PX + final_cmd_vel.IGNORE_PY + final_cmd_vel.IGNORE_PZ + \
                    final_cmd_vel.IGNORE_AFX + final_cmd_vel.IGNORE_AFY + final_cmd_vel.IGNORE_AFZ + final_cmd_vel.IGNORE_YAW_RATE
            else:
                final_cmd_vel.velocity.x = safe_cmd_vel_world[0]
                final_cmd_vel.velocity.y = safe_cmd_vel_world[1]
                final_cmd_vel.position.z = self.takeoff_pose.pose.position.z
                final_cmd_vel.yaw = goal_angle
                final_cmd_vel.type_mask = final_cmd_vel.IGNORE_PX + final_cmd_vel.IGNORE_PY + final_cmd_vel.IGNORE_VZ + \
                    final_cmd_vel.IGNORE_AFX + final_cmd_vel.IGNORE_AFY + final_cmd_vel.IGNORE_AFZ + final_cmd_vel.IGNORE_YAW_RATE                           
        else:
            final_cmd_vel = TwistStamped()
            final_cmd_vel.header.stamp = rospy.Time.now()
            final_cmd_vel.twist.linear.x = safe_cmd_vel_local[0]
            final_cmd_vel.twist.linear.y = safe_cmd_vel_local[1]
            if (self.height_control):
                final_cmd_vel.twist.linear.z = safe_cmd_vel_world[2]
            else:
                final_cmd_vel.twist.linear.z = 0
        self.action_pub.publish(final_cmd_vel)
        self.has_action = True

        # rollout_traj = self.get_rollout_traj(pos, vel_world, goal, dt=0.1, horizon=3.0)
        # traj_msg = Path()
        # traj_msg.header.frame_id = "map"
        # for i in range(len(rollout_traj)):
        #     p = PoseStamped()
        #     p.pose.position.x = rollout_traj[i][0]
        #     p.pose.position.y = rollout_traj[i][1]
        #     p.pose.position.z = rollout_traj[i][2]
        #     traj_msg.poses.append(p)
        # self.rollout_traj_pub.publish(traj_msg)
        end_time = time.time()
        # print("[nav-ros]: control time ", end_time - start_time)
        
    def pause_sim():
        rospy.wait_for_service('/gazebo/pause_physics')
        pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pause()

    def unpause_sim():
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause()

    def run(self):
        raycast_timer = rospy.Timer(rospy.Duration(0.05), self.raycast_callback)
        raycast_vis_timer = rospy.Timer(rospy.Duration(0.05), self.raycast_vis_callback)
        control_timer = rospy.Timer(rospy.Duration(0.05), self.control_callback)
        goal_vis_timer = rospy.Timer(rospy.Duration(0.05), self.goal_vis_callback)
        dynamic_obstacle_timer = rospy.Timer(rospy.Duration(0.05), self.dynamic_obstacle_callback)
        dynamic_obstacle_vis_timer = rospy.Timer(rospy.Duration(0.05), self.dynamic_obstacle_vis_callback)
        cmd_vis_timer = rospy.Timer(rospy.Duration(0.05), self.cmd_vis_callback)

    def raycast_vis_callback(self, event):
        if not self.odom_received and not self.goal_received:
            return
        msg = MarkerArray()
        pos = self.odom.pose.pose.position
        direction_init = None
        for i in range(len(self.raypoints)):
            point = Marker()
            point.header.frame_id = "map"
            point.header.stamp = rospy.get_rostime()
            point.ns = "raycast_points"
            point.id = i
            point.type = point.SPHERE
            point.action = point.ADD
            point.pose.position.x = self.raypoints[i][0]
            point.pose.position.y = self.raypoints[i][1]
            point.pose.position.z = self.raypoints[i][2]
            point.lifetime = rospy.Time(0.5)
            point.scale.x = 0.1
            point.scale.y = 0.1
            point.scale.z = 0.1
            point.color.a = 1.0
            point.color.r = 1.0
            msg.markers.append(point)

            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = rospy.get_rostime()
            line.ns = "raycast_lines"
            line.id = i
            line.type = line.LINE_LIST
            p = Point()
            p.x = self.raypoints[i][0]
            p.y = self.raypoints[i][1]
            p.z = self.raypoints[i][2]
            line.points.append(p)
            line.points.append(pos)
            line.scale.x = 0.03
            line.scale.y = 0.03
            line.scale.z = 0.03
            x_diff = (p.x - self.odom.pose.pose.position.x)
            y_diff = (p.y - self.odom.pose.pose.position.y)
            direction = np.array([x_diff, y_diff])
            direction = direction/np.linalg.norm(direction)
            if (i == 0 or (np.linalg.norm(direction - direction_init) <= 0.1)):
                line.color.b = 1.0
                line.color.a = 1.0
                if (i == 0):
                    direction_init = direction
            else:
                line.color.g = 1.0
                line.color.a = 0.5
            line.lifetime = rospy.Time(0.5)
            msg.markers.append(line)
        self.raycast_vis_pub.publish(msg)
    
    def goal_vis_callback(self, event):
        if not self.goal_received:
            return
        msg = MarkerArray()
        goal_point = Marker()
        goal_point.header.frame_id = "map"
        goal_point.header.stamp = rospy.get_rostime()
        goal_point.ns = "goal_point"
        goal_point.id = 1
        goal_point.type = goal_point.SPHERE
        goal_point.action = goal_point.ADD
        goal_point.pose.position.x = self.goal.pose.position.x
        goal_point.pose.position.y = self.goal.pose.position.y
        goal_point.pose.position.z = self.goal.pose.position.z
        goal_point.lifetime = rospy.Time(0.1)
        goal_point.scale.x = 0.3
        goal_point.scale.y = 0.3
        goal_point.scale.z = 0.3
        goal_point.color.r = 1.0
        goal_point.color.b = 1.0
        goal_point.color.a = 1.0
        msg.markers.append(goal_point)
        self.goal_vis_pub.publish(msg)

    def dynamic_obstacle_vis_callback(self, event):
        if (len(self.dynamic_obstacles) == 0):
            return
        dynamic_obstacle_pos = self.dynamic_obstacles[0]
        dynamic_obstacle_size = self.dynamic_obstacles[2]

        msg = MarkerArray()
        for i in range(dynamic_obstacle_pos.size(0)):
            pos = dynamic_obstacle_pos[i]
            size = dynamic_obstacle_size[i]

            # Increase the width
            width = torch.max(size[0], size[1])
            height = size[2]

            # Create the marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "dynamic_obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2] 
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = width
            marker.scale.y = width
            marker.scale.z = height 
            marker.color.a = 0.5  # Alpha value
            marker.color.r = 1.0  # Red color
            marker.color.g = 0.0
            marker.color.b = 0.0

            msg.markers.append(marker)

        # Publish the marker array
        self.dynamic_obstacle_vis_pub.publish(msg)            

    def cmd_vis_callback(self, event):
        if (not self.has_action):
            return
        msg = MarkerArray()

        # rl action vis
        rl_action_arrow = Marker()
        rl_action_arrow.header.frame_id = "map"
        rl_action_arrow.header.stamp = rospy.get_rostime()
        rl_action_arrow.ns = "rl_action"
        rl_action_arrow.id = 0
        rl_action_arrow.type = rl_action_arrow.ARROW
        rl_action_arrow.action = rl_action_arrow.ADD

        # start
        agent_pos = Point()
        agent_pos.x = self.odom.pose.pose.position.x
        agent_pos.y = self.odom.pose.pose.position.y
        agent_pos.z = self.odom.pose.pose.position.z
        
        # end
        vel_end = Point()
        vel_end.x = self.cmd_vel_world[0] + agent_pos.x
        vel_end.y = self.cmd_vel_world[1] + agent_pos.y
        vel_end.z = self.cmd_vel_world[2] + agent_pos.z

        rl_action_arrow.points.append(agent_pos)
        rl_action_arrow.points.append(vel_end)
        rl_action_arrow.lifetime = rospy.Duration(0.1)
        rl_action_arrow.scale.x = 0.06
        rl_action_arrow.scale.y = 0.06
        rl_action_arrow.scale.z = 0.06
        rl_action_arrow.color.a = 1.0
        rl_action_arrow.color.r = 1.0
        rl_action_arrow.color.g = 0.0
        rl_action_arrow.color.b = 0.0
        msg.markers.append(rl_action_arrow)


        # safe action vis
        safe_action_arrow = Marker()
        safe_action_arrow.header.frame_id = "map"
        safe_action_arrow.header.stamp = rospy.get_rostime()
        safe_action_arrow.ns = "safe_action"
        safe_action_arrow.id = 1
        safe_action_arrow.type = safe_action_arrow.ARROW
        safe_action_arrow.action = safe_action_arrow.ADD

        # start
        agent_pos = Point()
        agent_pos.x = self.odom.pose.pose.position.x
        agent_pos.y = self.odom.pose.pose.position.y
        agent_pos.z = self.odom.pose.pose.position.z

        # end
        vel_end = Point()
        vel_end.x = self.safe_cmd_vel_world[0] + agent_pos.x
        vel_end.y = self.safe_cmd_vel_world[1] + agent_pos.y
        vel_end.z = self.safe_cmd_vel_world[2] + agent_pos.z

        safe_action_arrow.points.append(agent_pos)
        safe_action_arrow.points.append(vel_end)
        safe_action_arrow.lifetime = rospy.Duration(0.1)
        safe_action_arrow.scale.x = 0.06
        safe_action_arrow.scale.y = 0.06
        safe_action_arrow.scale.z = 0.06
        safe_action_arrow.color.a = 1.0
        safe_action_arrow.color.r = 0.0
        safe_action_arrow.color.g = 1.0
        safe_action_arrow.color.b = 0.0

        msg.markers.append(safe_action_arrow)
        self.cmd_vis_pub.publish(msg)