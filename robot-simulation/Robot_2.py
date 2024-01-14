import rm_utilities as Rm
import numpy as np
import pyqtgraph.opengl as gl

"""
Simulation of a 6 DOF robot.
- Cédric Wassenaar & Joeri Verkerk
- C.M.Wassenaar@student.hhs.nl
- 24-09-2018
"""


class Link(object):
    """Storage type for robot arm object"""

    def __init__(self):
        self.arm_length = 0
        self.arm: gl.GLMeshItem | None = None
        self.frame: gl.GLAxisItem | None = None
        self.joint: gl.GLMeshItem | None = None
        self.DH_parameters = [0, 0, 0, 0]
        self.homogenious_matrix: np.array | None = None
        self.hg_matrix_list = []
        self.p = None
        # self.arm_color = (0.4, 1, 0.4, 1)  # 设置默认颜色
        # self.joint_color = (0.8, 0.2, 1, 1)  # 设置默认颜色
        self.homogenious_matrix_new = None


class Robot_2(object):
    """Class to create robot object for OpenGl implementation"""

    def __init__(self, view3D):

        self.view3D = view3D
        # Presets
        self.radius = 0.2
        self.width = .6 * 2 * self.radius
        self.depth = self.width
        self.depth_cylinder = 1.2 * self.width
        self.segments = 40
        self.joint_color = (0.9, 0.5, 0.9, 1)  # yellow
        self.arm_color = (0.1, 0.5, 0.5, 1)  # light blue
        self.cycle = 0
        self.iteration = 0
        self.N = 0
        self.ready = False
        self.print_text = []
        self.position = None
        # define arm lengths
        self.arm_lengths = [1.75,
                            1.60,
                            1.25,
                            0.00,
                            0.00,
                            0.00,
                            0.50,
                            0.00]

        # Create a list of arm pieces
        self.link = [Link()] * (len(self.arm_lengths) + 1)

        # define Target rotation for the end effector
        # self.target_rotation = Rm.make_rotation_matrix("x", np.radians(00))
        self.target_rotation = np.eye(3, 3)
        # define trajectory
        # X, Y, Z, Ax, Ay, Az (degrees)
        self.trajectory = np.array([[-1.5, 1, 2, 0, 0, 0],
                                    [1.5, 0.5, 4, 0, 0, np.pi / 2],
                                    [1.5, 2.0, 1, 0, np.pi, np.pi / 2],
                                    [-0.0, 2.0, 3, 0, np.pi, 0],
                                    [-1.5, 2, 2, 0, 0, 0]])
        self.setup()

    def setup(self):
        """
        Function for user defined objects and functions
        :return:
        """
        # create all arm pieces
        self.link[0].frame = self.create_main_axis()
        self.link[1] = self.add_arm(self.link[0], self.arm_lengths[0])
        self.link[1].frame.setSize(0.5, 0.5, 0.5)
        self.link[2] = self.add_arm(self.link[1], self.arm_lengths[1])
        self.link[2].frame.setSize(0.5, 0.5, 0.5)
        self.link[3] = self.add_arm(self.link[2], self.arm_lengths[2])
        self.link[3].frame.setSize(0.1, 0.1, 0.1)
        self.link[4] = self.add_arm(self.link[3], self.arm_lengths[3])
        self.link[4].frame.setSize(0.1, 0.1, 0.1)
        self.link[5] = self.add_arm(self.link[4], self.arm_lengths[4])
        self.link[5].frame.setSize(0.1, 0.1, 0.1)
        self.link[6] = self.add_arm(self.link[5], self.arm_lengths[5])
        self.link[6].frame.setSize(0.1, 0.1, 0.1)
        self.link[7] = self.add_arm(self.link[6], self.arm_lengths[6])
        self.link[7].frame.setSize(0.1, 0.1, 0.1)
        self.link[8] = self.add_arm(self.link[7], self.arm_lengths[7])
        # Rotate arms to work with Denavit Hartenberg parameters
        self.link[1].arm.rotate(90, 1, 0, 0)
        self.link[2].arm.rotate(-90, 0, 1, 0)
        self.link[3].arm.rotate(-90, 0, 1, 0)
        # Predefine static values of the Denavit Hartenberg parameters
        self.link[0].DH_parameters = [0, 0, 0, 0]
        self.link[1].DH_parameters = [0, 1.57079, self.arm_lengths[0], 0]
        self.link[2].DH_parameters = [self.arm_lengths[1], 0, 0, 0]
        self.link[3].DH_parameters = [self.arm_lengths[2], 0, 0, 0]
        self.link[4].DH_parameters = [0, 1.57079, 0, 1.57079]
        self.link[5].DH_parameters = [0, -1.57079, 0, 0]
        self.link[6].DH_parameters = [0, 1.57079, 0, 0]
        self.link[7].DH_parameters = [0, 0, 0, 0]
        self.link[8].DH_parameters = [0, 0, self.arm_lengths[6], 0]

    def calculate_dynamic_path(self, pos):
        rotation = Rm.rotate_xyz(pos[3:])
        # rotation是通过将轨迹点的朝向角度（Euler角）传递给
        # rotate_xyz函数计算得到的综合旋转矩阵。这个矩阵描述了机械臂在给定位置的姿态。
        # 使用Rm.rotate_xyz函数对位置信息中的旋转部分（通常是从第四个元素开始的）进行旋转变换，得到新的旋转信息。
        pos[0] = pos[0] - self.arm_lengths[6] * rotation[0, 2]
        pos[1] = pos[1] - self.arm_lengths[6] * rotation[1, 2]
        pos[2] = pos[2] - self.arm_lengths[6] * rotation[2, 2]

        # self.arm_lengths: 是机械臂各个关节的长度信息，作为逆运动学计算的输入之一。

        # pos: 是机械臂末端执行器在三维空间中的目标位置，由前文中从轨迹点复制得到。
        # Rm.inverse_algorithm_3DOF: 是负责计算三自由度机械臂逆运动学的函数，返回两个值：
        # angles: 包含机械臂各关节的角度信息，这是逆运动学计算的结果。
        # error: 是一个标志，指示逆运动学计算是否成功。如果计算成功，error 通常为 False；否则，为 True。
        angles, error = Rm.inverse_algorithm_3DOF(self.arm_lengths, pos)

        # If the robot arm is not capable of reaching the given coordinate it will raise an error
        if not error:
            # The dynamic values of the Denavit Hartenberg parameters are now updated.
            self.link[1].DH_parameters[3] = angles[0]
            self.link[2].DH_parameters[3] = angles[1]
            self.link[3].DH_parameters[3] = angles[2]
            # 这里更新了机械臂的
            # Denavit - Hartenberg（DH）参数。Denavit - Hartenberg参数描述了机械臂各个关节之间的几何关系和运动属性。在这里，针对机械臂的第1、2
            # 和3个关节（索引从1开始），将DH参数的第4个元素更新为通过逆运动学计算得到的对应关节的角度值angles。
            # 这一步是为了确保机械臂的参数与逆运动学计算得到的结果保持同步，以反映当前机械臂在给定目标位置时的准确配置。
            # The homogenious matrices are created for link 0 to 4
            for i in range(5):
                self.link[i].homogenious_matrix = Rm.make_DH_matrix(self.link[i].DH_parameters)
            # 它遍历了机械臂的前5个关节（索引从0到4），并为每个关节更新对应的齐次矩阵：
            # The H03 matrix is calculated using the @ symbol (matrix multiplication)
            H03 = self.link[0].homogenious_matrix @ \
                  self.link[1].homogenious_matrix @ \
                  self.link[2].homogenious_matrix @ \
                  self.link[3].homogenious_matrix @ \
                  self.link[4].homogenious_matrix
            R36 = np.matrix.transpose(H03[:3, :3]) @ rotation
            rotate, R_compare = Rm.inverse_kinematics_wrist(R36)
            self.link[5].DH_parameters[3] = rotate[0]
            self.link[6].DH_parameters[3] = rotate[1]
            self.link[7].DH_parameters[3] = rotate[2]
            for i in range(5, 9):
                self.link[i].homogenious_matrix = Rm.make_DH_matrix(self.link[i].DH_parameters)
            # H36 = self.link[5].homogenious_matrix @ \
            #       self.link[6].homogenious_matrix @ \
            #       self.link[7].homogenious_matrix @ \
            #       self.link[8].homogenious_matrix
            # H06 = H03 @ H36
            # p0 = np.array([pos[0],
            #                pos[1],
            #                pos[2],
            #                1])
            # TH6 = np.linalg.inv(H06).dot(p0)
            # 这一系列的计算步骤是为了确定机械臂末端执行器在给定目标位置时的关节角度
            # All flattened homogenious matrices are saved in a list
            for i in range(1, len(self.link)):
                # self.link[i].frame.setTransform(self.link[i].homogenious_matrix.flatten())
                matrix = self.link[i].homogenious_matrix.flatten()
                self.link[i].homogenious_matrix_new = matrix

    def update_window_new(self):
        from main_2 import glo_position
        self.calculate_dynamic_path(glo_position.copy())
        for i in range(1, len(self.link)):
            a1 = self.link[i].homogenious_matrix_new
            m1 = np.array_split(a1, 4)
            self.link[i].frame.setTransform(m1)

    def __str__(self):
        return self.print_text[self.iteration]

    def add_arm(self, parent_object, length):
        link = Link()
        link.frame = self.set_new_axis(parent_object.frame)
        link.arm = self.create_arm(length)
        link.arm.setParentItem(link.frame)
        link.arm_length = length
        link.arm.rotate(-90, 0, 1, 0)
        link.arm.translate(self.width / 2, -self.width / 2, self.width / 2)
        link.joint = self.create_joint()
        link.joint.setParentItem(link.frame)
        link.joint.translate(0, 0, -self.depth_cylinder / 2)
        return link

    def create_arm(self, length) -> gl.GLMeshItem:
        """Creates a box that is attached to a certain object"""
        size = (length, self.width, self.depth)
        vertices_arm, faces_arm = Rm.box(size)
        arm = gl.GLMeshItem(vertexes=vertices_arm, faces=faces_arm,
                            rawEdges=False, drawFaces=True, color=self.arm_color)
        self.view3D.addItem(arm)
        return arm

    def create_joint(self) -> gl.GLMeshItem:
        """Creates a joint that is attached to a certain object"""
        vertices_joint, faces_joint = Rm.cylinder(self.radius, self.depth_cylinder, self.segments)
        joint = gl.GLMeshItem(vertexes=vertices_joint, faces=faces_joint,
                              drawEdges=False, drawFaces=True, color=self.joint_color)
        self.view3D.addItem(joint)
        return joint

    def set_new_axis(self, parent_object) -> gl.GLAxisItem:
        """Adds axis to given object"""
        axis = gl.GLAxisItem(antialias=True, glOptions='opaque')
        axis.updateGLOptions({'glLineWidth': (3,)})
        axis.setParentItem(parent_object)
        self.view3D.addItem(axis)
        return axis

    def create_main_axis(self) -> gl.GLAxisItem:
        """Create the basis axis"""
        axis = gl.GLAxisItem()
        axis.updateGLOptions({'glLineWidth': (6,)})
        self.view3D.addItem(axis)
        return axis

    def show_path(self, frame):
        vertices_arm, faces_arm = Rm.box((0.05, 0.05, 0.05))
        box = gl.GLMeshItem(vertexes=vertices_arm, faces=faces_arm,
                            rawEdges=False, drawFaces=True, color=(1, 0.3, 0.4, 1))
        box.setParentItem(self.link[0].frame)
        m1 = np.array_split(frame.flatten(), 4)
        box.setTransform(m1)
        self.view3D.addItem(box)

    # def update_window(self):
    #     """Render new frame of 3D view"""
    #     if not self.ready:
    #         return
    #
    #         # When ready the robot is rendered and updated.
    #     for i in range(1, len(self.link)):
    #         a1 = self.link[i].hg_matrihomogenious_matrix_newx_list[self.iteration]
    #         m1 = np.array_split(a1, 4)
    #         self.link[i].frame.setTransform(m1)
    #     # increase the counter, such that next time we get the next point
    #     self.iteration += 1
    #     # if the counter arives at N, reset it to 0, to repeat the trajectory
    #     if self.iteration == self.N:
    #         self.iteration = 0
    #         self.cycle += 1

