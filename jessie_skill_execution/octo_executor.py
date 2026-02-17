#!/usr/bin/env python3
import threading
import time
import numpy as np
import jax
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

from xarm.wrapper import XArmAPI
from octo.model.octo_model import OctoModel

from data_collection_msgs.msg import SkillExecutionRequest, ExecutionStatus
from data_collection_msgs.srv import CheckSkillPreconditions

class JessieOctoExecutor(Node):
    """Node that handles skill execution on Jessie using Octo.
    """
    # Indicates whether the node is running
    running = False

    # Octo model
    model = None

    # xArm instance
    arm = None

    # OpenCV bridge instance for converting image data between ROS and numpy
    cv_bridge = None

    # Indicates whether a new image has been received
    received_new_img = False

    # Last received image message
    latest_image_msg = None

    # Task to execute
    task = None

    # Indicates whether skill execution is ongoing
    skill_executing = False

    # Indicates whether the skill preconditions have already been checked
    preconditions_checked = False

    # Indicates whether a currently executed task has been completed
    task_completed = False

    # Subscriber for execution requests
    skill_execution_request_sub = None

    # Subscriber for cancelling the execution of an ongoing skill
    skill_execution_cancel_sub = None

    # Execution status publisher
    status_pub = None

    # Placeholder execution status message
    execution_status_msg = None

    # Service client for checking whether the preconditions of a skill are satisfied
    check_preconditions_proxy = None

    # Placeholder precondition checking request
    precondition_checking_request = None

    def __init__(self, name='jessie_octo_executor',
                 arm_ip='192.168.1.204',
                 image_topic: str='/camera/left_camera/color/image_raw',
                 skill_execution_request_topic: str='/jessie/execute_skill',
                 skill_execution_status_topic: str='/jessie/execution_status',
                 skill_execution_cancel_topic: str='/cancel_execution',
                 precondition_checking_srv: str='check_skill_preconditions',
                 closed_gripper_position: float=100.,
                 octo_model_path: str="hf://rail-berkeley/octo-small-1.5",
                 close_gripper_threshold = 1e-2,
                 open_gripper_threshold = 0.98):
        """
        Keyword arguments:
        arm_ip: str -- IP address of the xArm that executes skills (default 192.168.1.204)
        image_topic: str -- Topic on which to receive images (default /left_camera/image_raw)
        skill_execution_request_topic: str -- Topic on which to receive skill execution requests (default /jessie/execute_skill)
        skill_execution_status_topic: str -- Topic on which to send execution status messages (default /jessie/execution_status)
        skill_execution_cancel_topic: str -- Topic on which to send requests for cancelling the execution (default /cancel_execution)
        precondition_checking_srv: str -- Name of a service for checking skill preconditions (default check_preconditions)
        closed_gripper_position: float -- Position at which the gripper is considered closed (default 100.)
        octo_model_path: str -- Location from which to load the Octo model (default hf://rail-berkeley/octo-small-1.5)
        close_gripper_threshold: float -- Threshold for closing the gripper
        open_gripper_threshold: float -- Threshold for opening the gripper

        """
        super().__init__(name)

        self.model = None
        self.arm = None
        self.cv_bridge = CvBridge()

        self.task = None
        self.latest_image_msg = None
        self.received_new_img = False
        self.skill_executing = False
        self.preconditions_checked = False
        self.task_completed = False

        self.close_gripper_threshold = close_gripper_threshold
        self.open_gripper_threshold = open_gripper_threshold
        self.past_gripper_values = deque(maxlen=5)

        self.load_octo_model(octo_model_path)
        self.set_up_arm(arm_ip)

        self.closed_gripper_position = closed_gripper_position
        self.execution_status_msg = ExecutionStatus()
        self.image_sub = self.create_subscription(Image,
                                                  image_topic,
                                                  self.image_cb,
                                                  10)
        self.skill_execution_request_sub = self.create_subscription(SkillExecutionRequest,
                                                                    skill_execution_request_topic,
                                                                    self.skill_execution_request_cb,
                                                                    10)
        self.skill_execution_cancel_sub = self.create_subscription(Empty,
                                                                   skill_execution_cancel_topic,
                                                                   self.skill_execution_cancel_cb,
                                                                   10)
        self.skill_execution_status_pub = self.create_publisher(ExecutionStatus,
                                                                skill_execution_status_topic,
                                                                10)
        self.check_preconditions_proxy = self.create_client(CheckSkillPreconditions, precondition_checking_srv)
        while not self.check_preconditions_proxy.wait_for_service(timeout_sec=1.):
            self.get_logger().info(f'[octo_executor] Service {precondition_checking_srv} not available, waiting...')
        self.precondition_checking_request = CheckSkillPreconditions.Request()

        self.running = True

    def load_octo_model(self, model_path: str):
        """Sets self.model with an Octo model loaded from model_path.

        Keyword arguments:
        model_path: str -- Location from which to load the Octo model

        """
        self.model = OctoModel.load_pretrained(model_path)

    def set_up_arm(self, arm_ip: str):
        """Initialises an xArm instance at the given IP address.
        The arm is set up to use position control.

        Keyword arguments:
        arm_ip: str --- IP address of an xArm

        """
        self.arm = XArmAPI(arm_ip, do_not_open=True)
        self.arm.register_error_warn_changed_callback(self.handle_err_warn_changed)
        self.arm.connect()

        self.arm.motion_enable(enable=True)

        # set position control mode
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.arm.set_gripper_mode(mode=0)
        self.arm.set_gripper_enable(enable=True)
        self.arm.set_gripper_speed(5000)

    def execute_skill(self):
        """Runs a single skill execution step by processing the latest received image.
        Returns immediately if no skill is being executed.
        """
        if not self.skill_executing:
            return

        self.get_logger().info(f'[octo_executor] Should check preconditions: {self.task.check_preconditions}')
        if self.task.check_preconditions and not self.preconditions_checked:
            self.get_logger().info(f'[octo_executor] Checking preconditions for task {self.task.skill_description}')
            self.precondition_checking_request.task_description = self.task.skill_description
            self.precondition_checking_request.current_image = self.latest_image_msg

            future = self.check_preconditions_proxy.call_async(self.precondition_checking_request)
            self.execution_status_msg.skill_name = self.task.skill_description
            self.execution_status_msg.status = ExecutionStatus.PREPARING
            self.execution_status_msg.feedback = 'Checking preconditions'
            while not future.done():
                self.skill_execution_status_pub.publish(self.execution_status_msg)
                time.sleep(0.5)
            precondition_check_response = future.result()

            self.get_logger().info(f'[octo_executor] Preconditions satisfied for task {self.task.skill_description}: {precondition_check_response.preconditions_satisfied}')
            if precondition_check_response.info:
                self.get_logger().info(f'[octo_executor] Preconditions checking info: {precondition_check_response.info}')

            if not precondition_check_response.preconditions_satisfied:
                self.get_logger().error('[octo_executor] Preconditions not satisfied; not executing')

                self.execution_status_msg.skill_name = self.task.skill_description
                self.execution_status_msg.status = ExecutionStatus.STOPPING
                self.execution_status_msg.feedback = f'Not executing task {self.task.skill_description} as the preconditions are not satisfied. {precondition_check_response.info}'
                self.skill_execution_status_pub.publish(self.execution_status_msg)

                self.stop_skill_execution()
                return

            self.execution_status_msg.skill_name = self.task.skill_description
            self.execution_status_msg.status = ExecutionStatus.EXECUTING
            self.execution_status_msg.feedback = f'The preconditions of task {self.task.skill_description} are satisfied; starting execution'
            self.skill_execution_status_pub.publish(self.execution_status_msg)
            self.preconditions_checked = True

        img = self.cv_bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding='passthrough')
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = img[80:561, :]
        img = cv2.resize(img, (256,256))
        img = img[np.newaxis, np.newaxis, ...]

        observation = {"image_primary": img,
                       "timestep_pad_mask": np.array([[True]])}
        task = self.model.create_tasks(texts=[self.task.skill_description])
        action = self.model.sample_actions(
            observation, 
            task, 
            unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"], 
            rng=jax.random.PRNGKey(0)
        )

        # The axes flipping is essentially a conversion from the camera frame to base_link
        x_velocity = float(action[0][0][2])
        y_velocity = float(action[0][0][0])
        z_velocity = -float(action[0][0][1])

        # multiplying the commands by 1000 because set_position uses mm as input units
        self.arm.set_position(x=x_velocity * 700,
                              y=y_velocity * 700,
                              z=z_velocity * 700,
                              roll=0, pitch=0, yaw=0, relative=True)

        gripper_action = float(action[0][0][-1])
        arm_vel_vector = np.array([x_velocity, y_velocity, z_velocity])

        self.past_gripper_values.append(gripper_action)
        close_gripper_threshold_reached = sum([int(abs(x) < self.close_gripper_threshold) for x in self.past_gripper_values]) >= 3

        # TODO: see how to integrate this / whether it is really needed (would it make more sense
        # to interpret what we need to do with the gripper from the natural language command?)
        # open_gripper_threshold_reached = sum([int(abs(x) > self.open_gripper_threshold) for x in self.past_gripper_values]) >= 3

        if close_gripper_threshold_reached or np.linalg.norm(arm_vel_vector) < 0.005:
            self.arm.set_gripper_position(300, wait=True)
            print(f'Completed task {self.task.skill_description}')

            self.execution_status_msg.skill_name = self.task.skill_description
            self.execution_status_msg.status = ExecutionStatus.COMPLETED
            self.execution_status_msg.feedback = f'Task {self.task.skill_description} completed'
            self.skill_execution_status_pub.publish(self.execution_status_msg)
            self.past_gripper_values.clear()

            self.stop_skill_execution()
        else:
            self.execution_status_msg.skill_name = self.task.skill_description
            self.execution_status_msg.status = ExecutionStatus.EXECUTING
            self.execution_status_msg.feedback = f'Arm velocity: {str(arm_vel_vector)}'
            self.skill_execution_status_pub.publish(self.execution_status_msg)

    def skill_execution_request_cb(self, skill_execution_request_msg: SkillExecutionRequest):
        """Registers a skill execution request.
        Ignores the request if:
        * the request is not for a general language conditioned model
        * the requested model is not octo
        * another skill is already being executed.

        Keyword arguments:
        skill_execution_request_msg: data_collection_msgs.msg.SkillExecutionRequest -- Execution request message

        """
        if skill_execution_request_msg.skill_type != SkillExecutionRequest.GENERAL_LANGUAGE_CONDITIONED:
            return

        if skill_execution_request_msg.model_name != 'octo':
            self.get_logger().info(f'[octo_executor] Ignoring skill execution request; model "{skill_execution_request_msg.model_name}" specified, but "octo" expected')

        if self.skill_executing:
            self.get_logger().info('[octo_executor] Ignoring skill execution request as another skill is already executing')

        self.task = skill_execution_request_msg
        self.get_logger().info(f'[octo_executor] Received request for task {skill_execution_request_msg.skill_description}')
        self.skill_executing = True

        self.execution_status_msg.skill_name = skill_execution_request_msg.skill_description
        self.execution_status_msg.status = ExecutionStatus.PREPARING
        self.skill_execution_status_pub.publish(self.execution_status_msg)

    def image_cb(self, image_msg: Image):
        """Registers the message as the latest received image message.

        Keyword arguments:
        image_msg: sensor_msgs.msg.Image

        """
        self.latest_image_msg = image_msg
        self.received_new_img = True

    def skill_execution_cancel_cb(self, msg: Empty):
        """Cancels an ongoing skill execution (if any).

        Keyword arguments:
        msg: std_msgs.msg.Empty

        """
        if self.skill_executing:
            self.get_logger().error(f'[octo_executor] Received a cancellation request while executing task {self.task.skill_description}')
            self.execution_status_msg.skill_name = self.task.skill_description
            self.execution_status_msg.status = ExecutionStatus.STOPPING
            self.execution_status_msg.feedback = f'Stopping the execution of task {self.task.skill_description} as a cancellation request was received'
            self.skill_execution_status_pub.publish(self.execution_status_msg)
            self.stop_skill_execution()
            self.get_logger().error('[octo_executor] Execution stopped')
        else:
            self.get_logger().error('[octo_executor] No skill is currently executing; nothing to cancel')

    def stop_skill_execution(self):
        """Resets all local variables related to the skill execution process.
        """
        self.skill_executing = False
        self.past_gripper_values.clear()
        self.task = SkillExecutionRequest()

        self.precondition_checking_request.task_description = ''
        self.precondition_checking_request.current_image = Image()
        self.preconditions_checked = False

        self.execution_status_msg.skill_name = ''
        self.execution_status_msg.status = ExecutionStatus.UNKNOWN
        self.execution_status_msg.feedback = ''

    def handle_err_warn_changed(self, item):
        """Prints an xArm error / warning.
        """
        print('ErrorCode: {}, WarnCode: {}'.format(item['error_code'], item['warn_code']))


def main(args=None):
    rclpy.init(args=args)
    octo_executor = JessieOctoExecutor()

    rate = octo_executor.create_rate(5, octo_executor.get_clock())
    rate_thread = threading.Thread(target=rclpy.spin, args=(octo_executor,), daemon=True)
    rate_thread.start()

    try:
        while rclpy.ok():
            if octo_executor.skill_executing and octo_executor.received_new_img:
                octo_executor.execute_skill()
                octo_executor.received_new_img = False
            rate.sleep()
    except Exception as exc:
        print(str(exc))

    print('Destroying octo execution node')
    octo_executor.running = False
    time.sleep(0.5)
    octo_executor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
