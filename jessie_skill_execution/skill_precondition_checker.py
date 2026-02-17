#!/usr/bin/env python3
from typing import List
import threading
import time
import base64
import yaml
import pydantic
import cv2
from cv_bridge import CvBridge

from ollama import chat
from ollama import ChatResponse

import rclpy
from rclpy.node import Node

from data_collection_msgs.srv import CheckSkillPreconditions

class ManipulationTaskSymbols(pydantic.BaseModel):
    task: str
    obj: str

class SkillPreconditionChecker(Node):
    """Node for checking whether the preconditions of skills are satisfied.
    """
    # A dictionary with skill descriptions; the format of the 
    # {"[skill_1_description_1, ..., skill_1_description_n]": [precondition_1, ..., precondition_k],
    # ... [skill_m_description_1, ..., skill_m_description_n]": [precondition_1, ..., precondition_k]}
    # namely the key is a string list of skill descriptions that have the same preconditions;
    # this is particularly so that synonymous skill descriptions can be accounted for
    skill_precondition_map = None

    # Name of a vision-language model that is used for precondition checking
    vlm_name = ''

    # ROS server for checking the preconditions of a skill
    check_skill_preconditions_service = None

    def __init__(self, name="skill_precondition_checker",
                 vlm_name='llama3.2-vision',
                 precondition_file_path: str=""):
        """
        Keyword arguments:
        precondition_file_path: str -- Path to a YAML file containing skill precondition descriptions in natural language.
                                       The format of the file should be
                                           "[skill_1_description_1, ..., skill_1_description_n]": ["precondition_1", ..., "precondition_k"]}
                                           ...
                                           "[skill_m_description_1, ..., skill_m_description_n]": ["precondition_1", ..., "precondition_k"]}
                                       namely the key is a string list of skill descriptions that have the same preconditions;
                                       this is particularly so that synonymous skill descriptions can be accounted for.

        """
        super().__init__(name)
        self.declare_parameter('precondition_file_path', '')
        precondition_file_path = self.get_parameter_or('precondition_file_path',
                                                       rclpy.Parameter('precondition_file_path', rclpy.Parameter.Type.STRING, '')).value

        self.declare_parameter('vlm_name', '')
        vlm_name = self.get_parameter_or('vlm_name',
                                         rclpy.Parameter('vlm_name', rclpy.Parameter.Type.STRING, 'llava')).value

        self.get_logger().info(f'Precondition file path: {precondition_file_path}')
        self.get_logger().info(f'VLM name: {vlm_name}')

        self.vlm_name = vlm_name
        self.load_preconditions(precondition_file_path)
        self.cv_bridge = CvBridge()
        self.check_skill_preconditions_service = self.create_service(CheckSkillPreconditions,
                                                                     'check_skill_preconditions',
                                                                     self.check_preconditions)

    def check_preconditions(self, request: CheckSkillPreconditions.Request,
                            response: CheckSkillPreconditions.Response) -> CheckSkillPreconditions.Response:
        """Service callback returning precondition check results, and an optional description of
        the result in case of issues with the precondition checking (e.g. the skill is not recognised).

        Keyword arguments:
        request: data_collection_msgs.srv.CheckSkillPreconditions.Request -- Request containing a description of the task to be executed
                                                                   and an image of the current scene.
        response: data_collection_msgs.srv.CheckSkillPreconditions.Response -- Response containing a flag describing whether the preconditions are satisfied
                                                                     and an optional description of the results.

        """
        try:
            # we extract the symbols from the task description and replace the
            # variables in the precondition descriptions with the extracted symbols
            symbols = self.ground_task_symbols(request.task_description)
            preconditions = self.get_skill_preconditions(symbols.task)
            self.get_logger().info(f'[skill_precondition_checker] Task: {symbols.task}')
            for i, x in enumerate(preconditions):
                preconditions[i] = x.replace('X', symbols.obj)
                self.get_logger().info(f'[skill_precondition_checker] Precondition {i+1}: {preconditions[i]}')

            # we convert the image to a byte array, suitable for processing by an Ollama VLM
            image = self.cv_bridge.imgmsg_to_cv2(request.current_image, desired_encoding='passthrough')
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(image, (256,256))
            image = base64.b64encode(cv2.imencode('.jpg', image)[1].tobytes()).decode()

            # we check whether the preconditions are satisfied in the current scene image;
            # note: this is a time-consuming operation!
            preconditions_satisfied = {precondition: False for precondition in preconditions}
            for precondition in preconditions:
                preconditions_satisfied[precondition] = self.check_precondition(precondition=precondition,
                                                                                image=image)

            # we set the result based on the precondition checks
            self.get_logger().info(f'Preconditions satisfied checks: {preconditions_satisfied}')
            all_preconditions_satisfied = sum([int(x) for _,x in preconditions_satisfied.items()]) == len(preconditions)
            if all_preconditions_satisfied:
                response.preconditions_satisfied = True
            else:
                response.info = 'The following preconditions were not satisfied:'
                for precondition, satisfied in preconditions_satisfied.items():
                    if not satisfied:
                        response.info += f'--- {precondition}'
            return response
        except ValueError as exc:
            self.get_logger().error(f'[skill_precondition_checker] {str(exc)}')

            # we report that the precondition are satisfied in the case of an exception so that errors
            # in this component (e.g. with the symbol extraction) don't escalate to the execution component
            response.preconditions_satisfied = True
            return response

    def ground_task_symbols(self, task_description: str) -> ManipulationTaskSymbols:
        """Attempts to ground symbols from the given task description.
        If the grounding is correct, the returned dictionary will have the format
        {'task': <skill to be executed>, 'obj': <object to be manipulated during the skill execution>}

        Keyword arguments:
        task_description: str -- Natural language description of the skill to be executed

        """
        symbol_grounder_response: ChatResponse = chat(model=self.vlm_name,
            format=ManipulationTaskSymbols.model_json_schema(),
            messages=[
            {
                'role': 'user',
                'content': f'Suppose you are a robot and you have been given the task "{task_description}". What task do you have to perform and which object do you have to manipulate? Print the output as a dictionary {{task: X, obj: Y}}, where you replace X with the name of the task and Y with the object to be manipulated. The dictionary should be your only output.',
            }
        ])

        symbols = ManipulationTaskSymbols.model_validate_json(symbol_grounder_response.message.content)
        return symbols

    def check_precondition(self, precondition: str, image: bytes) -> bool:
        """Returns False if the precondition is not satisfied in the given image;
        returns True otherwise (also in the case of output parsing problems).

        Keyword arguments:
        precondition: str -- A natural language description of a precondition
        image: bytes -- Image in which the precondition is checked

        """
        precondition_checker_response: ChatResponse = chat(model=self.vlm_name,
            messages=[
            {
                'role': 'user',
                'content': f'Is the condition {precondition} satisfied in this image? Answer with either yes or no (without punctuation). That should be your only output.',
                'images': [image]
            }
        ])

        response = precondition_checker_response.message.content.removesuffix('.').strip().lower()
        self.get_logger().info(f'Precondition checking output: {response}')
        if response == 'no':
            return False
        return True

    def get_skill_preconditions(self, skill: str) -> List[str]:
        """Returns the preconditions (from self.skill_precondition_map) corresponding to
        the given skill. Raises a ValueError if the given skill is not found in
        self.skill_precondition_map.

        Keyword arguments:
        skill: str -- Description of the skill whose preconditions should be retrieved

        """
        skill_names = list(self.skill_precondition_map.keys())
        skill_idx = 0
        for i, x in enumerate(skill_names):
            if skill in x:
                skill_idx = i
                break
        if skill_idx != -1:
            return list(self.skill_precondition_map[skill_names[skill_idx]])
        raise ValueError(f'No preconditions known for skill {skill}')

    def load_preconditions(self, precondition_file_path: str) -> None:
        """Loads skill precondition descriptions from the given file
        and stores them in safe.skill_precondition_map. Sets
        self.skill_precondition_map to an empty dictionary if the given
        file is not in YAML format (allowed extension .yaml and .yml).

        Keyword arguments:
        precondition_file_path: str -- Path to a file containing skill precondition descriptions

        """
        try:
            file_extension = precondition_file_path.split('.')[-1]
            if file_extension not in ['yaml', 'yml']:
                raise ValueError('File not in YAML format')

            with open(precondition_file_path, 'r') as f:
                self.skill_precondition_map = yaml.load(f, Loader=yaml.SafeLoader)
        except Exception as exc:
            self.get_logger().error(f'[skill_precondition_checker] {str(exc)}')
            self.skill_precondition_map = dict()

def main(args=None):
    rclpy.init(args=args)
    precondition_checker = SkillPreconditionChecker()

    rate = precondition_checker.create_rate(5, precondition_checker.get_clock())
    rate_thread = threading.Thread(target=rclpy.spin, args=(precondition_checker,), daemon=True)
    rate_thread.start()

    try:
        while rclpy.ok():
            rate.sleep()
    except:
        pass

    print('Destroying skill precondition checker node')
    precondition_checker.running = False
    time.sleep(0.5)
    precondition_checker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
