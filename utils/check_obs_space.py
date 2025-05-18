import os
import pickle
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env



"""
This file is used to check the observation space of the tasks in the LIBERO benchmark.

[INFO] Inspecting task: KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet
[OBS KEYS] ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'butter_1_pos', 'butter_1_quat', 'butter_1_to_robot0_eef_pos', 'butter_1_to_robot0_eef_quat', 'butter_2_pos', 'butter_2_quat', 'butter_2_to_robot0_eef_pos', 'butter_2_to_robot0_eef_quat', 'chocolate_pudding_1_pos', 'chocolate_pudding_1_quat', 'chocolate_pudding_1_to_robot0_eef_pos', 'chocolate_pudding_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']
==========================

[INFO] Inspecting task: KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet
[OBS KEYS] ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'butter_1_pos', 'butter_1_quat', 'butter_1_to_robot0_eef_pos', 'butter_1_to_robot0_eef_quat', 'butter_2_pos', 'butter_2_quat', 'butter_2_to_robot0_eef_pos', 'butter_2_to_robot0_eef_quat', 'chocolate_pudding_1_pos', 'chocolate_pudding_1_quat', 'chocolate_pudding_1_to_robot0_eef_pos', 'chocolate_pudding_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']
==========================

[INFO] Inspecting task: KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet
[OBS KEYS] ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']
==========================

[INFO] Inspecting task: KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet
[OBS KEYS] ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']
==========================

[INFO] Inspecting task: KITCHEN_SCENE1_put_the_black_bowl_on_the_plate
[OBS KEYS] ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']
"""


def log_obs_keys(benchmark_name):
    task_suite = benchmark.get_benchmark_dict()[benchmark_name]()
    for task in task_suite.tasks:
        task_name = task.name
        print(f"\n[INFO] Inspecting task: {task_name}")

        # Create environment
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)
        obs = env.reset()
        obs_keys = list(obs.keys())
        print(f"[OBS KEYS] {obs_keys}")


        EE_GEOM_NAMES = ["robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1"]
        # Get number of contacts
        n_contacts = env.sim.data.ncon
        print(f"Number of contacts: {n_contacts}")

        for i in range(n_contacts):
            contact = env.sim.data.contact[i]
            geom1_name = env.sim.model.geom_id2name(contact.geom1)
            geom2_name = env.sim.model.geom_id2name(contact.geom2)

            if geom1_name in EE_GEOM_NAMES or geom2_name in EE_GEOM_NAMES:
                print(f"EE Contact detected between {geom1_name} and {geom2_name}")



        print("==========================")
        exit(0)

if __name__ == "__main__":
    bm_name = "boss_44"
    log_obs_keys(bm_name)
