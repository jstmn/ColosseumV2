import h5py
import json
import numpy as np
import os
import cv2  # OpenCV 추가
from tqdm import tqdm

def convert():
    data_dir = "new_demos/real_data/hyeonho_feb12/stack_cube_50"
    output_dir = "new_demos/real_data/processed/stack_cube_50"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_h5 = os.path.join(output_dir, "trajectory.h5")
    output_json = os.path.join(output_dir, "trajectory.json")
    
    # 1. 카메라 정보 로드
    cam_info_path = os.path.join(data_dir, "base_camera_info_color.json")
    with open(cam_info_path, "r") as f:
        cam_info = json.load(f)
        intrinsic = np.array(cam_info["K"]).reshape(3, 3)

    # 이미지 크기 변화에 따른 Intrinsic 스케일 계산
    # 원본 (640, 480) -> 목표 (128, 128)
    scale_x = 128 / 640
    scale_y = 128 / 480
    intrinsic_resized = intrinsic.copy()
    intrinsic_resized[0, 0] *= scale_x  # fx
    intrinsic_resized[0, 2] *= scale_x  # cx
    intrinsic_resized[1, 1] *= scale_y  # fy
    intrinsic_resized[1, 2] *= scale_y  # cy

    episodes = []

    with h5py.File(output_h5, "w") as out_f:
        for i in tqdm(range(50), desc="Converting to 128x128"):
            h5_name = os.path.join(data_dir, f"traj_{i}.h5")
            if not os.path.exists(h5_name):
                continue
            
            with h5py.File(h5_name, "r") as in_f:
                group = out_f.create_group(f"traj_{i}")
                
                qpos = in_f["joint_states_q"][:]
                qvel = in_f["joint_states_dq"][:]
                # 원본 이미지 로드
                raw_images = in_f["base__color_image"][:]
                
                # 2. 이미지 리사이징 (480, 640, 3) -> (128, 128, 3)
                resized_images = np.array([
                    cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA) 
                    for img in raw_images
                ])
                
                actions = qpos[1:] 
                group.create_dataset("actions", data=actions)
                
                obs = group.create_group("obs")
                agent = obs.create_group("agent")
                agent.create_dataset("qpos", data=qpos[:-1])
                agent.create_dataset("qvel", data=qvel[:-1])
                
                # 3. 리사이징된 이미지 저장
                sensor_data = obs.create_group("sensor_data/base_camera")
                sensor_data.create_dataset("rgb", data=resized_images[:-1])
                
                # 4. 수정된 Intrinsic 적용
                sensor_param = obs.create_group("sensor_param/base_camera")
                sensor_param.create_dataset("intrinsic_cv", data=np.repeat(intrinsic_resized[None], len(actions), axis=0))
                
                n_steps = len(actions)
                group.create_dataset("success", data=np.array([False]*(n_steps-1) + [True]))
                group.create_dataset("terminated", data=np.array([False]*(n_steps-1) + [True]))
                group.create_dataset("truncated", data=np.array([False]*n_steps))

                episodes.append({
                    "episode_id": i,
                    "success": True,
                    "elapsed_steps": n_steps
                })

    json_data = {
        "env_info": {
            "env_id": "StackCube-v1",
            "env_kwargs": {"obs_mode": "rgb", "control_mode": "pd_joint_pos"}
        },
        "episodes": episodes
    }
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✅ 128x128 변환 완료!")

if __name__ == "__main__":
    convert()