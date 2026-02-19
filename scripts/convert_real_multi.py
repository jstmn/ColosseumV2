import h5py
import json
import numpy as np
import os
import cv2
from tqdm import tqdm

def get_intrinsic_resized(cam_info, target_size=(128, 128), original_size=(640, 480)):
    """Intrinsic Matrix를 목표 크기에 맞춰 스케일링하는 헬퍼 함수"""
    intrinsic = np.array(cam_info["K"]).reshape(3, 3)
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    
    intrinsic_resized = intrinsic.copy()
    intrinsic_resized[0, 0] *= scale_x  # fx
    intrinsic_resized[0, 2] *= scale_x  # cx
    intrinsic_resized[1, 1] *= scale_y  # fy
    intrinsic_resized[1, 2] *= scale_y  # cy
    return intrinsic_resized

def convert():
    # --- 설정 영역 ---
    data_dir = "new_demos_all/hyeonho_feb17_all_cameras/raise_cube_50_3camera"
    output_dir = "new_demos_all/real_data/processed/2camera/raise_cube_50"
    
    # 처리하고 싶은 카메라 목록을 여기에 넣으세요
    cameras = ["eih_camera", "base_camera"] 
    target_size = (128, 128)
    # -----------------

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_h5 = os.path.join(output_dir, "trajectory.h5")
    output_json = os.path.join(output_dir, "trajectory.json")
    
    cam_prefix = {
        "base_camera": "base",
        "eih_camera": "eih",
        "north_camera": "north",
    }

    # 핵심 수정: cam_configs 딕셔너리 초기화
    cam_configs = {}

    for cam in cameras:
        prefix = cam_prefix[cam]
        path = os.path.join(data_dir, f"{prefix}_camera_info_color.json")
        
        if not os.path.exists(path):
            print(f"⚠️ 경고: {path} 파일을 찾을 수 없어 {cam}는 건너뜁니다.")
            continue

        with open(path, "r") as f:
            info = json.load(f)
        
        cam_configs[cam] = {
            "intrinsic": get_intrinsic_resized(info, target_size=target_size),
            "raw_key": f"{prefix}__color_image"
        }

    episodes = []

    with h5py.File(output_h5, "w") as out_f:
        # 실제 파일 개수에 맞춰 range 조절 (예: 50개)
        for i in tqdm(range(50), desc=f"Converting {target_size[0]}x{target_size[1]}"):
            h5_name = os.path.join(data_dir, f"traj_{i}.h5")
            if not os.path.exists(h5_name): continue
            
            with h5py.File(h5_name, "r") as in_f:
                group = out_f.create_group(f"traj_{i}")
                qpos = in_f["joint_states_q"][:]
                qvel = in_f["joint_states_dq"][:]
                actions = qpos[1:] 
                
                group.create_dataset("actions", data=actions)
                obs = group.create_group("obs")
                agent = obs.create_group("agent")
                agent.create_dataset("qpos", data=qpos[:-1])
                agent.create_dataset("qvel", data=qvel[:-1])

                # 모든 카메라 데이터 처리
                for cam_name, config in cam_configs.items():
                    raw_images = in_f[config["raw_key"]][:]
                    resized = np.array([
                        cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) 
                        for img in raw_images
                    ])
                    
                    # sensor_data 저장
                    s_data = obs.create_group(f"sensor_data/{cam_name}")
                    s_data.create_dataset("rgb", data=resized[:-1])
                    
                    # sensor_param 저장
                    s_param = obs.create_group(f"sensor_param/{cam_name}")
                    s_param.create_dataset("intrinsic_cv", data=np.repeat(config["intrinsic"][None], len(actions), axis=0))
                
                n_steps = len(actions)
                group.create_dataset("success", data=np.array([False]*(n_steps-1) + [True]))
                group.create_dataset("terminated", data=np.array([False]*(n_steps-1) + [True]))
                group.create_dataset("truncated", data=np.array([False]*n_steps))

                episodes.append({"episode_id": i, "success": True, "elapsed_steps": n_steps})

    # JSON 결과 저장
    json_data = {
        "env_info": {
            "env_id": "RaiseCube-v1",
            "env_kwargs": {"obs_mode": "rgb", "control_mode": "pd_joint_pos", "cameras": list(cam_configs.keys())}
        },
        "episodes": episodes
    }
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n✅ 변환 완료! 출력 경로: {output_dir}")

if __name__ == "__main__":
    convert()