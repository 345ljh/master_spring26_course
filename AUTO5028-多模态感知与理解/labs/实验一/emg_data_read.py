import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def emg_get_prompt(subject_name):
    session_files = []  # 每名受试者5个文件
    subject_file = f"EMG-HDF5-20260409/{subject_name}"
    # 遍历文件夹中的所有文件
    for filename in os.listdir(subject_file):
        if filename.endswith(".h5"):
            session_files.append(filename)

    prompts = {}
    for session_file in session_files:
        with h5py.File(os.path.join(subject_file, session_file), 'r') as f:
            # emg1_2khz_adc下为samples行数据, 每行格式: ([ch1, ch2, ..., ch16], sd_frame_id, timestamp)
            data = np.array([line[0] for line in f['emg1_2khz_adc']])  # (samples, 16)
            timestamp = np.array([line[2] for line in f['emg1_2khz_adc']])  # (samples), 10位时间戳
            # 标记与时间
            prompt_name = f['prompts']['names'][:]
            prompt_time = f['prompts']['times'][:]
            # 在timestamp中找到prompt_time对应的索引, 转换为[prompt_id, sample_idx]的array
            # prompt文字以字节存储, 因此基于[-4]进行分类: 左/右/前/后 - 0xa6/0xb3/0x8d/0x8e
            prompt = np.zeros((0, 2)).astype(int)
            classify = [0xa6, 0xb3, 0x8d, 0x8e]
            for i in range(len(prompt_name)):
                idx = np.argmin(np.abs(timestamp - prompt_time[i]))
                cla = classify.index(prompt_name[i][-4])
                prompt = np.append(prompt, np.array([[cla, idx]]), axis=0)

            prompts[session_file] = prompt
    return prompts

if __name__ == '__main__':
    subject_name = 'T005'
    session_prompts = emg_get_prompt(subject_name)
 