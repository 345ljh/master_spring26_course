import h5py
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import pickle

fs = 2000

# 对每个文件进行数据提取与预处理, 并生成文件
# 格式: 
# {
#     [  # file1
#         features: ndarray(samples, 80), label: (samples),  # segment1
#         ...
#     ],
#     ...
# }
def emg_preprocess_feature(subject_name, session_prompts):
    subject_file = f"EMG-HDF5-20260409/{subject_name}"
    features = []
    # 遍历文件夹中的所有文件
    for session_name in session_prompts:
        if session_name.endswith(".h5"):
            session_features = np.zeros((0, 80))
            session_labels = np.zeros(0)
            with h5py.File(os.path.join(subject_file, session_name), 'r') as f:
                raw_emg = np.array([line[0] for line in f['emg1_2khz_adc']])  # (samples, 16)
                for prompt_line in session_prompts[session_name]:
                    raw_emg_segment = raw_emg[prompt_line[1]-2000:prompt_line[1]+2000, :]
                    # plt.plot(raw_emg_segment[:,0])

                    # 滤波: 20~500Hz带通 + 50Hz陷波
                    b, a = signal.butter(4, [20*2/fs, 500*2/fs], btype='band')
                    filted_emg_segment = signal.filtfilt(b, a, raw_emg_segment, axis=0)
                    b, a = signal.iirnotch(50*2/fs, 30)
                    filted_emg_segment = signal.filtfilt(b, a, filted_emg_segment, axis=0)

                    # plt.plot(filted_emg_segment[:,0])
                    # plt.show()

                    # z-score
                    preprocessed_emg_segment = (filted_emg_segment - np.mean(filted_emg_segment, axis=0)) / np.std(filted_emg_segment, axis=0)

                    # 提取5维特征
                    mav = np.mean(np.abs(preprocessed_emg_segment), axis=0)
                    rms = np.sqrt(np.mean(np.square(preprocessed_emg_segment), axis=0))
                    wl = np.sum(np.sqrt(np.diff(preprocessed_emg_segment, axis=0) ** 2 + (1 / fs) ** 2), axis=0)
                    zc = np.sum(np.diff(np.sign(preprocessed_emg_segment), axis=0) != 0, axis=0)
                    ss = np.sum(np.diff(np.sign(np.diff(preprocessed_emg_segment, axis=0)), axis=0) != 0, axis=0)

                    segment_feature = np.concatenate((mav, rms, wl, zc, ss), axis=0).reshape(1, -1)
                    session_features = np.concatenate((session_features, segment_feature), axis=0)
                    session_labels = np.append(session_labels, prompt_line[0])
            features.append([session_features, session_labels])
            print(f"{session_name} done")
    save_path = f"features/{subject_name}/features.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)


def emg_preprocess(subject_name, session_prompts):
    subject_file = f"EMG-HDF5-20260409/{subject_name}"
    features = []
    # 遍历文件夹中的所有文件
    for session_name in session_prompts:
        if session_name.endswith(".h5"):
            session_features = np.zeros((0, 4000, 16))
            session_labels = np.zeros(0)
            with h5py.File(os.path.join(subject_file, session_name), 'r') as f:
                raw_emg = np.array([line[0] for line in f['emg1_2khz_adc']])  # (samples, 16)
                for prompt_line in session_prompts[session_name]:
                    raw_emg_segment = raw_emg[prompt_line[1]-2000:prompt_line[1]+2000, :]
                    # plt.plot(raw_emg_segment[:,0])

                    # 滤波: 20~500Hz带通 + 50Hz陷波
                    b, a = signal.butter(4, [20*2/fs, 500*2/fs], btype='band')
                    filted_emg_segment = signal.filtfilt(b, a, raw_emg_segment, axis=0)
                    b, a = signal.iirnotch(50*2/fs, 30)
                    filted_emg_segment = signal.filtfilt(b, a, filted_emg_segment, axis=0)

                    # 直接保存原数据
                    session_features = np.concatenate((session_features, filted_emg_segment.reshape(1, -1, 16)), axis=0)
                    session_labels = np.append(session_labels, prompt_line[0])
            features.append([session_features, session_labels])
            print(f"{session_name} done")
    save_path = f"features/{subject_name}/emgseries.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)


if __name__ == '__main__':
    import emg_data_read
    subject_name = 'T005'
    session_prompts = emg_data_read.emg_get_prompt(subject_name)
    emg_preprocess(subject_name, session_prompts)
    emg_preprocess_feature(subject_name, session_prompts)
 