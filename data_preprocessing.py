import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import pickle
import json
from pathlib import Path
import warnings
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
warnings.filterwarnings('ignore')

class AudioPreprocessor:

    def __init__(self, data_path="C:\\Users\\86198\\Desktop\\我的毕设代码\\pythonProject1\\data", target_sr=16000, frame_length_ms=25,
                 frame_shift_ms=10, preemphasis_coeff=0.97, silence_threshold=0.01):
        self.data_path = data_path
        self.target_sr = target_sr
        self.frame_length = int(target_sr * frame_length_ms / 1000)
        self.frame_shift = int(target_sr * frame_shift_ms / 1000)
        self.preemphasis_coeff = preemphasis_coeff
        self.silence_threshold = silence_threshold

        self.output_dir = "preprocessed_data"
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_logging()

        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'class_counts': {},
            'total_duration': 0,
            'avg_duration': 0,
            'class_durations': {}
        }

        self.class_to_label = {}
        self.label_to_class = {}

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def preemphasis_filter(self, signal_data):
        filtered_signal = np.append(signal_data[0], signal_data[1:] - self.preemphasis_coeff * signal_data[:-1])
        return filtered_signal

    def frame_signal(self, signal_data, window_type='hamming'):
        signal_length = len(signal_data)

        if signal_length <= self.frame_length:
            num_frames = 1
        else:
            num_frames = 1 + int(np.ceil((1.0 * signal_length - self.frame_length) / self.frame_shift))

        pad_signal_length = num_frames * self.frame_shift + self.frame_length
        z = np.zeros((pad_signal_length - signal_length,))
        pad_signal = np.append(signal_data, z)

        indices = np.tile(np.arange(0, self.frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * self.frame_shift, self.frame_shift),
                          (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        if window_type == 'hamming':
            window = np.hamming(self.frame_length)
        elif window_type == 'hanning':
            window = np.hanning(self.frame_length)
        elif window_type == 'blackman':
            window = np.blackman(self.frame_length)
        else:
            window = np.ones(self.frame_length)

        windowed_frames = frames * window

        return windowed_frames

    def amplitude_normalization(self, frames):
        normalized_frames = np.zeros_like(frames)

        for i, frame in enumerate(frames):
            max_val = np.max(np.abs(frame))
            if max_val > 0:
                normalized_frames[i] = frame / max_val
            else:
                normalized_frames[i] = frame

        return normalized_frames

    def detect_silence(self, frames, energy_threshold=None, zcr_threshold=None):
        if energy_threshold is None:
            energy_threshold = self.silence_threshold
        if zcr_threshold is None:
            zcr_threshold = 0.3

        energy = np.sum(frames ** 2, axis=1)
        energy = energy / np.max(energy)

        zcr = np.zeros(len(frames))
        for i, frame in enumerate(frames):
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame))

        voice_frames = np.where((energy > energy_threshold) | (zcr < zcr_threshold))[0]

        return voice_frames

    def process_audio_file(self, file_path, class_name):
        try:
            audio_data, original_sr = librosa.load(file_path, sr=None)
            duration = len(audio_data) / original_sr

            if original_sr != self.target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.target_sr)

            preemphasized = self.preemphasis_filter(audio_data)

            frames = self.frame_signal(preemphasized)

            normalized_frames = self.amplitude_normalization(frames)

            voice_frame_indices = self.detect_silence(normalized_frames)

            if len(voice_frame_indices) > 0:
                voice_frames = normalized_frames[voice_frame_indices]
            else:
                voice_frames = normalized_frames
                self.logger.warning(f"所有帧都被检测为静音: {file_path}")

            processed_data = {
                'frames': voice_frames,
                'original_duration': duration,
                'processed_frames': len(voice_frames),
                'original_sr': original_sr,
                'target_sr': self.target_sr,
                'class_name': class_name,
                'file_path': file_path
            }

            return processed_data

        except Exception as e:
            self.logger.error(f"处理文件失败 {file_path}: {str(e)}")
            return None

    def scan_data_directory(self):
        file_list = []
        class_names = []

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")

        for item in os.listdir(self.data_path):
            class_dir = os.path.join(self.data_path, item)
            if os.path.isdir(class_dir):
                class_names.append(item)

                wav_files = []
                for file in os.listdir(class_dir):
                    if file.lower().endswith('.wav'):
                        file_path = os.path.join(class_dir, file)
                        wav_files.append(file_path)

                file_list.extend([(f, item) for f in wav_files])
                self.stats['class_counts'][item] = len(wav_files)

        class_names.sort()
        for i, class_name in enumerate(class_names):
            self.class_to_label[class_name] = i
            self.label_to_class[i] = class_name

        self.stats['total_files'] = len(file_list)
        self.logger.info(f"发现 {len(class_names)} 个类别，共 {len(file_list)} 个音频文件")
        self.logger.info(f"类别映射: {self.class_to_label}")

        return file_list

    def process_all_files(self):
        file_list = self.scan_data_directory()

        if not file_list:
            raise ValueError("未找到任何音频文件")

        all_processed_data = []
        class_durations = {class_name: [] for class_name in self.class_to_label.keys()}

        self.logger.info("开始处理音频文件...")

        for file_path, class_name in tqdm(file_list, desc="处理音频文件"):
            processed_data = self.process_audio_file(file_path, class_name)

            if processed_data is not None:
                processed_data['label'] = self.class_to_label[class_name]
                all_processed_data.append(processed_data)

                class_durations[class_name].append(processed_data['original_duration'])
                self.stats['total_duration'] += processed_data['original_duration']
                self.stats['processed_files'] += 1
            else:
                self.stats['failed_files'] += 1

        self._calculate_statistics(class_durations)

        self.logger.info(f"处理完成: 成功 {self.stats['processed_files']} 个，失败 {self.stats['failed_files']} 个")

        return all_processed_data

    def _calculate_statistics(self, class_durations):
        if self.stats['processed_files'] > 0:
            self.stats['avg_duration'] = self.stats['total_duration'] / self.stats['processed_files']

        for class_name, durations in class_durations.items():
            if durations:
                self.stats['class_durations'][class_name] = {
                    'count': len(durations),
                    'total_duration': sum(durations),
                    'avg_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations)
                }

    def save_processed_data(self, processed_data):
        pickle_file = os.path.join(self.output_dir, 'preprocessed_data.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(processed_data, f)
        self.logger.info(f"数据已保存到: {pickle_file}")

        mapping_file = os.path.join(self.output_dir, 'class_mapping.json')
        mapping_data = {
            'class_to_label': self.class_to_label,
            'label_to_class': self.label_to_class
        }
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"类别映射已保存到: {mapping_file}")

        stats_file = os.path.join(self.output_dir, 'preprocessing_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        self.logger.info(f"统计信息已保存到: {stats_file}")

        params_file = os.path.join(self.output_dir, 'preprocessing_params.json')
        params = {
            'target_sr': self.target_sr,
            'frame_length_ms': self.frame_length * 1000 / self.target_sr,
            'frame_shift_ms': self.frame_shift * 1000 / self.target_sr,
            'preemphasis_coeff': self.preemphasis_coeff,
            'silence_threshold': self.silence_threshold,
            'frame_length_samples': self.frame_length,
            'frame_shift_samples': self.frame_shift
        }
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        self.logger.info(f"预处理参数已保存到: {params_file}")

    def generate_visualization(self, processed_data):
        try:
            class_counts = [self.stats['class_counts'][class_name] for class_name in self.label_to_class.values()]
            class_names = list(self.label_to_class.values())

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            axes[0, 0].bar(class_names, class_counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('各类别音频文件数量分布')
            axes[0, 0].set_xlabel('鲸鱼类别')
            axes[0, 0].set_ylabel('文件数量')
            axes[0, 0].tick_params(axis='x', rotation=45)

            all_durations = []
            for data in processed_data:
                all_durations.append(data['original_duration'])

            axes[0, 1].hist(all_durations, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('音频时长分布')
            axes[0, 1].set_xlabel('时长 (秒)')
            axes[0, 1].set_ylabel('文件数量')

            avg_durations = []
            for class_name in class_names:
                if class_name in self.stats['class_durations']:
                    avg_durations.append(self.stats['class_durations'][class_name]['avg_duration'])
                else:
                    avg_durations.append(0)

            axes[1, 0].bar(class_names, avg_durations, color='orange', alpha=0.7)
            axes[1, 0].set_title('各类别平均音频时长')
            axes[1, 0].set_xlabel('鲸鱼类别')
            axes[1, 0].set_ylabel('平均时长 (秒)')
            axes[1, 0].tick_params(axis='x', rotation=45)

            frame_counts = [len(data['frames']) for data in processed_data]
            axes[1, 1].hist(frame_counts, bins=30, color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('处理后帧数分布')
            axes[1, 1].set_xlabel('帧数')
            axes[1, 1].set_ylabel('文件数量')

            plt.tight_layout()

            plot_file = os.path.join(self.output_dir, 'preprocessing_visualization.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"可视化图表已保存到: {plot_file}")

        except Exception as e:
            self.logger.error(f"生成可视化图表失败: {str(e)}")

    def print_statistics(self):
        print("\n" + "=" * 60)
        print("          鲸鱼叫声数据预处理统计")
        print("=" * 60)

        print(f"\n 总体统计:")
        print(f"   总文件数: {self.stats['total_files']}")
        print(f"   成功处理: {self.stats['processed_files']}")
        print(f"   处理失败: {self.stats['failed_files']}")
        print(f"   成功率: {self.stats['processed_files'] / self.stats['total_files'] * 100:.1f}%")

        print(f"\n  时长统计:")
        print(f"   总时长: {self.stats['total_duration']:.2f} 秒 ({self.stats['total_duration'] / 60:.1f} 分钟)")
        print(f"   平均时长: {self.stats['avg_duration']:.2f} 秒")

        print(f"\n 类别统计:")
        for i, (class_name, label) in enumerate(self.class_to_label.items()):
            count = self.stats['class_counts'].get(class_name, 0)
            print(f"   {label}. {class_name}: {count} 个文件")

            if class_name in self.stats['class_durations']:
                class_stat = self.stats['class_durations'][class_name]
                print(f"      平均时长: {class_stat['avg_duration']:.2f}±{class_stat['std_duration']:.2f} 秒")
                print(f"      时长范围: {class_stat['min_duration']:.2f}~{class_stat['max_duration']:.2f} 秒")

        print(f"\n 预处理参数:")
        print(f"   目标采样率: {self.target_sr} Hz")
        print(f"   帧长: {self.frame_length * 1000 / self.target_sr:.1f} ms ({self.frame_length} 采样点)")
        print(f"   帧移: {self.frame_shift * 1000 / self.target_sr:.1f} ms ({self.frame_shift} 采样点)")
        print(f"   预加重系数: {self.preemphasis_coeff}")
        print(f"   静音阈值: {self.silence_threshold}")

        print("\n" + "=" * 60)

def main():
    print("鲸鱼叫声识别 - 数据预处理模块")
    print("开始处理音频数据...")

    try:
        preprocessor = AudioPreprocessor(
            data_path="C:\\Users\\86198\\Desktop\\我的毕设代码\\pythonProject1\\data",
            target_sr=16000,
            frame_length_ms=25,
            frame_shift_ms=10,
            preemphasis_coeff=0.97,
            silence_threshold=0.01
        )

        processed_data = preprocessor.process_all_files()

        preprocessor.save_processed_data(processed_data)

        preprocessor.generate_visualization(processed_data)

        preprocessor.print_statistics()

        print(f"\n 数据预处理完成！结果保存在 '{preprocessor.output_dir}' 目录中")

    except Exception as e:
        print(f" 预处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
