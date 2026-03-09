import os
import numpy as np
import librosa
import pickle
import json
from scipy import signal, stats
from scipy.fftpack import fft, dct
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class FeatureExtractor:

    def __init__(self, input_dir="preprocessed_data", output_dir="extracted_features"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()
        self.processed_data = None
        self.preprocessing_params = None
        self.class_mapping = None
        self.feature_config = {
            'mfcc': {
                'n_mfcc': 13,
                'n_fft': 512,
                'hop_length': 160,
                'include_delta': True,
                'include_delta2': True
            },
            'spectral': {
                'n_fft': 512,
                'hop_length': 160
            },
            'chroma': {
                'n_chroma': 12,
                'n_fft': 512,
                'hop_length': 160
            },
            'contrast': {
                'n_bands': 6,
                'n_fft': 512,
                'hop_length': 160
            }
        }
        self.feature_stats = {}
        self._initialize_feature_names()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/feature_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_feature_names(self):
        self.feature_names = []
        time_features = ['zcr', 'ste', 'stam', 'rms', 'peak_amplitude', 'form_factor', 'crest_factor']
        for feature in time_features:
            for stat in ['mean', 'std', 'max', 'min', 'median', 'skewness', 'kurtosis', 'percentile_25', 'percentile_75']:
                self.feature_names.append(f'time_{feature}_{stat}')
        mfcc_types = ['mfcc', 'delta_mfcc', 'delta2_mfcc']
        for mfcc_type in mfcc_types:
            for i in range(13):
                for stat in ['mean', 'std', 'max', 'min', 'median', 'skewness', 'kurtosis', 'percentile_25', 'percentile_75']:
                    self.feature_names.append(f'{mfcc_type}_{i}_{stat}')
        spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                             'spectral_flatness', 'spectral_centroid_delta', 'low_band_energy',
                             'mid_low_band_energy', 'mid_high_band_energy', 'high_band_energy']
        for feature in spectral_features:
            for stat in ['mean', 'std', 'max', 'min', 'median', 'skewness', 'kurtosis', 'percentile_25', 'percentile_75']:
                self.feature_names.append(f'{feature}_{stat}')
        for i in range(12):
            for stat in ['mean', 'std', 'max', 'min', 'median', 'skewness', 'kurtosis', 'percentile_25', 'percentile_75']:
                self.feature_names.append(f'chroma_{i}_{stat}')
        for i in range(6):
            for stat in ['mean', 'std', 'max', 'min', 'median', 'skewness', 'kurtosis', 'percentile_25', 'percentile_75']:
                self.feature_names.append(f'spectral_contrast_{i}_{stat}')
        self.logger.info(f"预定义特征总数: {len(self.feature_names)}")

    def load_preprocessed_data(self):
        try:
            data_file = os.path.join(self.input_dir, 'preprocessed_data.pkl')
            with open(data_file, 'rb') as f:
                self.processed_data = pickle.load(f)
            self.logger.info(f"成功加载 {len(self.processed_data)} 个预处理音频文件")
            params_file = os.path.join(self.input_dir, 'preprocessing_params.json')
            with open(params_file, 'r', encoding='utf-8') as f:
                self.preprocessing_params = json.load(f)
            mapping_file = os.path.join(self.input_dir, 'class_mapping.json')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
            self.logger.info("预处理数据加载完成")
            return True
        except Exception as e:
            self.logger.error(f"加载预处理数据失败: {str(e)}")
            return False

    def extract_time_domain_features(self, frames):
        time_features = {}
        try:
            zcr = np.zeros(len(frames))
            for i, frame in enumerate(frames):
                zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))))
                zcr[i] = zero_crossings / (2.0 * len(frame))
            ste = np.sum(frames ** 2, axis=1)
            stam = np.mean(np.abs(frames), axis=1)
            rms = np.sqrt(ste)
            peak_amplitude = np.max(np.abs(frames), axis=1)
            form_factor = rms / (stam + 1e-8)
            crest_factor = peak_amplitude / (rms + 1e-8)
            time_features = {
                'zcr': zcr,
                'ste': ste,
                'stam': stam,
                'rms': rms,
                'peak_amplitude': peak_amplitude,
                'form_factor': form_factor,
                'crest_factor': crest_factor
            }
        except Exception as e:
            self.logger.error(f"时域特征提取失败: {str(e)}")
            num_frames = len(frames) if len(frames) > 0 else 1
            time_features = {
                'zcr': np.zeros(num_frames),
                'ste': np.zeros(num_frames),
                'stam': np.zeros(num_frames),
                'rms': np.zeros(num_frames),
                'peak_amplitude': np.zeros(num_frames),
                'form_factor': np.zeros(num_frames),
                'crest_factor': np.zeros(num_frames)
            }
        return time_features

    def extract_mfcc_features(self, frames, sr):
        try:
            hop_length = self.feature_config['mfcc']['hop_length']
            frame_length = frames.shape[1]
            reconstructed_length = (len(frames) - 1) * hop_length + frame_length
            reconstructed_signal = np.zeros(reconstructed_length)
            for i, frame in enumerate(frames):
                start = i * hop_length
                end = start + frame_length
                if end <= reconstructed_length:
                    reconstructed_signal[start:end] += frame
            n_mfcc = self.feature_config['mfcc']['n_mfcc']
            n_fft = self.feature_config['mfcc']['n_fft']
            mfcc = librosa.feature.mfcc(
                y=reconstructed_signal,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=40
            )
            mfcc_features = {'mfcc': mfcc}
            if self.feature_config['mfcc']['include_delta']:
                delta_mfcc = librosa.feature.delta(mfcc)
                mfcc_features['delta_mfcc'] = delta_mfcc
            if self.feature_config['mfcc']['include_delta2']:
                delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                mfcc_features['delta2_mfcc'] = delta2_mfcc
        except Exception as e:
            self.logger.error(f"MFCC特征提取失败: {str(e)}")
            n_mfcc = self.feature_config['mfcc']['n_mfcc']
            dummy_frames = max(1, len(frames) // 10)
            mfcc_features = {
                'mfcc': np.zeros((n_mfcc, dummy_frames)),
                'delta_mfcc': np.zeros((n_mfcc, dummy_frames)),
                'delta2_mfcc': np.zeros((n_mfcc, dummy_frames))
            }
        return mfcc_features

    def extract_spectral_features(self, frames, sr):
        try:
            hop_length = self.feature_config['spectral']['hop_length']
            frame_length = frames.shape[1]
            reconstructed_length = (len(frames) - 1) * hop_length + frame_length
            reconstructed_signal = np.zeros(reconstructed_length)
            for i, frame in enumerate(frames):
                start = i * hop_length
                end = start + frame_length
                if end <= reconstructed_length:
                    reconstructed_signal[start:end] += frame
            n_fft = self.feature_config['spectral']['n_fft']
            spectral_centroids = librosa.feature.spectral_centroid(
                y=reconstructed_signal, sr=sr, n_fft=n_fft, hop_length=hop_length
            )[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=reconstructed_signal, sr=sr, n_fft=n_fft, hop_length=hop_length
            )[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=reconstructed_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
            )[0]
            spectral_flatness = librosa.feature.spectral_flatness(
                y=reconstructed_signal, n_fft=n_fft, hop_length=hop_length
            )[0]
            spectral_centroid_delta = np.diff(spectral_centroids, prepend=spectral_centroids[0])
            stft = librosa.stft(reconstructed_signal, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            freq_bins = magnitude.shape[0]
            band_size = freq_bins // 4
            low_band_energy = np.mean(magnitude[:band_size], axis=0)
            mid_low_band_energy = np.mean(magnitude[band_size:2*band_size], axis=0)
            mid_high_band_energy = np.mean(magnitude[2*band_size:3*band_size], axis=0)
            high_band_energy = np.mean(magnitude[3*band_size:], axis=0)
            spectral_features = {
                'spectral_centroid': spectral_centroids,
                'spectral_bandwidth': spectral_bandwidth,
                'spectral_rolloff': spectral_rolloff,
                'spectral_flatness': spectral_flatness,
                'spectral_centroid_delta': spectral_centroid_delta,
                'low_band_energy': low_band_energy,
                'mid_low_band_energy': mid_low_band_energy,
                'mid_high_band_energy': mid_high_band_energy,
                'high_band_energy': high_band_energy
            }
        except Exception as e:
            self.logger.error(f"频域特征提取失败: {str(e)}")
            dummy_frames = max(1, len(frames) // 10)
            spectral_features = {
                'spectral_centroid': np.zeros(dummy_frames),
                'spectral_bandwidth': np.zeros(dummy_frames),
                'spectral_rolloff': np.zeros(dummy_frames),
                'spectral_flatness': np.zeros(dummy_frames),
                'spectral_centroid_delta': np.zeros(dummy_frames),
                'low_band_energy': np.zeros(dummy_frames),
                'mid_low_band_energy': np.zeros(dummy_frames),
                'mid_high_band_energy': np.zeros(dummy_frames),
                'high_band_energy': np.zeros(dummy_frames)
            }
        return spectral_features

    def extract_chroma_features(self, frames, sr):
        try:
            hop_length = self.feature_config['chroma']['hop_length']
            frame_length = frames.shape[1]
            reconstructed_length = (len(frames) - 1) * hop_length + frame_length
            reconstructed_signal = np.zeros(reconstructed_length)
            for i, frame in enumerate(frames):
                start = i * hop_length
                end = start + frame_length
                if end <= reconstructed_length:
                    reconstructed_signal[start:end] += frame
            chroma = librosa.feature.chroma_stft(
                y=reconstructed_signal,
                sr=sr,
                n_fft=self.feature_config['chroma']['n_fft'],
                hop_length=hop_length,
                n_chroma=self.feature_config['chroma']['n_chroma']
            )
        except Exception as e:
            self.logger.error(f"色度特征提取失败: {str(e)}")
            n_chroma = self.feature_config['chroma']['n_chroma']
            dummy_frames = max(1, len(frames) // 10)
            chroma = np.zeros((n_chroma, dummy_frames))
        return {'chroma': chroma}

    def extract_contrast_features(self, frames, sr):
        try:
            hop_length = self.feature_config['contrast']['hop_length']
            frame_length = frames.shape[1]
            reconstructed_length = (len(frames) - 1) * hop_length + frame_length
            reconstructed_signal = np.zeros(reconstructed_length)
            for i, frame in enumerate(frames):
                start = i * hop_length
                end = start + frame_length
                if end <= reconstructed_length:
                    reconstructed_signal[start:end] += frame
            contrast = librosa.feature.spectral_contrast(
                y=reconstructed_signal,
                sr=sr,
                n_fft=self.feature_config['contrast']['n_fft'],
                hop_length=hop_length,
                n_bands=self.feature_config['contrast']['n_bands']
            )
        except Exception as e:
            self.logger.error(f"谱对比度特征提取失败: {str(e)}")
            n_bands = self.feature_config['contrast']['n_bands']
            dummy_frames = max(1, len(frames) // 10)
            contrast = np.zeros((n_bands, dummy_frames))
        return {'spectral_contrast': contrast}

    def extract_statistical_features(self, feature_matrix):
        try:
            if feature_matrix.ndim == 1:
                feature_matrix = feature_matrix.reshape(1, -1)
            stats_features = []
            for feature_dim in feature_matrix:
                if len(feature_dim) == 0 or np.all(feature_dim == 0):
                    stats_features.extend([0.0] * 9)
                    continue
                feature_dim = np.nan_to_num(feature_dim, nan=0.0, posinf=0.0, neginf=0.0)
                mean_val = np.mean(feature_dim) if len(feature_dim) > 0 else 0.0
                std_val = np.std(feature_dim) if len(feature_dim) > 0 else 0.0
                max_val = np.max(feature_dim) if len(feature_dim) > 0 else 0.0
                min_val = np.min(feature_dim) if len(feature_dim) > 0 else 0.0
                median_val = np.median(feature_dim) if len(feature_dim) > 0 else 0.0
                try:
                    skewness = stats.skew(feature_dim) if len(feature_dim) > 2 else 0.0
                    kurtosis = stats.kurtosis(feature_dim) if len(feature_dim) > 2 else 0.0
                except:
                    skewness = 0.0
                    kurtosis = 0.0
                try:
                    percentile_25 = np.percentile(feature_dim, 25) if len(feature_dim) > 0 else 0.0
                    percentile_75 = np.percentile(feature_dim, 75) if len(feature_dim) > 0 else 0.0
                except:
                    percentile_25 = 0.0
                    percentile_75 = 0.0
                stat_values = [mean_val, std_val, max_val, min_val, median_val,
                               skewness, kurtosis, percentile_25, percentile_75]
                stat_values = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in stat_values]
                stats_features.extend(stat_values)
            return np.array(stats_features, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"统计特征计算失败: {str(e)}")
            expected_length = feature_matrix.shape[0] * 9 if feature_matrix.ndim == 2 else 9
            return np.zeros(expected_length, dtype=np.float32)

    def extract_features_from_audio(self, audio_data):
        try:
            frames = audio_data['frames']
            sr = audio_data['target_sr']
            if frames is None or len(frames) == 0:
                self.logger.warning(f"空的帧数据: {audio_data.get('file_path', 'unknown')}")
                return np.zeros(len(self.feature_names), dtype=np.float32)
            feature_vector = []
            time_features = self.extract_time_domain_features(frames)
            for feature_name in ['zcr', 'ste', 'stam', 'rms', 'peak_amplitude', 'form_factor', 'crest_factor']:
                if feature_name in time_features:
                    stats = self.extract_statistical_features(time_features[feature_name].reshape(1, -1))
                    feature_vector.extend(stats)
                else:
                    feature_vector.extend([0.0] * 9)
            mfcc_features = self.extract_mfcc_features(frames, sr)
            for mfcc_type in ['mfcc', 'delta_mfcc', 'delta2_mfcc']:
                if mfcc_type in mfcc_features:
                    mfcc_matrix = mfcc_features[mfcc_type]
                    if mfcc_matrix.shape[0] != 13:
                        target_mfcc = np.zeros((13, mfcc_matrix.shape[1]))
                        min_dim = min(13, mfcc_matrix.shape[0])
                        target_mfcc[:min_dim, :] = mfcc_matrix[:min_dim, :]
                        mfcc_matrix = target_mfcc
                    for i in range(13):
                        stats = self.extract_statistical_features(mfcc_matrix[i, :].reshape(1, -1))
                        feature_vector.extend(stats)
                else:
                    feature_vector.extend([0.0] * (13 * 9))
            spectral_features = self.extract_spectral_features(frames, sr)
            spectral_names = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                              'spectral_flatness', 'spectral_centroid_delta', 'low_band_energy',
                              'mid_low_band_energy', 'mid_high_band_energy', 'high_band_energy']
            for feature_name in spectral_names:
                if feature_name in spectral_features:
                    stats = self.extract_statistical_features(spectral_features[feature_name].reshape(1, -1))
                    feature_vector.extend(stats)
                else:
                    feature_vector.extend([0.0] * 9)
            chroma_features = self.extract_chroma_features(frames, sr)
            if 'chroma' in chroma_features:
                chroma_matrix = chroma_features['chroma']
                if chroma_matrix.shape[0] != 12:
                    target_chroma = np.zeros((12, chroma_matrix.shape[1]))
                    min_dim = min(12, chroma_matrix.shape[0])
                    target_chroma[:min_dim, :] = chroma_matrix[:min_dim, :]
                    chroma_matrix = target_chroma
                for i in range(12):
                    stats = self.extract_statistical_features(chroma_matrix[i, :].reshape(1, -1))
                    feature_vector.extend(stats)
            else:
                feature_vector.extend([0.0] * (12 * 9))
            contrast_features = self.extract_contrast_features(frames, sr)
            if 'spectral_contrast' in contrast_features:
                contrast_matrix = contrast_features['spectral_contrast']
                if contrast_matrix.shape[0] != 6:
                    target_contrast = np.zeros((6, contrast_matrix.shape[1]))
                    min_dim = min(6, contrast_matrix.shape[0])
                    target_contrast[:min_dim, :] = contrast_matrix[:min_dim, :]
                    contrast_matrix = target_contrast
                for i in range(6):
                    stats = self.extract_statistical_features(contrast_matrix[i, :].reshape(1, -1))
                    feature_vector.extend(stats)
            else:
                feature_vector.extend([0.0] * (6 * 9))
            feature_vector = np.array(feature_vector, dtype=np.float32)
            if len(feature_vector) != len(self.feature_names):
                self.logger.warning(f"特征向量长度不匹配: {len(feature_vector)} vs {len(self.feature_names)}")
                if len(feature_vector) > len(self.feature_names):
                    feature_vector = feature_vector[:len(self.feature_names)]
                else:
                    padding = np.zeros(len(self.feature_names) - len(feature_vector), dtype=np.float32)
                    feature_vector = np.concatenate([feature_vector, padding])
            return feature_vector
        except Exception as e:
            self.logger.error(f"特征提取失败 {audio_data.get('file_path', 'unknown')}: {str(e)}")
            return np.zeros(len(self.feature_names), dtype=np.float32)

    def extract_all_features(self):
        if self.processed_data is None:
            self.logger.error("请先加载预处理数据")
            return None
        all_features = []
        self.logger.info(f"开始提取 {len(self.processed_data)} 个音频文件的特征...")
        self.logger.info(f"预定义特征维度: {len(self.feature_names)}")
        for i, audio_data in enumerate(tqdm(self.processed_data, desc="提取特征")):
            feature_vector = self.extract_features_from_audio(audio_data)
            if feature_vector is not None and len(feature_vector) > 0:
                feature_data = {
                    'features': feature_vector,
                    'label': audio_data['label'],
                    'class_name': audio_data['class_name'],
                    'file_path': audio_data['file_path'],
                    'original_duration': audio_data['original_duration']
                }
                all_features.append(feature_data)
            else:
                self.logger.warning(f"文件 {audio_data['file_path']} 特征提取失败")
        self.logger.info(f"特征提取完成，成功提取 {len(all_features)} 个文件的特征")
        self.logger.info(f"实际特征维度: {len(all_features[0]['features']) if all_features else 0}")
        return all_features

    def normalize_features(self, all_features):
        self.logger.info("开始特征标准化...")
        if not all_features:
            self.logger.error("没有特征数据可供标准化")
            return [], {}
        features_by_class = {}
        for feature_data in all_features:
            class_name = feature_data['class_name']
            if class_name not in features_by_class:
                features_by_class[class_name] = []
            features_by_class[class_name].append(feature_data['features'])
        all_feature_vectors = np.array([fd['features'] for fd in all_features])
        if np.any(np.isnan(all_feature_vectors)) or np.any(np.isinf(all_feature_vectors)):
            self.logger.warning("发现NaN或无穷大值，正在清理...")
            all_feature_vectors = np.nan_to_num(all_feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)
        global_scaler = StandardScaler()
        normalized_all_features = global_scaler.fit_transform(all_feature_vectors)
        normalized_features = []
        for i, feature_data in enumerate(all_features):
            new_feature_data = feature_data.copy()
            new_feature_data['features'] = normalized_all_features[i]
            new_feature_data['original_features'] = feature_data['features']
            normalized_features.append(new_feature_data)
        class_scalers = {}
        for class_name, class_features in features_by_class.items():
            if len(class_features) > 1:
                class_feature_matrix = np.array(class_features)
                if np.any(np.isnan(class_feature_matrix)) or np.any(np.isinf(class_feature_matrix)):
                    class_feature_matrix = np.nan_to_num(class_feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                scaler = StandardScaler()
                scaler.fit(class_feature_matrix)
                class_scalers[class_name] = scaler
        scalers = {
            'global_scaler': global_scaler,
            'class_scalers': class_scalers
        }
        self.logger.info("特征标准化完成")
        return normalized_features, scalers

    def calculate_feature_statistics(self, all_features):
        self.logger.info("计算特征统计信息...")
        if not all_features:
            self.logger.error("没有特征数据可供统计")
            return
        class_stats = {}
        for class_name in self.class_mapping['label_to_class'].values():
            class_features = [fd['features'] for fd in all_features if fd['class_name'] == class_name]
            if class_features:
                class_feature_matrix = np.array(class_features)
                class_feature_matrix = np.nan_to_num(class_feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                class_stats[class_name] = {
                    'count': len(class_features),
                    'mean': np.mean(class_feature_matrix, axis=0),
                    'std': np.std(class_feature_matrix, axis=0),
                    'min': np.min(class_feature_matrix, axis=0),
                    'max': np.max(class_feature_matrix, axis=0)
                }
        all_feature_matrix = np.array([fd['features'] for fd in all_features])
        all_feature_matrix = np.nan_to_num(all_feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        global_stats = {
            'total_samples': len(all_features),
            'feature_dimension': all_feature_matrix.shape[1],
            'global_mean': np.mean(all_feature_matrix, axis=0),
            'global_std': np.std(all_feature_matrix, axis=0),
            'global_min': np.min(all_feature_matrix, axis=0),
            'global_max': np.max(all_feature_matrix, axis=0)
        }
        self.feature_stats = {
            'class_stats': class_stats,
            'global_stats': global_stats,
            'feature_names': self.feature_names
        }

    def save_features(self, normalized_features, scalers):
        self.logger.info("保存特征数据...")
        features_file = os.path.join(self.output_dir, 'extracted_features.pkl')
        with open(features_file, 'wb') as f:
            pickle.dump(normalized_features, f)
        scalers_file = os.path.join(self.output_dir, 'feature_scalers.pkl')
        with open(scalers_file, 'wb') as f:
            pickle.dump(scalers, f)
        names_file = os.path.join(self.output_dir, 'feature_names.pkl')
        with open(names_file, 'wb') as f:
            pickle.dump(self.feature_names, f)
        config_file = os.path.join(self.output_dir, 'feature_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.feature_config, f, ensure_ascii=False, indent=2)
        stats_to_save = {}
        for key, value in self.feature_stats.items():
            if key == 'class_stats':
                stats_to_save[key] = {}
                for class_name, class_stat in value.items():
                    stats_to_save[key][class_name] = {}
                    for stat_name, stat_value in class_stat.items():
                        if isinstance(stat_value, np.ndarray):
                            stats_to_save[key][class_name][stat_name] = stat_value.tolist()
                        else:
                            stats_to_save[key][class_name][stat_name] = stat_value
            elif key == 'global_stats':
                stats_to_save[key] = {}
                for stat_name, stat_value in value.items():
                    if isinstance(stat_value, np.ndarray):
                        stats_to_save[key][stat_name] = stat_value.tolist()
                    else:
                        stats_to_save[key][stat_name] = stat_value
            else:
                stats_to_save[key] = value
        stats_file = os.path.join(self.output_dir, 'feature_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
        self.logger.info("特征数据保存完成")

    def generate_feature_visualization(self, all_features):
        try:
            self.logger.info("生成特征可视化图表...")
            if not all_features:
                self.logger.warning("没有特征数据可供可视化")
                return
            feature_matrix = np.array([fd['features'] for fd in all_features])
            labels = [fd['label'] for fd in all_features]
            class_names = [fd['class_name'] for fd in all_features]
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            n_features_to_plot = min(3, feature_matrix.shape[1])
            for i in range(n_features_to_plot):
                axes[0, i].hist(feature_matrix[:, i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, i].set_title(f'特征 {i+1} 分布')
                axes[0, i].set_xlabel('特征值')
                axes[0, i].set_ylabel('频次')
                axes[0, i].grid(True, alpha=0.3)
            unique_classes = list(set(class_names))
            feature_indices = [0, 1, 2] if feature_matrix.shape[1] >= 3 else list(range(feature_matrix.shape[1]))
            for i, feature_idx in enumerate(feature_indices):
                if i < 3:
                    class_feature_data = []
                    class_labels = []
                    for class_name in unique_classes:
                        class_feat = feature_matrix[[j for j, cn in enumerate(class_names) if cn == class_name], feature_idx]
                        if len(class_feat) > 0:
                            class_feature_data.append(class_feat)
                            class_labels.append(class_name)
                    if class_feature_data:
                        box_plot = axes[1, i].boxplot(class_feature_data, labels=class_labels, patch_artist=True)
                        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
                        for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                            patch.set_facecolor(color)
                        axes[1, i].set_title(f'特征 {feature_idx+1} 按类别分布')
                        axes[1, i].set_ylabel('特征值')
                        axes[1, i].tick_params(axis='x', rotation=45)
                        axes[1, i].grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = os.path.join(self.output_dir, 'feature_visualization.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            if feature_matrix.shape[1] <= 100 and feature_matrix.shape[1] > 1:
                plt.figure(figsize=(12, 10))
                if feature_matrix.shape[1] > 50:
                    selected_indices = np.random.choice(feature_matrix.shape[1], 50, replace=False)
                    selected_features = feature_matrix[:, selected_indices]
                    correlation_matrix = np.corrcoef(selected_features.T)
                else:
                    correlation_matrix = np.corrcoef(feature_matrix.T)
                correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
                sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True,
                           cbar_kws={'label': '相关系数'})
                plt.title('特征相关性矩阵')
                corr_plot_file = os.path.join(self.output_dir, 'feature_correlation.png')
                plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"特征相关性图保存到: {corr_plot_file}")
            self.logger.info(f"特征可视化图保存到: {plot_file}")
        except Exception as e:
            self.logger.error(f"生成特征可视化失败: {str(e)}")

    def print_feature_summary(self):
        print("\n" + "="*70)
        print("          鲸鱼叫声特征提取统计报告")
        print("="*70)
        if not self.feature_stats:
            print(" 没有可用的特征统计信息")
            return
        global_stats = self.feature_stats['global_stats']
        class_stats = self.feature_stats['class_stats']
        print(f"\n 特征提取总体统计:")
        print(f"   总样本数: {global_stats['total_samples']}")
        print(f"   特征维度: {global_stats['feature_dimension']}")
        print(f"   特征名称数: {len(self.feature_names)}")
        feature_types = {}
        for name in self.feature_names:
            if '_' in name:
                prefix = name.split('_')[0]
                if prefix in ['mfcc', 'delta', 'delta2']:
                    prefix = 'mfcc_related'
                elif prefix in ['spectral']:
                    prefix = 'spectral'
                elif prefix in ['time']:
                    prefix = 'time'
                elif prefix in ['chroma']:
                    prefix = 'chroma'
                else:
                    prefix = 'other'
            else:
                prefix = 'other'
            feature_types[prefix] = feature_types.get(prefix, 0) + 1
        print(f"\n 特征类型分布:")
        for feature_type, count in feature_types.items():
            print(f"   {feature_type}: {count} 个特征")
        print(f"\n 各类别特征统计:")
        for class_name, stats in class_stats.items():
            print(f"   {class_name}: {stats['count']} 个样本")
            mean_values = stats['mean']
            std_values = stats['std']
            mean_values = np.nan_to_num(mean_values, nan=0.0, posinf=0.0, neginf=0.0)
            std_values = np.nan_to_num(std_values, nan=0.0, posinf=0.0, neginf=0.0)
            mean_norm = np.linalg.norm(mean_values)
            std_mean = np.mean(std_values)
            print(f"      特征均值范数: {mean_norm:.4f}")
            print(f"      平均标准差: {std_mean:.4f}")
        print(f"\n  特征提取配置:")
        for config_type, config_params in self.feature_config.items():
            print(f"   {config_type}:")
            for param, value in config_params.items():
                print(f"      {param}: {value}")
        print(f"\n 特征质量指标:")
        all_means = global_stats['global_mean']
        all_stds = global_stats['global_std']
        all_means = np.nan_to_num(all_means, nan=0.0, posinf=0.0, neginf=0.0)
        all_stds = np.nan_to_num(all_stds, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"   全局均值范围: [{np.min(all_means):.4f}, {np.max(all_means):.4f}]")
        print(f"   全局标准差范围: [{np.min(all_stds):.4f}, {np.max(all_stds):.4f}]")
        print(f"   零值特征比例: {np.sum(all_stds == 0) / len(all_stds) * 100:.2f}%")
        print("\n" + "="*70)

def main():
    print("鲸鱼叫声识别 - 特征提取模块")
    print("开始提取音频特征...")
    try:
        extractor = FeatureExtractor(
            input_dir="preprocessed_data",
            output_dir="extracted_features"
        )
        if not extractor.load_preprocessed_data():
            print(" 无法加载预处理数据，请先运行data_preprocessing.py")
            return
        all_features = extractor.extract_all_features()
        if not all_features:
            print(" 特征提取失败")
            return
        normalized_features, scalers = extractor.normalize_features(all_features)
        extractor.calculate_feature_statistics(normalized_features)
        extractor.save_features(normalized_features, scalers)
        extractor.generate_feature_visualization(normalized_features)
        extractor.print_feature_summary()
        print(f"\n 特征提取完成！结果保存在 '{extractor.output_dir}' 文件夹中")
        print(f"\n 生成的文件:")
        output_files = [
            "extracted_features.pkl - 提取的特征数据",
            "feature_scalers.pkl - 特征标准化器",
            "feature_names.pkl - 特征名称列表",
            "feature_config.json - 特征提取配置",
            "feature_statistics.json - 特征统计信息",
            "feature_visualization.png - 特征可视化图表",
            "feature_correlation.png - 特征相关性热图",
            "feature_extraction.log - 提取过程日志"
        ]
        for file_desc in output_files:
            print(f"   ✓ {file_desc}")
    except Exception as e:
        print(f" 特征提取过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
