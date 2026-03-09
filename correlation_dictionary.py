import os
import numpy as np
import pickle
import json
from scipy import signal, stats
from scipy.fft import fft, ifft
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


class CorrelationDictionaryBuilder:

    def __init__(self, input_dir="extracted_features", output_dir="correlation_dictionaries"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()
        self.features_data = None
        self.feature_names = None
        self.class_mapping = None
        self.scalers = None
        self.dict_config = {
            'template_ratio': 0.65,
            'min_templates_per_class': 5,
            'max_templates_per_class': 100,
            'pca_components': 0.95,
            'lda_components': 5,
            'kmeans_clusters': [3, 5, 8],
            'correlation_method': 'pearson'
        }
        self.feature_groups = {
            'time_domain': [],
            'mfcc': [],
            'spectral': [],
            'chroma': [],
            'contrast': []
        }
        self.dictionaries = {}
        self.dict_statistics = {}

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/dictionary_building.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_feature_data(self):
        try:
            features_file = os.path.join(self.input_dir, 'extracted_features.pkl')
            with open(features_file, 'rb') as f:
                self.features_data = pickle.load(f)
            self.logger.info(f"成功加载 {len(self.features_data)} 个特征样本")
            names_file = os.path.join(self.input_dir, 'feature_names.pkl')
            with open(names_file, 'rb') as f:
                self.feature_names = pickle.load(f)
            self.logger.info(f"特征维度: {len(self.feature_names)}")
            scalers_file = os.path.join(self.input_dir, 'feature_scalers.pkl')
            with open(scalers_file, 'rb') as f:
                self.scalers = pickle.load(f)
            mapping_file = os.path.join("preprocessed_data", 'class_mapping.json')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
            self._analyze_feature_groups()
            self.logger.info("特征数据加载完成")
            return True
        except Exception as e:
            self.logger.error(f"加载特征数据失败: {str(e)}")
            return False

    def _analyze_feature_groups(self):
        for i, name in enumerate(self.feature_names):
            if 'time_' in name:
                self.feature_groups['time_domain'].append(i)
            elif any(x in name for x in ['mfcc', 'delta']):
                self.feature_groups['mfcc'].append(i)
            elif 'spectral' in name or any(
                    x in name for x in ['band_energy', 'centroid', 'bandwidth', 'rolloff', 'flatness']):
                self.feature_groups['spectral'].append(i)
            elif 'chroma' in name:
                self.feature_groups['chroma'].append(i)
            elif 'contrast' in name:
                self.feature_groups['contrast'].append(i)
        self.logger.info("特征分组统计:")
        for group_name, indices in self.feature_groups.items():
            self.logger.info(f"  {group_name}: {len(indices)} 个特征")

    def split_train_test(self, test_ratio=0.3):
        class_data = {}
        for sample in self.features_data:
            class_name = sample['class_name']
            if class_name not in class_data:
                class_data[class_name] = []
            class_data[class_name].append(sample)
        train_data = []
        test_data = []
        for class_name, samples in class_data.items():
            random.shuffle(samples)
            split_idx = int(len(samples) * (1 - test_ratio))
            train_data.extend(samples[:split_idx])
            test_data.extend(samples[split_idx:])
        self.logger.info(f"数据划分: 训练集 {len(train_data)}, 测试集 {len(test_data)}")
        return train_data, test_data

    def select_templates(self, train_data, strategy='random'):
        templates = {}
        class_data = {}
        for sample in train_data:
            class_name = sample['class_name']
            if class_name not in class_data:
                class_data[class_name] = []
            class_data[class_name].append(sample)
        for class_name, samples in class_data.items():
            self.logger.info(f"为类别 {class_name} 选择模板...")
            feature_matrix = np.array([s['features'] for s in samples])
            template_count = max(
                self.dict_config['min_templates_per_class'],
                min(
                    int(len(samples) * self.dict_config['template_ratio']),
                    self.dict_config['max_templates_per_class']
                )
            )
            if strategy == 'random':
                selected_indices = random.sample(range(len(samples)), template_count)
            elif strategy == 'kmeans':
                selected_indices = self._kmeans_template_selection(feature_matrix, template_count)
            elif strategy == 'distance_based':
                selected_indices = self._distance_based_selection(feature_matrix, template_count)
            else:
                selected_indices = random.sample(range(len(samples)), template_count)
            class_templates = {
                'samples': [samples[i] for i in selected_indices],
                'features': feature_matrix[selected_indices],
                'mean_template': np.mean(feature_matrix[selected_indices], axis=0),
                'median_template': np.median(feature_matrix[selected_indices], axis=0),
                'std_template': np.std(feature_matrix[selected_indices], axis=0),
                'selection_strategy': strategy,
                'template_count': template_count
            }
            templates[class_name] = class_templates
            self.logger.info(f"  选择了 {template_count} 个模板")
        return templates

    def _kmeans_template_selection(self, feature_matrix, template_count):
        try:
            n_clusters = min(template_count, len(feature_matrix))
            if n_clusters < 2:
                return list(range(len(feature_matrix)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            selected_indices = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.any(cluster_mask):
                    cluster_samples = feature_matrix[cluster_mask]
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    distances = np.sum((cluster_samples - cluster_center) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    original_indices = np.where(cluster_mask)[0]
                    selected_indices.append(original_indices[closest_idx])
            return selected_indices
        except Exception as e:
            self.logger.warning(f"K-means选择失败，使用随机选择: {str(e)}")
            return random.sample(range(len(feature_matrix)), min(template_count, len(feature_matrix)))

    def _distance_based_selection(self, feature_matrix, template_count):
        try:
            if len(feature_matrix) <= template_count:
                return list(range(len(feature_matrix)))
            center = np.mean(feature_matrix, axis=0)
            distances = np.sum((feature_matrix - center) ** 2, axis=1)
            sorted_indices = np.argsort(distances)
            selected_indices = []
            total_samples = len(feature_matrix)
            for i in range(template_count):
                percentile = (i / template_count) * 80 + 10
                target_idx = int(percentile / 100 * total_samples)
                selected_indices.append(sorted_indices[target_idx])
            return selected_indices
        except Exception as e:
            self.logger.warning(f"基于距离的选择失败，使用随机选择: {str(e)}")
            return random.sample(range(len(feature_matrix)), template_count)

    def build_single_feature_dictionaries(self, train_data):
        self.logger.info("开始构建单特征字典...")
        single_feature_dicts = {}
        for group_name, feature_indices in self.feature_groups.items():
            if not feature_indices:
                continue
            self.logger.info(f"构建 {group_name} 特征字典...")
            group_features = []
            for sample in train_data:
                group_feature = sample['features'][feature_indices]
                group_features.append({
                    'features': group_feature,
                    'class_name': sample['class_name'],
                    'label': sample['label']
                })
            templates = self.select_templates(group_features, strategy='kmeans')
            dict_stats = self._calculate_dictionary_statistics(templates)
            single_feature_dicts[group_name] = {
                'templates': templates,
                'feature_indices': feature_indices,
                'statistics': dict_stats,
                'feature_count': len(feature_indices)
            }
        self.dictionaries['single_feature'] = single_feature_dicts
        self.logger.info("单特征字典构建完成")
        return single_feature_dicts

    def build_fusion_dictionaries(self, train_data):
        self.logger.info("开始构建多特征融合字典...")
        fusion_dicts = {}
        self.logger.info("构建加权融合字典...")
        weighted_dict = self._build_weighted_fusion_dict(train_data)
        fusion_dicts['weighted_fusion'] = weighted_dict
        self.logger.info("构建PCA降维字典...")
        pca_dict = self._build_pca_dict(train_data)
        fusion_dicts['pca_fusion'] = pca_dict
        self.logger.info("构建LDA优化字典...")
        lda_dict = self._build_lda_dict(train_data)
        fusion_dicts['lda_fusion'] = lda_dict
        self.dictionaries['fusion'] = fusion_dicts
        self.logger.info("多特征融合字典构建完成")
        return fusion_dicts

    def _build_weighted_fusion_dict(self, train_data):
        feature_weights = self._calculate_feature_weights(train_data)
        weighted_features = []
        for sample in train_data:
            weighted_feature = sample['features'] * feature_weights
            weighted_features.append({
                'features': weighted_feature,
                'class_name': sample['class_name'],
                'label': sample['label']
            })
        templates = self.select_templates(weighted_features, strategy='distance_based')
        return {
            'templates': templates,
            'feature_weights': feature_weights,
            'statistics': self._calculate_dictionary_statistics(templates)
        }

    def _build_pca_dict(self, train_data):
        try:
            all_features = np.array([sample['features'] for sample in train_data])
            pca = PCA(n_components=self.dict_config['pca_components'])
            pca_features = pca.fit_transform(all_features)
            pca_data = []
            for i, sample in enumerate(train_data):
                pca_data.append({
                    'features': pca_features[i],
                    'class_name': sample['class_name'],
                    'label': sample['label']
                })
            templates = self.select_templates(pca_data, strategy='kmeans')
            return {
                'templates': templates,
                'pca_model': pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'n_components': pca.n_components_,
                'statistics': self._calculate_dictionary_statistics(templates)
            }
        except Exception as e:
            self.logger.error(f"PCA字典构建失败: {str(e)}")
            return None

    def _build_lda_dict(self, train_data):
        try:
            all_features = np.array([sample['features'] for sample in train_data])
            all_labels = np.array([sample['label'] for sample in train_data])
            lda = LinearDiscriminantAnalysis(n_components=min(self.dict_config['lda_components'], 5))
            lda_features = lda.fit_transform(all_features, all_labels)
            lda_data = []
            for i, sample in enumerate(train_data):
                lda_data.append({
                    'features': lda_features[i],
                    'class_name': sample['class_name'],
                    'label': sample['label']
                })
            templates = self.select_templates(lda_data, strategy='distance_based')
            return {
                'templates': templates,
                'lda_model': lda,
                'explained_variance_ratio': lda.explained_variance_ratio_,
                'n_components': lda_features.shape[1],
                'statistics': self._calculate_dictionary_statistics(templates)
            }
        except Exception as e:
            self.logger.error(f"LDA字典构建失败: {str(e)}")
            return None

    def _calculate_feature_weights(self, train_data):
        try:
            class_features = {}
            for sample in train_data:
                class_name = sample['class_name']
                if class_name not in class_features:
                    class_features[class_name] = []
                class_features[class_name].append(sample['features'])
            for class_name in class_features:
                class_features[class_name] = np.array(class_features[class_name])
            n_features = len(train_data[0]['features'])
            feature_weights = np.ones(n_features)
            for i in range(n_features):
                within_class_var = 0
                total_samples = 0
                for class_name, features in class_features.items():
                    if len(features) > 1:
                        var = np.var(features[:, i])
                        within_class_var += var * len(features)
                        total_samples += len(features)
                within_class_var /= total_samples
                class_means = [np.mean(features[:, i]) for features in class_features.values()]
                between_class_var = np.var(class_means)
                if within_class_var > 0:
                    feature_weights[i] = between_class_var / within_class_var
                else:
                    feature_weights[i] = 1.0
            feature_weights = feature_weights / np.max(feature_weights)
            return feature_weights
        except Exception as e:
            self.logger.warning(f"特征权重计算失败，使用均匀权重: {str(e)}")
            return np.ones(len(train_data[0]['features']))

    def _calculate_dictionary_statistics(self, templates):
        stats = {
            'class_count': len(templates),
            'total_templates': 0,
            'class_statistics': {}
        }
        for class_name, class_templates in templates.items():
            template_features = class_templates['features']
            class_stats = {
                'template_count': len(template_features),
                'feature_dimension': template_features.shape[1],
                'mean_norm': np.linalg.norm(class_templates['mean_template']),
                'std_norm': np.linalg.norm(class_templates['std_template']),
                'intra_class_distance': self._calculate_intra_class_distance(template_features)
            }
            stats['class_statistics'][class_name] = class_stats
            stats['total_templates'] += class_stats['template_count']
        stats['inter_class_distances'] = self._calculate_inter_class_distances(templates)
        return stats

    def _calculate_intra_class_distance(self, template_features):
        if len(template_features) < 2:
            return 0.0
        distances = []
        for i in range(len(template_features)):
            for j in range(i + 1, len(template_features)):
                dist = np.linalg.norm(template_features[i] - template_features[j])
                distances.append(dist)
        return np.mean(distances)

    def _calculate_inter_class_distances(self, templates):
        class_names = list(templates.keys())
        distances = {}
        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if i != j:
                    mean1 = templates[class1]['mean_template']
                    mean2 = templates[class2]['mean_template']
                    dist = np.linalg.norm(mean1 - mean2)
                    distances[f"{class1}_vs_{class2}"] = dist
        return distances

    def calculate_correlation(self, signal1, signal2, method='pearson'):
        try:
            min_len = min(len(signal1), len(signal2))
            s1 = signal1[:min_len]
            s2 = signal2[:min_len]
            if method == 'pearson':
                correlation = np.corrcoef(s1, s2)[0, 1]
            elif method == 'normalized_cross_correlation':
                s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-8)
                s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-8)
                correlation = np.max(np.correlate(s1_norm, s2_norm, mode='full'))
            elif method == 'cosine_similarity':
                dot_product = np.dot(s1, s2)
                norm_product = np.linalg.norm(s1) * np.linalg.norm(s2)
                correlation = dot_product / (norm_product + 1e-8)
            else:
                correlation = np.corrcoef(s1, s2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            return correlation
        except Exception as e:
            self.logger.warning(f"相关性计算失败: {str(e)}")
            return 0.0

    def fft_correlation(self, signal1, signal2):
        try:
            max_len = max(len(signal1), len(signal2))
            s1 = np.pad(signal1, (0, max_len - len(signal1)), 'constant')
            s2 = np.pad(signal2, (0, max_len - len(signal2)), 'constant')
            fft1 = fft(s1)
            fft2 = fft(s2[::-1])
            correlation = ifft(fft1 * fft2).real
            max_corr = np.max(np.abs(correlation))
            if max_corr > 0:
                correlation = correlation / max_corr
            return np.max(correlation)
        except Exception as e:
            self.logger.warning(f"FFT相关性计算失败: {str(e)}")
            return 0.0

    def sliding_window_correlation(self, signal, template, window_size=None):
        if window_size is None:
            window_size = len(template)
        correlations = []
        signal_len = len(signal)
        for i in range(signal_len - window_size + 1):
            window = signal[i:i + window_size]
            corr = self.calculate_correlation(window, template)
            correlations.append(corr)
        return np.array(correlations)

    def save_dictionaries(self):
        self.logger.info("保存互相关字典...")
        dict_file = os.path.join(self.output_dir, 'correlation_dictionaries.pkl')
        with open(dict_file, 'wb') as f:
            pickle.dump(self.dictionaries, f)
        config_file = os.path.join(self.output_dir, 'dictionary_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.dict_config, f, ensure_ascii=False, indent=2)
        stats_file = os.path.join(self.output_dir, 'dictionary_statistics.json')
        self._save_dictionary_statistics(stats_file)
        self.logger.info("字典保存完成")

    def _save_dictionary_statistics(self, stats_file):
        stats_to_save = {}
        for dict_type, dict_data in self.dictionaries.items():
            stats_to_save[dict_type] = {}
            for dict_name, dict_info in dict_data.items():
                if 'statistics' in dict_info:
                    stats = dict_info['statistics']
                    stats_to_save[dict_type][dict_name] = self._convert_numpy_to_list(stats)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)

    def _convert_numpy_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    def generate_dictionary_visualization(self):
        try:
            self.logger.info("生成字典可视化图表...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            dict_sizes = {}
            for dict_type, dict_data in self.dictionaries.items():
                for dict_name, dict_info in dict_data.items():
                    if 'statistics' in dict_info:
                        total_templates = dict_info['statistics']['total_templates']
                        dict_sizes[f"{dict_type}_{dict_name}"] = total_templates
            if dict_sizes:
                names = list(dict_sizes.keys())
                sizes = list(dict_sizes.values())
                axes[0, 0].bar(range(len(names)), sizes, color='skyblue')
                axes[0, 0].set_title('字典模板数量统计')
                axes[0, 0].set_xlabel('字典类型')
                axes[0, 0].set_ylabel('模板数量')
                axes[0, 0].set_xticks(range(len(names)))
                axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
                axes[0, 0].grid(True, alpha=0.3)
            if 'single_feature' in self.dictionaries:
                sample_dict = None
                for dict_name, dict_info in self.dictionaries['single_feature'].items():
                    if 'statistics' in dict_info and 'inter_class_distances' in dict_info['statistics']:
                        sample_dict = dict_info['statistics']['inter_class_distances']
                        break
                if sample_dict:
                    class_names = list(self.class_mapping['label_to_class'].values())
                    n_classes = len(class_names)
                    distance_matrix = np.zeros((n_classes, n_classes))
                    for i, class1 in enumerate(class_names):
                        for j, class2 in enumerate(class_names):
                            if i != j:
                                key = f"{class1}_vs_{class2}"
                                if key in sample_dict:
                                    distance_matrix[i, j] = sample_dict[key]
                    im = axes[0, 1].imshow(distance_matrix, cmap='viridis')
                    axes[0, 1].set_title('类间距离矩阵')
                    axes[0, 1].set_xticks(range(n_classes))
                    axes[0, 1].set_yticks(range(n_classes))
                    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
                    axes[0, 1].set_yticklabels(class_names)
                    plt.colorbar(im, ax=axes[0, 1])
            if 'fusion' in self.dictionaries and 'weighted_fusion' in self.dictionaries['fusion']:
                weights = self.dictionaries['fusion']['weighted_fusion']['feature_weights']
                axes[1, 0].hist(weights, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1, 0].set_title('特征权重分布')
                axes[1, 0].set_xlabel('权重值')
                axes[1, 0].set_ylabel('特征数量')
                axes[1, 0].grid(True, alpha=0.3)
            class_template_counts = {}
            for class_name in self.class_mapping['label_to_class'].values():
                class_template_counts[class_name] = 0
            for dict_type, dict_data in self.dictionaries.items():
                for dict_name, dict_info in dict_data.items():
                    if 'templates' in dict_info:
                        for class_name, class_templates in dict_info['templates'].items():
                            if 'template_count' in class_templates:
                                class_template_counts[class_name] += class_templates['template_count']
            if any(count > 0 for count in class_template_counts.values()):
                classes = list(class_template_counts.keys())
                counts = list(class_template_counts.values())
                bars = axes[1, 1].bar(classes, counts, color='lightcoral')
                axes[1, 1].set_title('各类别总模板数量')
                axes[1, 1].set_xlabel('鲸鱼类别')
                axes[1, 1].set_ylabel('模板数量')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                for bar, count in zip(bars, counts):
                    axes[1, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                                    f'{count}', ha='center', va='bottom')
            plt.tight_layout()
            plot_file = os.path.join(self.output_dir, 'dictionary_visualization.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"字典可视化图保存到: {plot_file}")
        except Exception as e:
            self.logger.error(f"生成字典可视化失败: {str(e)}")

    def print_dictionary_summary(self):
        print("\n" + "=" * 70)
        print("          鲸鱼叫声互相关字典构建报告")
        print("=" * 70)
        if not self.dictionaries:
            print(" 没有构建任何字典")
            return
        print(f"\n 字典构建总体统计:")
        total_dict_count = 0
        total_template_count = 0
        for dict_type, dict_data in self.dictionaries.items():
            type_dict_count = len(dict_data)
            type_template_count = 0
            for dict_name, dict_info in dict_data.items():
                if 'statistics' in dict_info:
                    type_template_count += dict_info['statistics']['total_templates']
            total_dict_count += type_dict_count
            total_template_count += type_template_count
            print(f"   {dict_type}: {type_dict_count} 个字典, {type_template_count} 个模板")
        print(f"   总计: {total_dict_count} 个字典, {total_template_count} 个模板")
        print(f"\n 单特征字典详情:")
        if 'single_feature' in self.dictionaries:
            for feature_type, dict_info in self.dictionaries['single_feature'].items():
                if 'statistics' in dict_info:
                    stats = dict_info['statistics']
                    feature_count = dict_info['feature_count']
                    print(f"   {feature_type}:")
                    print(f"      特征维度: {feature_count}")
                    print(f"      总模板数: {stats['total_templates']}")
                    print(f"      类别数: {stats['class_count']}")
        print(f"\n 融合字典详情:")
        if 'fusion' in self.dictionaries:
            for fusion_type, dict_info in self.dictionaries['fusion'].items():
                if dict_info and 'statistics' in dict_info:
                    stats = dict_info['statistics']
                    print(f"   {fusion_type}:")
                    print(f"      总模板数: {stats['total_templates']}")
                    print(f"      类别数: {stats['class_count']}")
                    if fusion_type == 'pca_fusion' and 'n_components' in dict_info:
                        print(f"      PCA降维后维度: {dict_info['n_components']}")
                    elif fusion_type == 'lda_fusion' and 'n_components' in dict_info:
                        print(f"      LDA降维后维度: {dict_info['n_components']}")
        print(f"\n 各类别统计:")
        class_totals = {}
        for class_name in self.class_mapping['label_to_class'].values():
            class_totals[class_name] = 0
        for dict_type, dict_data in self.dictionaries.items():
            for dict_name, dict_info in dict_data.items():
                if 'statistics' in dict_info and 'class_statistics' in dict_info['statistics']:
                    for class_name, class_stats in dict_info['statistics']['class_statistics'].items():
                        if class_name in class_totals:
                            class_totals[class_name] += class_stats['template_count']
        for class_name, total_templates in class_totals.items():
            print(f"   {class_name}: {total_templates} 个模板")
        print(f"\n  字典构建配置:")
        for param, value in self.dict_config.items():
            print(f"   {param}: {value}")
        print(f"\n 字典质量指标:")
        inter_distances = []
        intra_distances = []
        for dict_type, dict_data in self.dictionaries.items():
            for dict_name, dict_info in dict_data.items():
                if 'statistics' in dict_info:
                    stats = dict_info['statistics']
                    if 'inter_class_distances' in stats:
                        inter_distances.extend(stats['inter_class_distances'].values())
                    if 'class_statistics' in stats:
                        for class_stats in stats['class_statistics'].values():
                            if 'intra_class_distance' in class_stats:
                                intra_distances.append(class_stats['intra_class_distance'])
        if inter_distances and intra_distances:
            avg_inter = np.mean(inter_distances)
            avg_intra = np.mean(intra_distances)
            separation_ratio = avg_inter / avg_intra if avg_intra > 0 else float('inf')
            print(f"   平均类间距离: {avg_inter:.4f}")
            print(f"   平均类内距离: {avg_intra:.4f}")
            print(f"   分离度比值: {separation_ratio:.4f}")
        print("\n" + "=" * 70)


def main():
    print("鲸鱼叫声识别 - 互相关字典构建模块")
    print("开始构建互相关字典...")
    try:
        builder = CorrelationDictionaryBuilder(
            input_dir="extracted_features",
            output_dir="correlation_dictionaries"
        )
        if not builder.load_feature_data():
            print(" 无法加载特征数据，请先运行feature_extraction.py")
            return
        train_data, test_data = builder.split_train_test(test_ratio=0.3)
        single_dicts = builder.build_single_feature_dictionaries(train_data)
        fusion_dicts = builder.build_fusion_dictionaries(train_data)
        builder.save_dictionaries()
        builder.generate_dictionary_visualization()
        builder.print_dictionary_summary()
        print(f"\n 互相关字典构建完成！结果保存在 '{builder.output_dir}' 目录中")
        print(f"\n 生成的文件:")
        output_files = [
            "correlation_dictionaries.pkl - 互相关字典数据",
            "dictionary_config.json - 字典构建配置",
            "dictionary_statistics.json - 字典统计信息",
            "dictionary_visualization.png - 字典可视化图表",
            "dictionary_building.log - 构建过程日志"
        ]
        for file_desc in output_files:
            print(f"   ✓ {file_desc}")
        print(f"\n 构建的字典类型:")
        for dict_type, dict_data in builder.dictionaries.items():
            print(f"   • {dict_type}:")
            for dict_name in dict_data.keys():
                print(f"     - {dict_name}")
    except Exception as e:
        print(f" 字典构建过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
