import os
import numpy as np
import pickle
import json
import time
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class CrossCorrelationMatcher:

    def __init__(self, dict_dir="correlation_dictionaries",
                 feature_dir="extracted_features", output_dir="matching_results"):
        self.dict_dir = dict_dir
        self.feature_dir = feature_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()
        self.dictionaries = None
        self.test_data = None
        self.feature_names = None
        self.class_mapping = None
        self.match_config = {
            'correlation_methods': ['pearson', 'cosine', 'normalized_cross_correlation'],
            'decision_strategies': ['max_correlation', 'weighted_voting', 'threshold_based'],
            'confidence_threshold': 0.3,
            'top_k_templates': 5,
            'use_parallel': True,
            'max_workers': min(8, mp.cpu_count()),
            'enable_preprocessing': True,
            'sliding_window_size': None,
            'multi_scale_factors': [0.8, 1.0, 1.2]
        }
        self.matching_results = {}
        self.performance_metrics = {}

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/matching.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            dict_file = os.path.join(self.dict_dir, 'correlation_dictionaries.pkl')
            with open(dict_file, 'rb') as f:
                self.dictionaries = pickle.load(f)
            self.logger.info(f"成功加载字典: {len(self.dictionaries)} 种类型")
            features_file = os.path.join(self.feature_dir, 'extracted_features.pkl')
            with open(features_file, 'rb') as f:
                all_features = pickle.load(f)
            names_file = os.path.join(self.feature_dir, 'feature_names.pkl')
            with open(names_file, 'rb') as f:
                self.feature_names = pickle.load(f)
            mapping_file = os.path.join("preprocessed_data", 'class_mapping.json')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
            self.test_data = self._get_test_data(all_features)
            self.logger.info(f"测试数据: {len(self.test_data)} 个样本")
            return True
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return False

    def _get_test_data(self, all_features, test_ratio=0.3):
        class_data = {}
        for sample in all_features:
            class_name = sample['class_name']
            if class_name not in class_data:
                class_data[class_name] = []
            class_data[class_name].append(sample)
        test_data = []
        np.random.seed(42)
        for class_name, samples in class_data.items():
            samples.sort(key=lambda x: x['file_path'])
            split_idx = int(len(samples) * (1 - test_ratio))
            test_data.extend(samples[split_idx:])
        return test_data

    def extract_feature_group(self, features, group_name):
        if group_name == 'time_domain':
            indices = [i for i, name in enumerate(self.feature_names) if 'time_' in name]
        elif group_name == 'mfcc':
            indices = [i for i, name in enumerate(self.feature_names)
                       if any(x in name for x in ['mfcc', 'delta'])]
        elif group_name == 'spectral':
            indices = [i for i, name in enumerate(self.feature_names)
                       if 'spectral' in name or any(
                    x in name for x in ['band_energy', 'centroid', 'bandwidth', 'rolloff', 'flatness'])]
        elif group_name == 'chroma':
            indices = [i for i, name in enumerate(self.feature_names) if 'chroma' in name]
        else:
            indices = list(range(len(features)))
        return features[indices] if indices else features

    def calculate_correlation(self, signal1, signal2, method='pearson'):
        try:
            s1 = np.array(signal1).flatten()
            s2 = np.array(signal2).flatten()
            min_len = min(len(s1), len(s2))
            if min_len == 0:
                return 0.0
            s1 = s1[:min_len]
            s2 = s2[:min_len]
            if method == 'pearson':
                if np.std(s1) == 0 or np.std(s2) == 0:
                    return 0.0
                correlation, _ = pearsonr(s1, s2)
            elif method == 'cosine':
                dot_product = np.dot(s1, s2)
                norm_product = np.linalg.norm(s1) * np.linalg.norm(s2)
                correlation = dot_product / (norm_product + 1e-8)
            elif method == 'normalized_cross_correlation':
                s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-8)
                s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-8)
                correlation = np.corrcoef(s1_norm, s2_norm)[0, 1]
            else:
                correlation = 0.0
            if np.isnan(correlation):
                correlation = 0.0
            return float(correlation)
        except Exception:
            return 0.0

    def sliding_window_correlation(self, signal, template, method='pearson'):
        try:
            signal = np.array(signal).flatten()
            template = np.array(template).flatten()
            if len(template) > len(signal):
                return self.calculate_correlation(signal, template[:len(signal)], method)
            max_correlation = -1.0
            window_size = len(template)
            for i in range(len(signal) - window_size + 1):
                window = signal[i:i + window_size]
                corr = self.calculate_correlation(window, template, method)
                max_correlation = max(max_correlation, abs(corr))
            return max_correlation
        except Exception:
            return 0.0

    def multi_scale_correlation(self, signal, template, method='pearson'):
        try:
            correlations = []
            for scale in self.match_config['multi_scale_factors']:
                new_length = max(1, int(len(template) * scale))
                if new_length != len(template):
                    scaled_template = np.interp(
                        np.linspace(0, len(template) - 1, new_length),
                        np.arange(len(template)),
                        template
                    )
                else:
                    scaled_template = template
                corr = self.sliding_window_correlation(signal, scaled_template, method)
                correlations.append(corr)
            return max(correlations)
        except Exception:
            return 0.0

    def match_sample_to_templates(self, sample_features, templates, feature_group=None, method='pearson'):
        class_correlations = {}
        if feature_group:
            sample_features = self.extract_feature_group(sample_features, feature_group)
        for class_name, class_templates in templates.items():
            correlations = []
            if 'features' in class_templates:
                template_features = class_templates['features']
                if feature_group:
                    template_features = np.array([
                        self.extract_feature_group(tf, feature_group)
                        for tf in template_features
                    ])
                for template in template_features:
                    corr = self.multi_scale_correlation(sample_features, template, method)
                    correlations.append(corr)
                if correlations:
                    top_k = min(self.match_config['top_k_templates'], len(correlations))
                    top_correlations = sorted(correlations, reverse=True)[:top_k]
                    class_score = np.mean(top_correlations)
                else:
                    class_score = 0.0
            else:
                class_score = 0.0
            class_correlations[class_name] = class_score
        return class_correlations

    def make_decision(self, class_correlations, strategy='max_correlation'):
        if not class_correlations or all(score <= 0 for score in class_correlations.values()):
            return None, 0.0
        if strategy == 'max_correlation':
            predicted_class = max(class_correlations, key=class_correlations.get)
            max_score = class_correlations[predicted_class]
            sorted_scores = sorted(class_correlations.values(), reverse=True)
            if len(sorted_scores) > 1:
                confidence = max_score - sorted_scores[1]
            else:
                confidence = max_score
        elif strategy == 'weighted_voting':
            scores = list(class_correlations.values())
            weights = np.exp(np.array(scores))
            weights = weights / np.sum(weights)
            weighted_scores = {class_name: score * weight
                               for (class_name, score), weight in zip(class_correlations.items(), weights)}
            predicted_class = max(weighted_scores, key=weighted_scores.get)
            confidence = weighted_scores[predicted_class]
        elif strategy == 'threshold_based':
            max_class = max(class_correlations, key=class_correlations.get)
            max_score = class_correlations[max_class]
            if max_score >= self.match_config['confidence_threshold']:
                predicted_class = max_class
                confidence = max_score
            else:
                predicted_class = None
                confidence = max_score
        else:
            predicted_class = max(class_correlations, key=class_correlations.get)
            confidence = class_correlations[predicted_class]
        return predicted_class, confidence

    def match_single_sample(self, sample, dict_type, dict_name, method='pearson'):
        try:
            start_time = time.time()
            if dict_type not in self.dictionaries or dict_name not in self.dictionaries[dict_type]:
                return None
            dict_info = self.dictionaries[dict_type][dict_name]
            templates = dict_info['templates']
            sample_features = sample['features']
            if dict_type == 'fusion':
                sample_features = self._apply_fusion_transform(sample_features, dict_info)
            feature_group = dict_name if dict_type == 'single_feature' else None
            class_correlations = self.match_sample_to_templates(
                sample_features, templates, feature_group, method
            )
            for strategy in self.match_config['decision_strategies']:
                predicted_class, confidence = self.make_decision(class_correlations, strategy)
                end_time = time.time()
                result = {
                    'sample_info': {
                        'file_path': sample['file_path'],
                        'true_class': sample['class_name'],
                        'true_label': sample['label']
                    },
                    'prediction': {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'class_correlations': class_correlations
                    },
                    'method_info': {
                        'dict_type': dict_type,
                        'dict_name': dict_name,
                        'correlation_method': method,
                        'decision_strategy': strategy,
                        'matching_time': end_time - start_time
                    }
                }
                return result
        except Exception as e:
            self.logger.error(f"样本匹配失败: {str(e)}")
            return None

    def _apply_fusion_transform(self, features, dict_info):
        try:
            if 'pca_model' in dict_info:
                features = dict_info['pca_model'].transform(features.reshape(1, -1))[0]
            elif 'lda_model' in dict_info:
                features = dict_info['lda_model'].transform(features.reshape(1, -1))[0]
            elif 'feature_weights' in dict_info:
                features = features * dict_info['feature_weights']
            return features
        except Exception as e:
            self.logger.warning(f"融合变换失败: {str(e)}")
            return features

    def batch_matching(self, dict_type, dict_name):
        self.logger.info(f"开始批量匹配: {dict_type}/{dict_name}")
        results = []
        if self.match_config['use_parallel']:
            with ThreadPoolExecutor(max_workers=self.match_config['max_workers']) as executor:
                futures = []
                for method in self.match_config['correlation_methods']:
                    for sample in self.test_data:
                        future = executor.submit(
                            self.match_single_sample, sample, dict_type, dict_name, method
                        )
                        futures.append(future)
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"匹配{dict_type}/{dict_name}"):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for method in self.match_config['correlation_methods']:
                for sample in tqdm(self.test_data, desc=f"匹配{dict_type}/{dict_name}/{method}"):
                    result = self.match_single_sample(sample, dict_type, dict_name, method)
                    if result:
                        results.append(result)
        return results

    def evaluate_performance(self, results):
        if not results:
            return {}
        method_results = {}
        for result in results:
            method_key = (
                result['method_info']['correlation_method'],
                result['method_info']['decision_strategy']
            )
            if method_key not in method_results:
                method_results[method_key] = []
            method_results[method_key].append(result)
        performance = {}
        for method_key, method_results_list in method_results.items():
            correlation_method, decision_strategy = method_key
            true_labels = []
            pred_labels = []
            confidences = []
            matching_times = []
            for result in method_results_list:
                true_labels.append(result['sample_info']['true_class'])
                pred_class = result['prediction']['predicted_class']
                pred_labels.append(pred_class if pred_class else 'Unknown')
                confidences.append(result['prediction']['confidence'])
                matching_times.append(result['method_info']['matching_time'])
            correct_predictions = sum(1 for true, pred in zip(true_labels, pred_labels)
                                      if true == pred)
            accuracy = correct_predictions / len(true_labels) if true_labels else 0.0
            rejection_rate = sum(1 for pred in pred_labels if pred == 'Unknown') / len(pred_labels)
            unique_classes = list(self.class_mapping['label_to_class'].values()) + ['Unknown']
            cm = confusion_matrix(true_labels, pred_labels, labels=unique_classes)
            perf = {
                'accuracy': accuracy,
                'rejection_rate': rejection_rate,
                'avg_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'avg_matching_time': np.mean(matching_times),
                'total_samples': len(true_labels),
                'confusion_matrix': cm.tolist(),
                'class_labels': unique_classes
            }
            method_name = f"{correlation_method}_{decision_strategy}"
            performance[method_name] = perf
        return performance

    def run_all_matching(self):
        self.logger.info("开始全面匹配测试...")
        all_results = {}
        for dict_type, dict_data in self.dictionaries.items():
            for dict_name in dict_data.keys():
                key = f"{dict_type}_{dict_name}"
                self.logger.info(f"测试字典: {key}")
                results = self.batch_matching(dict_type, dict_name)
                performance = self.evaluate_performance(results)
                all_results[key] = {
                    'results': results,
                    'performance': performance
                }
                self.logger.info(f"完成 {key}: 处理了 {len(results)} 个结果")
        self.matching_results = all_results
        self.logger.info("全面匹配测试完成")
        return all_results

    def save_results(self):
        self.logger.info("保存匹配结果...")
        results_file = os.path.join(self.output_dir, 'matching_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(self.matching_results, f)
        performance_summary = {}
        for dict_key, dict_results in self.matching_results.items():
            performance_summary[dict_key] = dict_results['performance']
        perf_file = os.path.join(self.output_dir, 'performance_summary.json')
        with open(perf_file, 'w', encoding='utf-8') as f:
            json.dump(self._convert_numpy_to_list(performance_summary), f,
                      ensure_ascii=False, indent=2)
        config_file = os.path.join(self.output_dir, 'matching_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.match_config, f, ensure_ascii=False, indent=2)
        self.logger.info("结果保存完成")

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

    def generate_performance_visualization(self):
        try:
            self.logger.info("生成性能可视化图表...")
            dict_names = []
            method_names = []
            accuracies = []
            for dict_key, dict_results in self.matching_results.items():
                for method_name, performance in dict_results['performance'].items():
                    dict_names.append(dict_key)
                    method_names.append(method_name)
                    accuracies.append(performance['accuracy'])
            if not accuracies:
                self.logger.warning("没有性能数据可供可视化")
                return
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            unique_dicts = list(set(dict_names))
            dict_accuracies = {}
            for dict_name in unique_dicts:
                dict_accs = [acc for dn, acc in zip(dict_names, accuracies) if dn == dict_name]
                dict_accuracies[dict_name] = np.mean(dict_accs) if dict_accs else 0
            axes[0, 0].bar(range(len(unique_dicts)), list(dict_accuracies.values()),
                           color='skyblue', alpha=0.7)
            axes[0, 0].set_title('不同字典平均识别准确率')
            axes[0, 0].set_xlabel('字典类型')
            axes[0, 0].set_ylabel('准确率')
            axes[0, 0].set_xticks(range(len(unique_dicts)))
            axes[0, 0].set_xticklabels(unique_dicts, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            for i, acc in enumerate(dict_accuracies.values()):
                axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            unique_methods = list(set(method_names))
            method_accuracies = {}
            for method in unique_methods:
                method_accs = [acc for mn, acc in zip(method_names, accuracies) if mn == method]
                method_accuracies[method] = np.mean(method_accs) if method_accs else 0
            axes[0, 1].bar(range(len(unique_methods)), list(method_accuracies.values()),
                           color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('不同方法平均识别准确率')
            axes[0, 1].set_xlabel('匹配方法')
            axes[0, 1].set_ylabel('准确率')
            axes[0, 1].set_xticks(range(len(unique_methods)))
            axes[0, 1].set_xticklabels(unique_methods, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].hist(accuracies, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('识别准确率分布')
            axes[1, 0].set_xlabel('准确率')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].axvline(np.mean(accuracies), color='red', linestyle='--',
                               label=f'平均值: {np.mean(accuracies):.3f}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            best_performance = None
            best_accuracy = 0
            best_key = ""
            for dict_key, dict_results in self.matching_results.items():
                for method_name, performance in dict_results['performance'].items():
                    if performance['accuracy'] > best_accuracy:
                        best_accuracy = performance['accuracy']
                        best_performance = performance
                        best_key = f"{dict_key}_{method_name}"
            if best_performance:
                cm = np.array(best_performance['confusion_matrix'])
                class_labels = best_performance['class_labels']
                valid_indices = [i for i in range(len(class_labels))
                                 if np.sum(cm[i, :]) > 0 or np.sum(cm[:, i]) > 0]
                if valid_indices:
                    valid_cm = cm[np.ix_(valid_indices, valid_indices)]
                    valid_labels = [class_labels[i] for i in valid_indices]
                    im = axes[1, 1].imshow(valid_cm, interpolation='nearest', cmap='Blues')
                    axes[1, 1].set_title(f'最佳性能混淆矩阵\n{best_key} (准确率: {best_accuracy:.3f})')
                    tick_marks = np.arange(len(valid_labels))
                    axes[1, 1].set_xticks(tick_marks)
                    axes[1, 1].set_yticks(tick_marks)
                    axes[1, 1].set_xticklabels(valid_labels, rotation=45, ha='right')
                    axes[1, 1].set_yticklabels(valid_labels)
                    thresh = valid_cm.max() / 2.
                    for i in range(len(valid_labels)):
                        for j in range(len(valid_labels)):
                            axes[1, 1].text(j, i, f'{valid_cm[i, j]}',
                                            ha="center", va="center",
                                            color="white" if valid_cm[i, j] > thresh else "black")
                    axes[1, 1].set_ylabel('真实类别')
                    axes[1, 1].set_xlabel('预测类别')
                    plt.colorbar(im, ax=axes[1, 1])
            plt.tight_layout()
            plot_file = os.path.join(self.output_dir, 'performance_visualization.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"性能可视化图表保存到: {plot_file}")
        except Exception as e:
            self.logger.error(f"生成性能可视化失败: {str(e)}")

    def print_performance_summary(self):
        print("\n" + "=" * 80)
        print("          鲸鱼叫声互相关匹配识别性能报告")
        print("=" * 80)
        if not self.matching_results:
            print("没有匹配结果可供分析")
            return
        all_performances = []
        best_performance = None
        best_accuracy = 0
        best_config = ""
        print(f"\n 总体统计:")
        total_dict_count = len(self.matching_results)
        total_method_count = 0
        total_samples = 0
        for dict_key, dict_results in self.matching_results.items():
            method_count = len(dict_results['performance'])
            total_method_count += method_count
            for method_name, performance in dict_results['performance'].items():
                all_performances.append(performance)
                total_samples = max(total_samples, performance['total_samples'])
                if performance['accuracy'] > best_accuracy:
                    best_accuracy = performance['accuracy']
                    best_performance = performance
                    best_config = f"{dict_key} + {method_name}"
        print(f"   测试字典数量: {total_dict_count}")
        print(f"   测试方法组合: {total_method_count}")
        print(f"   测试样本总数: {total_samples}")
        if all_performances:
            accuracies = [p['accuracy'] for p in all_performances]
            print(f"   平均识别准确率: {np.mean(accuracies):.4f}")
            print(f"   准确率标准差: {np.std(accuracies):.4f}")
            print(f"   最高准确率: {np.max(accuracies):.4f}")
            print(f"   最低准确率: {np.min(accuracies):.4f}")
        print(f"\n🏆 最佳性能配置:")
        if best_performance:
            print(f"   配置: {best_config}")
            print(f"   准确率: {best_performance['accuracy']:.4f}")
            print(f"   拒绝率: {best_performance['rejection_rate']:.4f}")
            print(f"   平均置信度: {best_performance['avg_confidence']:.4f}")
            print(f"   平均匹配时间: {best_performance['avg_matching_time']:.4f}秒")
        print(f"\n 各字典类型性能对比:")
        dict_performances = {}
        for dict_key, dict_results in self.matching_results.items():
            dict_type = dict_key.split('_')[0] + '_' + dict_key.split('_')[1]
            if dict_type not in dict_performances:
                dict_performances[dict_type] = []
            for performance in dict_results['performance'].values():
                dict_performances[dict_type].append(performance['accuracy'])
        for dict_type, accuracies in dict_performances.items():
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"   {dict_type}: {avg_acc:.4f}±{std_acc:.4f}")
        print(f"\n 各匹配方法性能对比:")
        method_performances = {}
        for dict_results in self.matching_results.values():
            for method_name, performance in dict_results['performance'].items():
                if method_name not in method_performances:
                    method_performances[method_name] = []
                method_performances[method_name].append(performance['accuracy'])
        for method_name, accuracies in method_performances.items():
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"   {method_name}: {avg_acc:.4f}±{std_acc:.4f}")
        print(f"\n  性能指标:")
        if all_performances:
            avg_times = [p['avg_matching_time'] for p in all_performances]
            confidences = [p['avg_confidence'] for p in all_performances]
            rejection_rates = [p['rejection_rate'] for p in all_performances]
            print(f"   平均匹配时间: {np.mean(avg_times):.4f}±{np.std(avg_times):.4f}秒")
            print(f"   平均置信度: {np.mean(confidences):.4f}±{np.std(confidences):.4f}")
            print(f"   平均拒绝率: {np.mean(rejection_rates):.4f}±{np.std(rejection_rates):.4f}")
        print(f"\n 详细结果 (前5名):")
        all_configs = []
        for dict_key, dict_results in self.matching_results.items():
            for method_name, performance in dict_results['performance'].items():
                all_configs.append({
                    'config': f"{dict_key}_{method_name}",
                    'accuracy': performance['accuracy'],
                    'confidence': performance['avg_confidence'],
                    'time': performance['avg_matching_time'],
                    'rejection_rate': performance['rejection_rate']
                })
        all_configs.sort(key=lambda x: x['accuracy'], reverse=True)
        for i, config in enumerate(all_configs[:5]):
            print(f"   {i + 1}. {config['config']}")
            print(f"      准确率: {config['accuracy']:.4f}, 置信度: {config['confidence']:.4f}")
            print(f"      时间: {config['time']:.4f}s, 拒绝率: {config['rejection_rate']:.4f}")
        print(f"\n  匹配配置:")
        for param, value in self.match_config.items():
            print(f"   {param}: {value}")
        print("\n" + "=" * 80)

def main():
    print("鲸鱼叫声识别 - 互相关匹配识别模块")
    print("开始执行匹配识别测试...")
    try:
        matcher = CrossCorrelationMatcher(
            dict_dir="correlation_dictionaries",
            feature_dir="extracted_features",
            output_dir="matching_results"
        )
        if not matcher.load_data():
            print("无法加载数据，请先运行前序模块")
            return
        print(f"\n开始测试 {len(matcher.dictionaries)} 种字典类型...")
        results = matcher.run_all_matching()
        if not results:
            print(" 没有生成任何匹配结果")
            return
        matcher.save_results()
        matcher.generate_performance_visualization()
        matcher.print_performance_summary()
        print(f"\n 互相关匹配识别完成！结果保存在 '{matcher.output_dir}' 目录中")
        print(f"\n 生成的文件:")
        output_files = [
            "matching_results.pkl - 完整匹配结果数据",
            "performance_summary.json - 性能汇总统计",
            "matching_config.json - 匹配配置参数",
            "performance_visualization.png - 性能对比图表",
            "matching.log - 匹配过程日志"
        ]
        for file_desc in output_files:
            print(f"   ✓ {file_desc}")
        total_tests = sum(len(dict_results['performance']) for dict_results in results.values())
        print(f"\n 测试摘要:")
        print(f"   • 测试了 {len(results)} 个字典")
        print(f"   • 运行了 {total_tests} 种方法组合")
        print(f"   • 处理了 {len(matcher.test_data)} 个测试样本")
    except Exception as e:
        print(f" 匹配识别过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
