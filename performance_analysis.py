import os
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy import stats
from itertools import combinations
import warnings
import logging
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class PerformanceAnalyzer:

    def __init__(self, results_dir="matching_results", output_dir="performance_analysis"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()
        self.matching_results = None
        self.class_mapping = None
        self.feature_data = None
        self.analysis_results = {}

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            results_file = os.path.join(self.results_dir, 'matching_results.pkl')
            with open(results_file, 'rb') as f:
                self.matching_results = pickle.load(f)
            mapping_file = os.path.join("preprocessed_data", 'class_mapping.json')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
            features_file = os.path.join("extracted_features", 'extracted_features.pkl')
            with open(features_file, 'rb') as f:
                self.feature_data = pickle.load(f)
            self.logger.info(f"数据加载完成: {len(self.matching_results)} 个匹配结果")
            return True
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return False

    def analyze_accuracy_performance(self):
        self.logger.info("开始分析识别准确率性能...")
        accuracy_analysis = {}
        for dict_name, result in self.matching_results.items():
            true_labels = result['true_labels']
            predictions = result['predictions']
            processed_predictions = [pred if pred is not None else 'Unknown' for pred in predictions]
            total_accuracy = result['performance']['accuracy']
            class_accuracies = {}
            for class_name in self.class_mapping['label_to_class'].values():
                class_true = [i for i, label in enumerate(true_labels) if label == class_name]
                if class_true:
                    class_correct = sum(1 for i in class_true if processed_predictions[i] == class_name)
                    class_accuracies[class_name] = class_correct / len(class_true)
                else:
                    class_accuracies[class_name] = 0.0
            class_names = list(self.class_mapping['label_to_class'].values())
            cm = confusion_matrix(true_labels, processed_predictions, labels=class_names)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, processed_predictions, labels=class_names, average=None, zero_division=0
            )
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                true_labels, processed_predictions, labels=class_names, average='weighted', zero_division=0
            )
            accuracy_analysis[dict_name] = {
                'total_accuracy': total_accuracy,
                'class_accuracies': class_accuracies,
                'confusion_matrix': cm.tolist(),
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1.tolist(),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'support': support.tolist(),
                'class_names': class_names
            }
        self.analysis_results['accuracy_analysis'] = accuracy_analysis
        self.logger.info("识别准确率分析完成")
        return accuracy_analysis

    def analyze_feature_comparison(self):
        self.logger.info("开始分析不同特征对比...")
        single_feature_results = {k: v for k, v in self.matching_results.items()
                                  if k.startswith('single_feature')}
        fusion_results = {k: v for k, v in self.matching_results.items()
                          if k.startswith('fusion')}
        feature_performance = {}
        for dict_name, result in single_feature_results.items():
            feature_type = dict_name.split('_')[2]
            true_labels = result['true_labels']
            predictions = result['predictions']
            processed_predictions = [pred if pred is not None else 'Unknown' for pred in predictions]
            class_sensitivity = {}
            for class_name in self.class_mapping['label_to_class'].values():
                class_indices = [i for i, label in enumerate(true_labels) if label == class_name]
                if class_indices:
                    correct_predictions = sum(1 for i in class_indices if processed_predictions[i] == class_name)
                    sensitivity = correct_predictions / len(class_indices)
                    class_sensitivity[class_name] = sensitivity
                else:
                    class_sensitivity[class_name] = 0.0
            feature_performance[feature_type] = {
                'accuracy': result['performance']['accuracy'],
                'avg_confidence': result['performance']['avg_confidence'],
                'class_sensitivity': class_sensitivity,
                'processing_time': self._estimate_processing_time(feature_type)
            }
        fusion_performance = {}
        for dict_name, result in fusion_results.items():
            fusion_type = dict_name.split('_')[1]
            fusion_performance[fusion_type] = {
                'accuracy': result['performance']['accuracy'],
                'avg_confidence': result['performance']['avg_confidence'],
                'improvement_over_best_single': 0.0
            }
        best_single_accuracy = max(fp['accuracy'] for fp in feature_performance.values())
        for fusion_type, fp in fusion_performance.items():
            fp['improvement_over_best_single'] = fp['accuracy'] - best_single_accuracy
        feature_comparison = {
            'single_feature_performance': feature_performance,
            'fusion_performance': fusion_performance,
            'best_single_feature': max(feature_performance.items(),
                                       key=lambda x: x[1]['accuracy'])[0],
            'best_fusion_method': max(fusion_performance.items(),
                                      key=lambda x: x[1]['accuracy'])[0]
        }
        self.analysis_results['feature_comparison'] = feature_comparison
        self.logger.info("特征对比分析完成")
        return feature_comparison

    def _estimate_processing_time(self, feature_type):
        complexity_map = {
            'time_domain': 1.0,
            'spectral': 2.0,
            'mfcc': 3.0,
            'chroma': 2.5,
        }
        return complexity_map.get(feature_type, 2.0)

    def analyze_confusion_patterns(self):
        self.logger.info("开始分析混淆模式...")
        confusion_analysis = {}
        best_result_name, best_result = max(self.matching_results.items(),
                                             key=lambda x: x[1]['performance']['accuracy'])
        true_labels = best_result['true_labels']
        predictions = best_result['predictions']
        processed_predictions = [pred if pred is not None else 'Unknown' for pred in predictions]
        class_names = list(self.class_mapping['label_to_class'].values())
        cm = confusion_matrix(true_labels, processed_predictions, labels=class_names)
        confusion_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_rate = cm[i, j] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                    confusion_pairs.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'confusion_count': int(cm[i, j]),
                        'confusion_rate': float(confusion_rate)
                    })
        confusion_pairs.sort(key=lambda x: x['confusion_rate'], reverse=True)
        class_correlations = self._analyze_class_correlations()
        confusion_analysis = {
            'best_model': best_result_name,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'top_confusion_pairs': confusion_pairs[:10],
            'class_correlations': class_correlations
        }
        self.analysis_results['confusion_analysis'] = confusion_analysis
        self.logger.info("混淆模式分析完成")
        return confusion_analysis

    def _analyze_class_correlations(self):
        try:
            class_features = {}
            for sample in self.feature_data:
                class_name = sample['class_name']
                class_features.setdefault(class_name, []).append(sample['features'])
            class_centers = {}
            for class_name, features in class_features.items():
                class_centers[class_name] = np.mean(features, axis=0)
            class_names = list(class_centers.keys())
            correlation_matrix = np.zeros((len(class_names), len(class_names)))
            for i, class1 in enumerate(class_names):
                for j, class2 in enumerate(class_names):
                    if i != j:
                        corr = np.corrcoef(class_centers[class1], class_centers[class2])[0, 1]
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[i, j] = 1.0
            return {
                'class_names': class_names,
                'correlation_matrix': correlation_matrix.tolist()
            }
        except Exception as e:
            self.logger.warning(f"类别相关性分析失败: {str(e)}")
            return {'class_names': [], 'correlation_matrix': []}

    def analyze_confidence_distribution(self):
        self.logger.info("开始分析置信度分布...")
        confidence_analysis = {}
        for dict_name, result in self.matching_results.items():
            confidences = result['confidences']
            true_labels = result['true_labels']
            predictions = result['predictions']
            confidence_stats = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences)),
                'percentile_25': float(np.percentile(confidences, 25)),
                'percentile_75': float(np.percentile(confidences, 75))
            }
            correct_confidences = []
            incorrect_confidences = []
            for true_label, pred_label, conf in zip(true_labels, predictions, confidences):
                if pred_label == true_label:
                    correct_confidences.append(conf)
                else:
                    incorrect_confidences.append(conf)
            confidence_by_accuracy = {
                'correct_predictions': {
                    'mean': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
                    'count': len(correct_confidences)
                },
                'incorrect_predictions': {
                    'mean': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
                    'count': len(incorrect_confidences)
                }
            }
            confidence_analysis[dict_name] = {
                'overall_stats': confidence_stats,
                'by_accuracy': confidence_by_accuracy,
                'distribution': confidences
            }
        self.analysis_results['confidence_analysis'] = confidence_analysis
        self.logger.info("置信度分布分析完成")
        return confidence_analysis

    def generate_comprehensive_visualization(self):
        self.logger.info("生成综合可视化分析...")
        try:
            fig = plt.figure(figsize=(20, 24))
            ax1 = plt.subplot(4, 2, 1)
            self._plot_accuracy_comparison(ax1)
            ax2 = plt.subplot(4, 2, 2)
            self._plot_feature_performance(ax2)
            ax3 = plt.subplot(4, 2, 3)
            self._plot_confusion_heatmap(ax3)
            ax4 = plt.subplot(4, 2, 4)
            self._plot_confidence_distribution(ax4)
            ax5 = plt.subplot(4, 2, 5, projection='polar')
            self._plot_class_sensitivity_radar(ax5)
            ax6 = plt.subplot(4, 2, 6)
            self._plot_performance_improvement(ax6)
            ax7 = plt.subplot(4, 2, 7)
            self._plot_precision_recall_curves(ax7)
            ax8 = plt.subplot(4, 2, 8)
            self._plot_error_analysis(ax8)
            plt.tight_layout()
            plot_file = os.path.join(self.output_dir, 'comprehensive_analysis.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"综合可视化保存到: {plot_file}")
        except Exception as e:
            self.logger.error(f"生成可视化失败: {str(e)}")

    def _plot_accuracy_comparison(self, ax):
        if 'accuracy_analysis' not in self.analysis_results:
            return
        accuracy_data = self.analysis_results['accuracy_analysis']
        dict_names = list(accuracy_data.keys())
        accuracies = [data['total_accuracy'] for data in accuracy_data.values()]
        bars = ax.bar(range(len(dict_names)), accuracies, color='skyblue', alpha=0.7)
        ax.set_title('各字典识别准确率对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('字典类型')
        ax.set_ylabel('准确率')
        ax.set_xticks(range(len(dict_names)))
        ax.set_xticklabels([name.replace('_', '\n') for name in dict_names], rotation=0, ha='center')
        ax.grid(True, alpha=0.3)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    def _plot_feature_performance(self, ax):
        if 'feature_comparison' not in self.analysis_results:
            return
        feature_data = self.analysis_results['feature_comparison']['single_feature_performance']
        feature_types = list(feature_data.keys())
        accuracies = [data['accuracy'] for data in feature_data.values()]
        confidences = [data['avg_confidence'] for data in feature_data.values()]
        x = np.arange(len(feature_types))
        width = 0.35
        bars1 = ax.bar(x - width/2, accuracies, width, label='准确率', color='lightcoral', alpha=0.7)
        bars2 = ax.bar(x + width/2, confidences, width, label='平均置信度', color='lightgreen', alpha=0.7)
        ax.set_title('单特征性能对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('特征类型')
        ax.set_ylabel('性能指标')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    def _plot_confusion_heatmap(self, ax):
        if 'confusion_analysis' not in self.analysis_results:
            return
        confusion_data = self.analysis_results['confusion_analysis']
        cm = np.array(confusion_data['confusion_matrix'])
        class_names = confusion_data['class_names']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.set_title(f'混淆矩阵热图\n({confusion_data["best_model"]})', fontsize=14, fontweight='bold')
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([name.replace('（', '\n（') for name in class_names], rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        thresh = cm_normalized.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=8)
        ax.set_ylabel('真实类别')
        ax.set_xlabel('预测类别')

    def _plot_confidence_distribution(self, ax):
        if 'confidence_analysis' not in self.analysis_results:
            return
        best_model = max(self.analysis_results['confidence_analysis'].items(),
                         key=lambda x: x[1]['overall_stats']['mean'])[0]
        confidence_data = self.analysis_results['confidence_analysis'][best_model]
        confidences = confidence_data['distribution']
        ax.hist(confidences, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title(f'置信度分布\n({best_model})', fontsize=14, fontweight='bold')
        ax.set_xlabel('置信度')
        ax.set_ylabel('频次')
        ax.axvline(confidence_data['overall_stats']['mean'], color='red', linestyle='--',
                  label=f'平均值: {confidence_data["overall_stats"]["mean"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_class_sensitivity_radar(self, ax):
        if 'feature_comparison' not in self.analysis_results:
            return
        feature_data = self.analysis_results['feature_comparison']['single_feature_performance']
        class_names = list(self.class_mapping['label_to_class'].values())
        angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
        angles += angles[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        colors = ['red', 'blue', 'green', 'orange']
        for i, (feature_type, data) in enumerate(feature_data.items()):
            values = [data['class_sensitivity'].get(class_name, 0) for class_name in class_names]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=feature_type, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([name.split('（')[0] for name in class_names])
        ax.set_ylim(0, 1)
        ax.set_title('各特征类别敏感性雷达图', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    def _plot_performance_improvement(self, ax):
        if 'feature_comparison' not in self.analysis_results:
            return
        feature_comp = self.analysis_results['feature_comparison']
        single_acc = [data['accuracy'] for data in feature_comp['single_feature_performance'].values()]
        fusion_acc = [data['accuracy'] for data in feature_comp['fusion_performance'].values()]
        single_names = list(feature_comp['single_feature_performance'].keys())
        fusion_names = list(feature_comp['fusion_performance'].keys())
        x1 = np.arange(len(single_names))
        x2 = np.arange(len(fusion_names)) + len(single_names) + 1
        bars1 = ax.bar(x1, single_acc, color='lightblue', alpha=0.7, label='单特征')
        bars2 = ax.bar(x2, fusion_acc, color='lightgreen', alpha=0.7, label='融合方法')
        ax.set_title('单特征 vs 融合方法性能对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('方法类型')
        ax.set_ylabel('准确率')
        ax.set_xticks(list(x1) + list(x2))
        ax.set_xticklabels(single_names + fusion_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    def _plot_precision_recall_curves(self, ax):
        if 'accuracy_analysis' not in self.analysis_results:
            return
        best_model = max(self.analysis_results['accuracy_analysis'].items(),
                         key=lambda x: x[1]['total_accuracy'])[0]
        acc_data = self.analysis_results['accuracy_analysis'][best_model]
        precision = acc_data['precision']
        recall = acc_data['recall']
        class_names = acc_data['class_names']
        x = np.arange(len(class_names))
        width = 0.35
        bars1 = ax.bar(x - width/2, precision, width, label='精确率', color='skyblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, recall, width, label='召回率', color='lightcoral', alpha=0.7)
        ax.set_title(f'各类别精确率与召回率\n({best_model})', fontsize=14, fontweight='bold')
        ax.set_xlabel('类别')
        ax.set_ylabel('性能指标')
        ax.set_xticks(x)
        ax.set_xticklabels([name.split('（')[0] for name in class_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    def _plot_error_analysis(self, ax):
        if 'confusion_analysis' not in self.analysis_results:
            return
        confusion_data = self.analysis_results['confusion_analysis']
        top_confusions = confusion_data['top_confusion_pairs'][:5]
        if not top_confusions:
            ax.text(0.5, 0.5, '无混淆数据', ha='center', va='center', transform=ax.transAxes)
            return
        labels = [f"{pair['true_class'].split('（')[0]}\n→\n{pair['predicted_class'].split('（')[0]}"
                 for pair in top_confusions]
        rates = [pair['confusion_rate'] for pair in top_confusions]
        bars = ax.bar(range(len(labels)), rates, color='salmon', alpha=0.7)
        ax.set_title('最易混淆的类别对', fontsize=14, fontweight='bold')
        ax.set_xlabel('混淆对')
        ax.set_ylabel('混淆率')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.grid(True, alpha=0.3)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

    def generate_detailed_report(self):
        self.logger.info("生成详细分析报告...")
        report_content = []
        report_content.append("# 鲸鱼叫声识别系统性能分析报告")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        report_content.extend(self._generate_executive_summary())
        report_content.extend(self._generate_accuracy_section())
        report_content.extend(self._generate_feature_comparison_section())
        report_content.extend(self._generate_confusion_section())
        report_content.extend(self._generate_confidence_section())
        report_content.extend(self._generate_conclusions())
        report_file = os.path.join(self.output_dir, 'performance_analysis_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        self.logger.info(f"详细报告保存到: {report_file}")
        json_file = os.path.join(self.output_dir, 'analysis_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self._convert_to_serializable(self.analysis_results), f,
                     ensure_ascii=False, indent=2)

    def _generate_executive_summary(self):
        content = []
        content.append("## 执行摘要")
        content.append("")
        if 'accuracy_analysis' in self.analysis_results:
            best_model = max(self.analysis_results['accuracy_analysis'].items(),
                             key=lambda x: x[1]['total_accuracy'])
            best_name, best_data = best_model
            content.append(f"### 关键发现")
            content.append("")
            content.append(f"- **最佳识别模型**: {best_name}")
            content.append(f"- **最高准确率**: {best_data['total_accuracy']:.4f} ({best_data['total_accuracy']*100:.2f}%)")
            content.append(f"- **加权F1分数**: {best_data['f1_weighted']:.4f}")
            content.append("")
            all_accuracies = [data['total_accuracy'] for data in self.analysis_results['accuracy_analysis'].values()]
            content.append(f"- **平均识别准确率**: {np.mean(all_accuracies):.4f}")
            content.append(f"- **性能标准差**: {np.std(all_accuracies):.4f}")
            content.append("")
        if 'feature_comparison' in self.analysis_results:
            feature_comp = self.analysis_results['feature_comparison']
            content.append(f"- **最佳单特征**: {feature_comp['best_single_feature']}")
            content.append(f"- **最佳融合方法**: {feature_comp['best_fusion_method']}")
            content.append("")
        return content

    def _generate_accuracy_section(self):
        content = []
        content.append("## 识别准确率分析")
        content.append("")
        if 'accuracy_analysis' not in self.analysis_results:
            content.append("无准确率分析数据。")
            return content
        accuracy_data = self.analysis_results['accuracy_analysis']
        content.append("### 总体性能对比")
        content.append("")
        content.append("| 模型 | 准确率 | 加权精确率 | 加权召回率 | 加权F1分数 |")
        content.append("|------|--------|------------|------------|------------|")
        for model_name, data in accuracy_data.items():
            content.append(f"| {model_name} | {data['total_accuracy']:.4f} | "
                           f"{data['precision_weighted']:.4f} | {data['recall_weighted']:.4f} | "
                           f"{data['f1_weighted']:.4f} |")
        content.append("")
        content.append("### 各类别性能详情")
        content.append("")
        best_model = max(accuracy_data.items(), key=lambda x: x[1]['total_accuracy'])
        best_name, best_data = best_model
        content.append(f"**基于最佳模型 ({best_name}) 的类别性能分析：**")
        content.append("")
        content.append("| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |")
        content.append("|------|--------|--------|--------|--------|")
        for i, class_name in enumerate(best_data['class_names']):
            content.append(f"| {class_name} | {best_data['precision'][i]:.4f} | "
                           f"{best_data['recall'][i]:.4f} | {best_data['f1_score'][i]:.4f} | "
                           f"{best_data['support'][i]} |")
        content.append("")
        return content

    def _generate_feature_comparison_section(self):
        content = []
        content.append("## 特征对比分析")
        content.append("")
        if 'feature_comparison' not in self.analysis_results:
            content.append("无特征对比分析数据。")
            return content
        feature_data = self.analysis_results['feature_comparison']
        content.append("### 单特征性能对比")
        content.append("")
        content.append("| 特征类型 | 准确率 | 平均置信度 | 相对复杂度 |")
        content.append("|----------|--------|------------|------------|")
        for feature_type, data in feature_data['single_feature_performance'].items():
            content.append(f"| {feature_type} | {data['accuracy']:.4f} | "
                           f"{data['avg_confidence']:.4f} | {data['processing_time']:.1f} |")
        content.append("")
        content.append("### 融合方法性能对比")
        content.append("")
        content.append("| 融合方法 | 准确率 | 相比最佳单特征提升 |")
        content.append("|----------|--------|--------------------|")
        for fusion_type, data in feature_data['fusion_performance'].items():
            improvement = data['improvement_over_best_single']
            improvement_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
            content.append(f"| {fusion_type} | {data['accuracy']:.4f} | {improvement_str} |")
        content.append("")
        content.append("### 特征对类别的敏感性分析")
        content.append("")
        for feature_type, data in feature_data['single_feature_performance'].items():
            content.append(f"**{feature_type}特征:**")
            for class_name, sensitivity in data['class_sensitivity'].items():
                content.append(f"- {class_name}: {sensitivity:.4f}")
            content.append("")
        return content

    def _generate_confusion_section(self):
        content = []
        content.append("## 混淆模式分析")
        content.append("")
        if 'confusion_analysis' not in self.analysis_results:
            content.append("无混淆分析数据。")
            return content
        confusion_data = self.analysis_results['confusion_analysis']
        content.append(f"**基于最佳模型 ({confusion_data['best_model']}) 的混淆分析：**")
        content.append("")
        content.append("### 最易混淆的类别对")
        content.append("")
        content.append("| 真实类别 | 预测类别 | 混淆次数 | 混淆率 |")
        content.append("|----------|----------|----------|--------|")
        for pair in confusion_data['top_confusion_pairs'][:10]:
            content.append(f"| {pair['true_class']} | {pair['predicted_class']} | "
                           f"{pair['confusion_count']} | {pair['confusion_rate']:.4f} |")
        content.append("")
        if 'class_correlations' in confusion_data and confusion_data['class_correlations']['class_names']:
            content.append("### 类别间相关性分析")
            content.append("")
            content.append("类别特征中心之间的相关性可以解释某些混淆模式：")
            content.append("")
            corr_data = confusion_data['class_correlations']
            class_names = corr_data['class_names']
            corr_matrix = np.array(corr_data['correlation_matrix'])
            high_corr_pairs = []
            for i in range(len(class_names)):
                for j in range(i+1, len(class_names)):
                    corr_value = corr_matrix[i, j]
                    if abs(corr_value) > 0.5:
                        high_corr_pairs.append((class_names[i], class_names[j], corr_value))
            if high_corr_pairs:
                content.append("**高相关性类别对:**")
                for class1, class2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                    content.append(f"- {class1} ↔ {class2}: {corr:.4f}")
                content.append("")
        return content

    def _generate_confidence_section(self):
        content = []
        content.append("## 置信度分析")
        content.append("")
        if 'confidence_analysis' not in self.analysis_results:
            content.append("无置信度分析数据。")
            return content
        confidence_data = self.analysis_results['confidence_analysis']
        content.append("### 各模型置信度统计")
        content.append("")
        content.append("| 模型 | 平均置信度 | 标准差 | 最小值 | 最大值 |")
        content.append("|------|------------|--------|--------|--------|")
        for model_name, data in confidence_data.items():
            stats = data['overall_stats']
            content.append(f"| {model_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                           f"{stats['min']:.4f} | {stats['max']:.4f} |")
        content.append("")
        content.append("### 正确与错误预测的置信度对比")
        content.append("")
        content.append("| 模型 | 正确预测置信度 | 错误预测置信度 | 置信度差异 |")
        content.append("|------|----------------|----------------|------------|")
        for model_name, data in confidence_data.items():
            correct_conf = data['by_accuracy']['correct_predictions']['mean']
            incorrect_conf = data['by_accuracy']['incorrect_predictions']['mean']
            diff = correct_conf - incorrect_conf
            content.append(f"| {model_name} | {correct_conf:.4f} | {incorrect_conf:.4f} | {diff:.4f} |")
        content.append("")
        return content

    def _generate_conclusions(self):
        content = []
        content.append("## 结论与建议")
        content.append("")
        if 'accuracy_analysis' in self.analysis_results:
            best_accuracy = max(data['total_accuracy']
                                for data in self.analysis_results['accuracy_analysis'].values())
            content.append("### 主要结论")
            content.append("")
            if best_accuracy > 0.9:
                content.append("1. **识别效果优秀**: 最佳模型达到了非常高的识别准确率，可应用中。")
            elif best_accuracy > 0.7:
                content.append("1. **识别效果良好**: 系统表现出较好的识别能力，但仍有优化空间。")
            else:
                content.append("1. **识别效果有待提高**: 当前系统准确率较低，需要进一步优化。")
            content.append("")
        if 'feature_comparison' in self.analysis_results:
            feature_comp = self.analysis_results['feature_comparison']
            best_single = feature_comp['best_single_feature']
            best_fusion = feature_comp['best_fusion_method']
            content.append(f"2. **特征选择**: ")
            content.append(f"   - 最有效的单一特征: {best_single}")
            content.append(f"   - 最佳融合方法: {best_fusion}")
            content.append("")
            fusion_improvement = max(data['improvement_over_best_single']
                                     for data in feature_comp['fusion_performance'].values())
            if fusion_improvement > 0.1:
                content.append("3. **融合策略有效**: 多特征融合显著提升了识别性能，采用融合方法。")
            else:
                content.append("3. **融合效果有限**: 特征融合带来的提升较小，可以考虑优化融合策略。")
            content.append("")
        content.append("### 改进建议")
        content.append("")
        content.append("1. **数据增强**: 考虑使用数据增强技术增加训练样本的多样性")
        content.append("2. **特征工程**: 探索更多有效的音频特征，如深度学习特征")
        content.append("3. **模型优化**: 尝试更先进的机器学习算法，如深度神经网络")
        content.append("4. **后处理**: 实现更智能的后处理和决策融合策略")
        content.append("5. **实时优化**: 针对应用场景优化计算效率")
        content.append("")
        content.append("### 应用建议")
        content.append("")
        if 'accuracy_analysis' in self.analysis_results:
            best_model_name = max(self.analysis_results['accuracy_analysis'].items(),
                                  key=lambda x: x[1]['total_accuracy'])[0]
            content.append(f"- **推荐部署模型**: {best_model_name}")
            content.append("- **置信度阈值**: 根据具体应用场景设置合适的置信度阈值")
            content.append("- **实时监控**: 在应用中持续监控模型性能并及时调整")
        content.append("")
        return content

    def _convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def run_complete_analysis(self):
        self.logger.info("开始运行完整性能分析...")
        self.analyze_accuracy_performance()
        self.analyze_feature_comparison()
        self.analyze_confusion_patterns()
        self.analyze_confidence_distribution()
        self.generate_comprehensive_visualization()
        self.generate_detailed_report()
        self.logger.info("完整性能分析完成")

    def print_summary(self):
        print("\n" + "="*80)
        print("          鲸鱼叫声识别系统性能分析摘要")
        print("="*80)
        if not self.analysis_results:
            print(" 没有分析结果")
            return
        if 'accuracy_analysis' in self.analysis_results:
            best_model = max(self.analysis_results['accuracy_analysis'].items(),
                             key=lambda x: x[1]['total_accuracy'])
            best_name, best_data = best_model
            print(f"\n 最佳性能模型:")
            print(f"   模型名称: {best_name}")
            print(f"   识别准确率: {best_data['total_accuracy']:.4f} ({best_data['total_accuracy']*100:.2f}%)")
            print(f"   加权F1分数: {best_data['f1_weighted']:.4f}")
            print(f"   加权精确率: {best_data['precision_weighted']:.4f}")
            print(f"   加权召回率: {best_data['recall_weighted']:.4f}")
        if 'feature_comparison' in self.analysis_results:
            feature_comp = self.analysis_results['feature_comparison']
            print(f"\n 特征性能对比:")
            print(f"   最佳单特征: {feature_comp['best_single_feature']}")
            print(f"   最佳融合方法: {feature_comp['best_fusion_method']}")
            single_best_acc = max(data['accuracy']
                                  for data in feature_comp['single_feature_performance'].values())
            fusion_best_acc = max(data['accuracy']
                                  for data in feature_comp['fusion_performance'].values())
            improvement = fusion_best_acc - single_best_acc
            print(f"   融合方法提升: {improvement:.4f} ({improvement*100:.2f}%)")
        if 'confusion_analysis' in self.analysis_results:
            confusion_data = self.analysis_results['confusion_analysis']
            top_confusion = confusion_data['top_confusion_pairs'][0] if confusion_data['top_confusion_pairs'] else None
            if top_confusion:
                print(f"\n 混淆分析:")
                print(f"   最易混淆类别对: {top_confusion['true_class']} → {top_confusion['predicted_class']}")
                print(f"   混淆率: {top_confusion['confusion_rate']:.4f}")
        if 'confidence_analysis' in self.analysis_results:
            best_conf_model = max(self.analysis_results['confidence_analysis'].items(),
                                  key=lambda x: x[1]['overall_stats']['mean'])
            best_conf_name, best_conf_data = best_conf_model
            print(f"\n 置信度分析:")
            print(f"   最高平均置信度: {best_conf_data['overall_stats']['mean']:.4f} ({best_conf_name})")
            print(f"   正确预测置信度: {best_conf_data['by_accuracy']['correct_predictions']['mean']:.4f}")
            print(f"   错误预测置信度: {best_conf_data['by_accuracy']['incorrect_predictions']['mean']:.4f}")
        print(f"\n 生成的分析文件:")
        print(f"   ✓ comprehensive_analysis.png - 综合可视化分析")
        print(f"   ✓ performance_analysis_report.md - 详细分析报告")
        print(f"   ✓ analysis_results.json - 分析结果数据")
        print(f"   ✓ analysis.log - 分析过程日志")
        print("\n" + "="*80)

def main():
    print("鲸鱼叫声识别项目 - 性能分析与评估模块")
    print("开始执行深度性能分析...")
    try:
        analyzer = PerformanceAnalyzer()
        if not analyzer.load_data():
            print(" 数据加载失败，请先运行匹配识别模块")
            return
        analyzer.run_complete_analysis()
        analyzer.print_summary()
        print(f"\n 性能分析完成！")
        print(f" 详细分析报告已生成，请查看 'performance_analysis' 文件夹")
    except Exception as e:
        print(f" 性能分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
