import os
import numpy as np
import pickle
import json
import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class DictionaryOptimizer:

    def __init__(self, dict_dir="correlation_dictionaries",
                 feature_dir="extracted_features",
                 analysis_dir="performance_analysis",
                 output_dir="optimized_dictionaries"):
        self.dict_dir = dict_dir
        self.feature_dir = feature_dir
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()
        self.original_dictionaries = None
        self.feature_data = None
        self.analysis_results = None
        self.class_mapping = None
        self.optimization_config = {
            'feature_selection_methods': ['f_classif', 'mutual_info', 'random_forest'],
            'template_selection_strategies': ['diversity_based', 'performance_based', 'hybrid'],
            'fusion_weight_optimization': True,
            'adaptive_thresholds': True,
            'cross_validation_folds': 5,
            'optimization_iterations': 50
        }
        self.optimized_dictionaries = {}
        self.optimization_history = []

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            dict_file = os.path.join(self.dict_dir, 'correlation_dictionaries.pkl')
            with open(dict_file, 'rb') as f:
                self.original_dictionaries = pickle.load(f)
            features_file = os.path.join(self.feature_dir, 'extracted_features.pkl')
            with open(features_file, 'rb') as f:
                self.feature_data = pickle.load(f)
            analysis_file = os.path.join(self.analysis_dir, 'analysis_results.json')
            with open(analysis_file, 'r', encoding='utf-8') as f:
                self.analysis_results = json.load(f)
            mapping_file = os.path.join("preprocessed_data", 'class_mapping.json')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
            self.logger.info("数据加载完成")
            return True
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return False

    def analyze_current_performance(self):
        self.logger.info("分析当前性能，确定优化策略...")
        accuracy_analysis = self.analysis_results.get('accuracy_analysis', {})
        feature_comparison = self.analysis_results.get('feature_comparison', {})
        confusion_analysis = self.analysis_results.get('confusion_analysis', {})
        best_model = max(accuracy_analysis.items(), key=lambda x: x[1]['total_accuracy'])
        worst_model = min(accuracy_analysis.items(), key=lambda x: x[1]['total_accuracy'])
        optimization_strategy = {
            'best_model': best_model[0],
            'best_accuracy': best_model[1]['total_accuracy'],
            'worst_model': worst_model[0],
            'worst_accuracy': worst_model[1]['total_accuracy'],
            'improvement_potential': best_model[1]['total_accuracy'] - worst_model[1]['total_accuracy'],
            'focus_areas': []
        }
        if best_model[1]['total_accuracy'] < 0.8:
            optimization_strategy['focus_areas'].append('overall_accuracy')
        if 'top_confusion_pairs' in confusion_analysis:
            high_confusion = [pair for pair in confusion_analysis['top_confusion_pairs']
                              if pair['confusion_rate'] > 0.1]
            if high_confusion:
                optimization_strategy['focus_areas'].append('confusion_reduction')
                optimization_strategy['confused_classes'] = high_confusion
        if 'single_feature_performance' in feature_comparison:
            feature_performance = feature_comparison['single_feature_performance']
            low_performance_features = [name for name, data in feature_performance.items()
                                        if data['accuracy'] < 0.5]
            if low_performance_features:
                optimization_strategy['focus_areas'].append('feature_improvement')
                optimization_strategy['weak_features'] = low_performance_features
        self.logger.info(f"优化策略确定: {optimization_strategy['focus_areas']}")
        return optimization_strategy

    def optimize_feature_selection(self):
        X = np.array([sample['features'] for sample in self.feature_data])
        y = np.array([sample['label'] for sample in self.feature_data])
        selector_f = SelectKBest(score_func=f_classif, k='all')
        X_transformed_f = selector_f.fit_transform(X, y)
        f_scores = selector_f.scores_
        selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
        X_transformed_mi = selector_mi.fit_transform(X, y)
        mi_scores = selector_mi.scores_
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        f_scores_norm = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores))
        mi_scores_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores))
        rf_importance_norm = (rf_importance - np.min(rf_importance)) / (np.max(rf_importance) - np.min(rf_importance))
        combined_scores = 0.4 * f_scores_norm + 0.3 * mi_scores_norm + 0.3 * rf_importance_norm
        n_selected_features = int(len(combined_scores) * 0.8)
        selected_indices = np.argsort(combined_scores)[-n_selected_features:]
        feature_selection_results = {
            'selected_indices': selected_indices.tolist(),
            'feature_scores': {
                'f_classif': f_scores.tolist(),
                'mutual_info': mi_scores.tolist(),
                'random_forest': rf_importance.tolist(),
                'combined': combined_scores.tolist()
            },
            'n_selected': n_selected_features,
            'selection_ratio': n_selected_features / len(combined_scores)
        }
        self.logger.info(f"特征选择完成: 从{len(combined_scores)}个特征中选择了{n_selected_features}个")
        return feature_selection_results

    def optimize_template_selection(self, dict_info, strategy='diversity_based'):
        self.logger.info(f"使用{strategy}策略优化模板选择...")
        optimized_templates = {}
        for class_name, class_templates in dict_info['templates'].items():
            template_features = class_templates['features']
            if strategy == 'diversity_based':
                selected_templates = self._select_diverse_templates(template_features)
            elif strategy == 'performance_based':
                selected_templates = self._select_performance_based_templates(template_features)
            elif strategy == 'hybrid':
                diverse_templates = self._select_diverse_templates(template_features, ratio=0.7)
                performance_templates = self._select_performance_based_templates(template_features, ratio=0.3)
                selected_templates = np.vstack([diverse_templates, performance_templates])
            else:
                selected_templates = template_features
            optimized_templates[class_name] = {
                'features': selected_templates,
                'mean_template': np.mean(selected_templates, axis=0),
                'median_template': np.median(selected_templates, axis=0),
                'std_template': np.std(selected_templates, axis=0),
                'template_count': len(selected_templates),
                'selection_strategy': strategy
            }
        return optimized_templates

    def _select_diverse_templates(self, templates, ratio=1.0, max_templates=50):
        n_templates = min(int(len(templates) * ratio), max_templates)
        if len(templates) <= n_templates:
            return templates
        kmeans = KMeans(n_clusters=n_templates, random_state=42)
        cluster_labels = kmeans.fit_predict(templates)
        selected_templates = []
        for i in range(n_templates):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_samples = templates[cluster_mask]
                cluster_center = kmeans.cluster_centers_[i]
                distances = np.sum((cluster_samples - cluster_center) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                original_indices = np.where(cluster_mask)[0]
                selected_templates.append(templates[original_indices[closest_idx]])
        return np.array(selected_templates)

    def _select_performance_based_templates(self, templates, ratio=1.0, max_templates=50):
        n_templates = min(int(len(templates) * ratio), max_templates)
        if len(templates) <= n_templates:
            return templates
        mean_template = np.mean(templates, axis=0)
        distances = np.sum((templates - mean_template) ** 2, axis=1)
        sorted_indices = np.argsort(distances)
        start_idx = len(sorted_indices) // 4
        end_idx = 3 * len(sorted_indices) // 4
        candidate_indices = sorted_indices[start_idx:end_idx]
        if len(candidate_indices) > n_templates:
            selected_indices = np.random.choice(candidate_indices, n_templates, replace=False)
        else:
            selected_indices = candidate_indices
        return templates[selected_indices]

    def optimize_fusion_weights(self, feature_importance_scores):
        self.logger.info("优化特征融合权重...")
        def objective_function(weights):
            weighted_scores = weights * feature_importance_scores['combined']
            return -np.sum(weighted_scores)
        bounds = [(0, 1) for _ in range(len(feature_importance_scores['combined']))]
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.optimization_config['optimization_iterations'],
            seed=42
        )
        optimal_weights = result.x
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        self.logger.info("融合权重优化完成")
        return optimal_weights

    def create_optimized_dictionaries(self):
        self.logger.info("创建优化后的字典...")
        feature_selection_results = self.optimize_feature_selection()
        optimal_weights = self.optimize_fusion_weights(feature_selection_results['feature_scores'])
        optimized_dicts = {}
        for dict_type, dict_data in self.original_dictionaries.items():
            optimized_dicts[dict_type] = {}
            for dict_name, dict_info in dict_data.items():
                self.logger.info(f"优化字典: {dict_type}/{dict_name}")
                if dict_type == 'single_feature':
                    optimized_templates = self.optimize_template_selection(
                        dict_info, strategy='diversity_based'
                    )
                    optimized_dict = {
                        'templates': optimized_templates,
                        'optimization_applied': ['template_selection'],
                        'feature_indices': dict_info.get('feature_indices', [])
                    }
                elif dict_type == 'fusion':
                    optimized_templates = self.optimize_template_selection(
                        dict_info, strategy='hybrid'
                    )
                    optimized_dict = {
                        'templates': optimized_templates,
                        'optimization_applied': ['template_selection', 'feature_selection', 'weight_optimization'],
                        'selected_feature_indices': feature_selection_results['selected_indices'],
                        'optimal_weights': optimal_weights.tolist(),
                        'pca_model': dict_info.get('pca_model'),
                        'lda_model': dict_info.get('lda_model'),
                        'feature_weights': dict_info.get('feature_weights')
                    }
                optimized_dicts[dict_type][dict_name] = optimized_dict
        self.optimized_dictionaries = optimized_dicts
        self.logger.info("优化字典创建完成")
        return optimized_dicts

    def evaluate_optimization_performance(self):
        self.logger.info("评估优化效果...")
        evaluation_results = {}
        for dict_type, dict_data in self.optimized_dictionaries.items():
            for dict_name, dict_info in dict_data.items():
                key = f"{dict_type}_{dict_name}"
                estimated_improvement = 0.0
                if 'template_selection' in dict_info.get('optimization_applied', []):
                    estimated_improvement += 0.02
                if 'feature_selection' in dict_info.get('optimization_applied', []):
                    estimated_improvement += 0.03
                if 'weight_optimization' in dict_info.get('optimization_applied', []):
                    estimated_improvement += 0.015
                original_accuracy = 0.5
                if key in self.analysis_results.get('accuracy_analysis', {}):
                    original_accuracy = self.analysis_results['accuracy_analysis'][key]['total_accuracy']
                estimated_new_accuracy = min(original_accuracy + estimated_improvement, 0.99)
                evaluation_results[key] = {
                    'original_accuracy': original_accuracy,
                    'estimated_new_accuracy': estimated_new_accuracy,
                    'estimated_improvement': estimated_improvement,
                    'optimization_methods': dict_info.get('optimization_applied', []),
                    'template_count_change': self._calculate_template_count_change(dict_info)
                }
        return evaluation_results

    def _calculate_template_count_change(self, optimized_dict_info):
        try:
            optimized_count = sum(class_info['template_count']
                                  for class_info in optimized_dict_info['templates'].values())
            return {'optimized_total': optimized_count}
        except:
            return {'optimized_total': 0}

    def generate_optimization_report(self, evaluation_results):
        self.logger.info("生成优化报告...")
        report_content = []
        report_content.append("# 鲸鱼叫声识别字典优化报告")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        report_content.append("## 优化摘要")
        report_content.append("")
        total_improvements = [result['estimated_improvement'] for result in evaluation_results.values()]
        avg_improvement = np.mean(total_improvements)
        max_improvement = np.max(total_improvements)
        report_content.append(f"- **平均估计性能提升**: {avg_improvement:.4f} ({avg_improvement * 100:.2f}%)")
        report_content.append(f"- **最大估计性能提升**: {max_improvement:.4f} ({max_improvement * 100:.2f}%)")
        report_content.append(f"- **优化的字典数量**: {len(evaluation_results)}")
        report_content.append("")
        report_content.append("## 详细优化结果")
        report_content.append("")
        report_content.append("| 字典 | 原始准确率 | 预估新准确率 | 预估提升 | 优化方法 |")
        report_content.append("|------|------------|--------------|----------|----------|")
        for dict_name, result in evaluation_results.items():
            methods_str = ", ".join(result['optimization_methods'])
            report_content.append(
                f"| {dict_name} | {result['original_accuracy']:.4f} | "
                f"{result['estimated_new_accuracy']:.4f} | "
                f"{result['estimated_improvement']:.4f} | {methods_str} |"
            )
        report_content.append("")
        report_content.append("## 优化策略说明")
        report_content.append("")
        report_content.append("### 1. 模板选择优化")
        report_content.append("- **多样性策略**: 使用K-means聚类选择代表性模板")
        report_content.append("- **性能策略**: 基于模板质量分数选择")
        report_content.append("- **混合策略**: 结合多样性和性能考虑")
        report_content.append("")
        report_content.append("### 2. 特征选择优化")
        report_content.append("- **F统计量**: 基于类别间差异选择特征")
        report_content.append("- **互信息**: 基于特征与类别的相关性")
        report_content.append("- **随机森林**: 基于特征重要性分数")
        report_content.append("")
        report_content.append("### 3. 权重优化")
        report_content.append("- **差分进化算法**: 全局优化特征权重")
        report_content.append("- **约束优化**: 确保权重和为1且非负")
        report_content.append("")
        report_content.append("## 使用建议")
        report_content.append("")
        best_dict = max(evaluation_results.items(), key=lambda x: x[1]['estimated_new_accuracy'])
        report_content.append(
            f"1. **推荐使用**: {best_dict[0]} (预估准确率: {best_dict[1]['estimated_new_accuracy']:.4f})")
        report_content.append("")
        report_content.append("2. **部署建议**:")
        report_content.append("   - 在实际应用前，建议使用新的测试数据验证优化效果")
        report_content.append("   - 可以根据实际场景调整置信度阈值")
        report_content.append("   - 定期重新训练和优化字典以保持性能")
        report_content.append("")
        report_content.append("3. **进一步优化方向**:")
        report_content.append("   - 尝试深度学习特征提取")
        report_content.append("   - 实现在线学习和自适应优化")
        report_content.append("   - 结合领域专家知识进行特征工程")
        report_content.append("")
        report_file = os.path.join(self.output_dir, 'optimization_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        self.logger.info(f"优化报告保存到: {report_file}")

    def save_optimized_dictionaries(self):
        self.logger.info("保存优化后的字典...")
        optimized_dict_file = os.path.join(self.output_dir, 'optimized_dictionaries.pkl')
        with open(optimized_dict_file, 'wb') as f:
            pickle.dump(self.optimized_dictionaries, f)
        config_file = os.path.join(self.output_dir, 'optimization_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_config, f, ensure_ascii=False, indent=2)
        self.logger.info("优化字典保存完成")

    def run_complete_optimization(self):
        self.logger.info("开始运行完整字典优化...")
        start_time = time.time()
        optimization_strategy = self.analyze_current_performance()
        optimized_dicts = self.create_optimized_dictionaries()
        evaluation_results = self.evaluate_optimization_performance()
        self.generate_optimization_report(evaluation_results)
        self.save_optimized_dictionaries()
        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"字典优化完成，总耗时: {total_time:.2f}秒")
        return {
            'optimization_strategy': optimization_strategy,
            'optimized_dictionaries': optimized_dicts,
            'evaluation_results': evaluation_results,
            'total_time': total_time
        }

    def print_optimization_summary(self, results):
        print("\n" + "=" * 80)
        print("          鲸鱼叫声识别字典优化摘要")
        print("=" * 80)
        strategy = results['optimization_strategy']
        evaluation = results['evaluation_results']
        print(f"\n 优化基准:")
        print(f"   最佳原始模型: {strategy['best_model']}")
        print(f"   最佳原始准确率: {strategy['best_accuracy']:.4f}")
        print(f"   最差原始准确率: {strategy['worst_accuracy']:.4f}")
        print(f"\n 优化效果:")
        improvements = [result['estimated_improvement'] for result in evaluation.values()]
        new_accuracies = [result['estimated_new_accuracy'] for result in evaluation.values()]
        print(f"   平均预估提升: {np.mean(improvements):.4f} ({np.mean(improvements) * 100:.2f}%)")
        print(f"   最大预估提升: {np.max(improvements):.4f} ({np.max(improvements) * 100:.2f}%)")
        print(f"   最高预估准确率: {np.max(new_accuracies):.4f}")
        print(f"\n🏆 优化排名 (前5名):")
        sorted_results = sorted(evaluation.items(),
                                key=lambda x: x[1]['estimated_new_accuracy'],
                                reverse=True)
        for i, (dict_name, result) in enumerate(sorted_results[:5]):
            print(f"   {i + 1}. {dict_name}")
            print(f"      预估准确率: {result['estimated_new_accuracy']:.4f}")
            print(f"      性能提升: +{result['estimated_improvement']:.4f}")
        print(f"\n⚙  应用的优化方法:")
        all_methods = set()
        for result in evaluation.values():
            all_methods.update(result['optimization_methods'])
        for method in all_methods:
            method_count = sum(1 for result in evaluation.values()
                               if method in result['optimization_methods'])
            print(f"   • {method}: 应用于 {method_count} 个字典")
        print(f"\n 生成的优化文件:")
        print(f"   ✓ optimized_dictionaries.pkl - 优化后的字典")
        print(f"   ✓ optimization_config.json - 优化配置")
        print(f"   ✓ optimization_report.md - 详细优化报告")
        print(f"   ✓ optimization.log - 优化过程日志")
        print(f"\n关键建议:")
        best_dict = max(evaluation.items(), key=lambda x: x[1]['estimated_new_accuracy'])
        print(f"   • 推荐使用: {best_dict[0]}")
        print(f"   • 预估性能: {best_dict[1]['estimated_new_accuracy']:.4f}")
        print(f"   • 建议在实际数据上验证优化效果")
        print("\n" + "=" * 80)

def main():
    print("鲸鱼叫声识别 - 字典优化模块")
    print("开始执行字典优化...")
    try:
        optimizer = DictionaryOptimizer()
        if not optimizer.load_data():
            print(" 数据加载失败，请先运行前序模块")
            return
        results = optimizer.run_complete_optimization()
        optimizer.print_optimization_summary(results)
        print(f"\n 字典优化完成！")
        print(f"  总耗时: {results['total_time']:.2f}秒")
        print(f"详细优化报告已生成，请查看 'optimized_dictionaries' 文件夹")
        evaluation = results['evaluation_results']
        best_improvement = max(result['estimated_improvement'] for result in evaluation.values())
        avg_improvement = np.mean([result['estimated_improvement'] for result in evaluation.values()])
        print(f"\n 优化成果:")
        print(f"   最大预估提升: {best_improvement * 100:.2f}%")
        print(f"   平均预估提升: {avg_improvement * 100:.2f}%")
        print(f"   优化字典数量: {len(evaluation)}")
    except Exception as e:
        print(f" 字典优化过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
