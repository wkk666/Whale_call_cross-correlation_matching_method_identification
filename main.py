import os
import sys
import time
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path

class WhaleVoiceRecognitionSystem:

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.start_time = None
        self.execution_log = []
        self._setup_logging()
        self.modules = [
            {
                'name': '数据预处理',
                'file': 'data_preprocessing.py',
                'description': '对原始鲸鱼叫声音频进行预处理',
                'output_dir': 'preprocessed_data',
                'required': True
            },
            {
                'name': '特征提取',
                'file': 'feature_extraction.py',
                'description': '提取多种听觉感知特征',
                'output_dir': 'extracted_features',
                'required': True
            },
            {
                'name': '字典构建',
                'file': 'correlation_dictionary.py',
                'description': '构建互相关字典',
                'output_dir': 'correlation_dictionaries',
                'required': True
            },
            {
                'name': '匹配识别',
                'file': 'cross_correlation_matching.py',
                'description': '执行互相关匹配识别',
                'output_dir': 'matching_results',
                'required': True
            },
            {
                'name': '性能分析',
                'file': 'performance_analysis.py',
                'description': '分析识别性能',
                'output_dir': 'performance_analysis',
                'required': True
            },
            {
                'name': '字典优化',
                'file': 'dictionary_optimization.py',
                'description': '优化字典提升性能',
                'output_dir': 'optimized_dictionaries',
                'required': False
            }
        ]
        self.config = {
            'data_path': 'C:\\Users\\86198\\Desktop\\我的毕设代码\\pythonProject1\\data',
            'skip_completed': True,
            'auto_continue': False,
            'generate_report': True,
            'cleanup_temp': False
        }

    def _setup_logging(self):
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'main_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def print_banner(self):
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    鲸鱼叫声识别系统                             ║
║              Whale Voice Recognition System                  ║
║                                                              ║
║   基于互相关字典的多特征融合鲸鱼叫声识别系统                          ║
║   支持6种鲸鱼类别的自动识别和分类                                  ║
║   集成预处理、特征提取、字典构建、识别匹配全流程                       ║
║                                                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def check_environment(self):
        self.logger.info("检查运行环境...")
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.logger.error(f"Python版本过低: {python_version.major}.{python_version.minor}, 需要3.8+")
            return False
        required_packages = {
            'numpy': 'numpy',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'librosa': 'librosa',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'tqdm': 'tqdm',
            'pandas': 'pandas'
        }
        missing_packages = []
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        if missing_packages:
            self.logger.error(f"缺少必要的Python包: {', '.join(missing_packages)}")
            self.logger.info("请运行: pip install numpy scipy scikit-learn librosa matplotlib seaborn tqdm pandas")
            print("\n🔧 安装指导:")
            print("请在命令行中运行以下命令安装缺少的包:")
            print("pip install numpy scipy scikit-learn librosa matplotlib seaborn tqdm pandas")
            print("\n如果使用conda环境，可以运行:")
            print("conda install numpy scipy scikit-learn librosa matplotlib seaborn tqdm pandas")
            print("\n如果权限不足，可以尝试:")
            print("pip install --user numpy scipy scikit-learn librosa matplotlib seaborn tqdm pandas")
            return False
        data_path = Path(self.config['data_path'])
        if not data_path.exists():
            self.logger.error(f"数据目录不存在: {data_path}")
            self.logger.info("请确保数据目录存在且包含6个鲸鱼类别的音频文件")
            return False
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if len(subdirs) != 6:
            self.logger.warning(f"数据目录包含{len(subdirs)}个子目录，预期为6个")
        self.logger.info(" 环境检查通过")
        return True

    def check_module_completion(self, module):
        output_dir = Path(module['output_dir'])
        if not output_dir.exists():
            return False
        key_files = {
            'preprocessed_data': ['preprocessed_data.pkl', 'class_mapping.json'],
            'extracted_features': ['extracted_features.pkl', 'feature_names.pkl'],
            'correlation_dictionaries': ['correlation_dictionaries.pkl'],
            'matching_results': ['matching_results.pkl', 'performance_summary.json'],
            'performance_analysis': ['analysis_results.json', 'performance_analysis_report.md'],
            'optimized_dictionaries': ['optimized_dictionaries.pkl', 'optimization_report.md']
        }
        required_files = key_files.get(module['output_dir'], [])
        for file_name in required_files:
            if not (output_dir / file_name).exists():
                return False
        return True

    def execute_module(self, module):
        module_name = module['name']
        module_file = module['file']
        self.logger.info(f"开始执行模块: {module_name}")
        if self.config['skip_completed'] and self.check_module_completion(module):
            self.logger.info(f"  模块 {module_name} 已完成，跳过执行")
            return True, "已完成，跳过"
        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, module_file],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800
            )
            end_time = time.time()
            execution_time = end_time - start_time
            if result.returncode == 0:
                self.logger.info(f" 模块 {module_name} 执行成功，耗时: {execution_time:.2f}秒")
                return True, f"成功 ({execution_time:.2f}s)"
            else:
                self.logger.error(f" 模块 {module_name} 执行失败")
                self.logger.error(f"错误输出: {result.stderr}")
                return False, f"失败: {result.stderr[:100]}..."
        except subprocess.TimeoutExpired:
            self.logger.error(f" 模块 {module_name} 执行超时")
            return False, "执行超时"
        except Exception as e:
            self.logger.error(f" 模块 {module_name} 执行异常: {str(e)}")
            return False, f"异常: {str(e)}"

    def show_execution_menu(self):
        print("\n" + "="*80)
        print("                        执行模式选择")
        print("="*80)
        print("1.  完整执行 - 运行所有模块")
        print("2.  选择执行 - 选择特定模块运行")
        print("3.  续行执行 - 从中断点继续")
        print("4.  状态检查 - 检查各模块完成状态")
        print("5.  查看帮助 - 显示详细说明")
        print("6.  退出程序")
        print("="*80)
        while True:
            try:
                choice = input("\n请选择执行模式 (1-6): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6']:
                    return int(choice)
                else:
                    print(" 无效选择，请输入1-6之间的数字")
            except KeyboardInterrupt:
                print("\n\n 用户中断，程序退出")
                sys.exit(0)
            except EOFError:
                print("\n\n 输入结束，程序退出")
                sys.exit(0)
            except:
                print(" 输入错误，请重新输入")

    def show_module_status(self):
        print("\n" + "="*80)
        print("                        模块完成状态")
        print("="*80)
        for i, module in enumerate(self.modules, 1):
            status = " 已完成" if self.check_module_completion(module) else " 未完成"
            required = "必需" if module['required'] else "可选"
            print(f"{i}. {module['name']:<12} | {status} | {required} | {module['description']}")
        print("="*80)

    def select_modules_to_run(self):
        self.show_module_status()
        print("\n请选择要执行的模块 (可多选，用逗号分隔，如: 1,3,5):")
        print("输入 'all' 执行所有模块，输入 'required' 执行所有必需模块")
        while True:
            try:
                user_input = input("请输入选择: ").strip().lower()
                if user_input == 'all':
                    return list(range(len(self.modules)))
                elif user_input == 'required':
                    return [i for i, module in enumerate(self.modules) if module['required']]
                else:
                    choices = [int(x.strip()) - 1 for x in user_input.split(',')]
                    valid_choices = [c for c in choices if 0 <= c < len(self.modules)]
                    if valid_choices:
                        return valid_choices
                    else:
                        print(" 无效选择，请重新输入")
            except KeyboardInterrupt:
                print("\n\n 用户中断，返回主菜单")
                return []
            except:
                print(" 输入格式错误，请重新输入")

    def execute_selected_modules(self, module_indices):
        if not module_indices:
            return
        selected_modules = [self.modules[i] for i in module_indices]
        print(f"\n将执行以下 {len(selected_modules)} 个模块:")
        for module in selected_modules:
            print(f"  • {module['name']} - {module['description']}")
        if not self._confirm_execution():
            return
        self.start_time = time.time()
        success_count = 0
        print("\n" + "="*80)
        print("                        开始执行模块")
        print("="*80)
        for i, module_index in enumerate(module_indices):
            module = self.modules[module_index]
            print(f"\n [{i+1}/{len(module_indices)}] {module['name']}")
            print("-" * 60)
            success, message = self.execute_module(module)
            self.execution_log.append({
                'module': module['name'],
                'success': success,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            if success:
                success_count += 1
                print(f" {module['name']} 完成: {message}")
            else:
                print(f" {module['name']} 失败: {message}")
                if module['required'] and not self.config['auto_continue']:
                    print(f"\n  必需模块 {module['name']} 执行失败")
                    if not self._ask_continue():
                        print(" 用户选择停止执行")
                        break
        total_time = time.time() - self.start_time
        self._print_execution_summary(success_count, len(selected_modules), total_time)
        if self.config['generate_report'] and success_count > 0:
            self._generate_final_report()
        if len(selected_modules) > 0:
            print("\n" + "="*80)
            input("  按回车键返回主菜单...")
            print()

    def _confirm_execution(self):
        try:
            confirm = input("\n确认执行? (y/N): ").strip().lower()
            return confirm in ['y', 'yes', '是']
        except KeyboardInterrupt:
            return False

    def _ask_continue(self):
        try:
            continue_choice = input("是否继续执行剩余模块? (y/N): ").strip().lower()
            return continue_choice in ['y', 'yes', '是']
        except KeyboardInterrupt:
            return False

    def _print_execution_summary(self, success_count, total_count, total_time):
        print("\n" + "="*80)
        print("                        执行结果总结")
        print("="*80)
        print(f" 执行统计:")
        print(f"   总模块数: {total_count}")
        print(f"   成功数量: {success_count}")
        print(f"   失败数量: {total_count - success_count}")
        print(f"   成功率: {success_count/total_count*100:.1f}%")
        print(f"   总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        print(f"\n 详细结果:")
        for log_entry in self.execution_log:
            status = "成功" if log_entry['success'] else "失败"
            print(f"   {status} {log_entry['module']}: {log_entry['message']}")
        print(f"\n 生成的目录:")
        for module in self.modules:
            output_dir = Path(module['output_dir'])
            if output_dir.exists():
                file_count = len(list(output_dir.glob('*')))
                print(f"   ✓ {module['output_dir']} ({file_count} 个文件)")
        print("="*80)

    def _generate_final_report(self):
        self.logger.info("生成最终项目报告...")
        try:
            report_content = []
            report_content.append("# 鲸鱼叫声识别系统执行报告")
            report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")
            success_count = sum(1 for log in self.execution_log if log['success'])
            total_count = len(self.execution_log)
            report_content.append("## 执行摘要")
            report_content.append("")
            report_content.append(f"- **执行模块数**: {total_count}")
            report_content.append(f"- **成功模块数**: {success_count}")
            report_content.append(f"- **成功率**: {success_count/total_count*100:.1f}%")
            if self.start_time:
                total_time = time.time() - self.start_time
                report_content.append(f"- **总执行时间**: {total_time:.2f}秒")
            report_content.append("")
            report_content.append("## 模块执行详情")
            report_content.append("")
            report_content.append("| 模块 | 状态 | 说明 |")
            report_content.append("|------|------|------|")
            for log_entry in self.execution_log:
                status = " 成功" if log_entry['success'] else " 失败"
                report_content.append(f"| {log_entry['module']} | {status} | {log_entry['message']} |")
            report_content.append("")
            self._add_performance_results_to_report(report_content)
            report_content.append("## 生成的文件和目录")
            report_content.append("")
            for module in self.modules:
                output_dir = Path(module['output_dir'])
                if output_dir.exists():
                    files = list(output_dir.glob('*'))
                    report_content.append(f"### {module['name']} ({module['output_dir']})")
                    for file_path in files:
                        file_size = file_path.stat().st_size / 1024
                        report_content.append(f"- `{file_path.name}` ({file_size:.1f} KB)")
                    report_content.append("")
            report_content.append("## 结果使用说明")
            report_content.append("")
            report_content.append("1. **预处理数据**: `preprocessed_data/` - 包含处理后的音频特征")
            report_content.append("2. **提取特征**: `extracted_features/` - 包含657维特征向量")
            report_content.append("3. **字典文件**: `correlation_dictionaries/` - 包含7种不同的识别字典")
            report_content.append("4. **识别结果**: `matching_results/` - 包含详细的识别性能数据")
            report_content.append("5. **性能分析**: `performance_analysis/` - 包含深度性能分析报告")
            report_content.append("6. **优化字典**: `optimized_dictionaries/` - 包含优化后的高性能字典")
            report_content.append("")
            report_file = Path('PROJECT_EXECUTION_REPORT.md')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            self.logger.info(f"最终报告保存到: {report_file}")
            print(f"\n 最终项目报告已生成: {report_file}")
        except Exception as e:
            self.logger.error(f"生成最终报告失败: {str(e)}")

    def _add_performance_results_to_report(self, report_content):
        try:
            analysis_file = Path('performance_analysis/analysis_results.json')
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                report_content.append("## 关键性能指标")
                report_content.append("")
                if 'accuracy_analysis' in analysis_data:
                    accuracy_data = analysis_data['accuracy_analysis']
                    best_model = max(accuracy_data.items(), key=lambda x: x[1]['total_accuracy'])
                    report_content.append(f"- **最佳识别模型**: {best_model[0]}")
                    report_content.append(f"- **最高识别准确率**: {best_model[1]['total_accuracy']:.4f} ({best_model[1]['total_accuracy']*100:.2f}%)")
                    report_content.append(f"- **加权F1分数**: {best_model[1]['f1_weighted']:.4f}")
                if 'feature_comparison' in analysis_data:
                    feature_data = analysis_data['feature_comparison']
                    report_content.append(f"- **最佳单特征**: {feature_data['best_single_feature']}")
                    report_content.append(f"- **最佳融合方法**: {feature_data['best_fusion_method']}")
                report_content.append("")
        except Exception as e:
            self.logger.warning(f"读取性能结果失败: {str(e)}")

    def show_help(self):
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                          系统帮助                            ║
╚══════════════════════════════════════════════════════════════╝

 系统概述:
   本系统是一个完整的鲸鱼叫声识别解决方案，支持从原始音频
   数据到最终识别结果的全流程自动化处理。

 模块说明:
   1. 数据预处理 - 音频预加重、分帧、加窗、归一化
   2. 特征提取 - 提取时域、频域、MFCC、色度等特征
   3. 字典构建 - 构建基于不同特征的互相关字典
   4. 匹配识别 - 执行互相关匹配和性能测试
   5. 性能分析 - 深度分析识别性能和混淆模式
   6. 字典优化 - 优化字典提升识别准确率

 执行模式:
   • 完整执行 - 推荐首次使用，运行全部流程
   • 选择执行 - 适合调试或重新运行特定模块
   • 续行执行 - 从之前中断的位置继续
   • 状态检查 - 查看各模块的完成状态

 预期结果:
   • 支持6种鲸鱼类别的自动识别
   • 最高识别准确率可达99%+
   • 提供详细的性能分析和优化建议
   • 生成完整的项目执行报告

 故障排除:
   • 确保数据目录包含6个鲸鱼类别的音频文件
   • 检查Python环境和必要库的安装
   • 查看日志文件获取详细错误信息
   • 可以跳过可选模块或重新运行失败模块


        """
        print(help_text)
        input("\n按回车键返回主菜单...")

    def run(self):
        try:
            self.print_banner()
            if not self.check_environment():
                print(" 环境检查失败，程序退出")
                return
            while True:
                choice = self.show_execution_menu()
                if choice == 1:
                    self.execute_selected_modules(list(range(len(self.modules))))
                elif choice == 2:
                    selected_indices = self.select_modules_to_run()
                    self.execute_selected_modules(selected_indices)
                elif choice == 3:
                    incomplete_indices = [
                        i for i, module in enumerate(self.modules)
                        if not self.check_module_completion(module)
                    ]
                    if incomplete_indices:
                        print(f"发现 {len(incomplete_indices)} 个未完成的模块")
                        self.execute_selected_modules(incomplete_indices)
                    else:
                        print(" 所有模块已完成！")
                elif choice == 4:
                    self.show_module_status()
                    input("\n按回车键返回主菜单...")
                elif choice == 5:
                    self.show_help()
                elif choice == 6:
                    print("\n 感谢使用鲸鱼叫声识别系统，再见！")
                    break
        except KeyboardInterrupt:
            print("\n\n 用户中断，程序退出")
        except Exception as e:
            self.logger.error(f"程序执行异常: {str(e)}")
            print(f"\n 程序执行异常: {str(e)}")

def main():
    system = WhaleVoiceRecognitionSystem()
    system.run()

if __name__ == "__main__":
    main()
