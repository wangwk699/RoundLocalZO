import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoModelForCausalLM
from collections import defaultdict
import seaborn as sns

# 设置随机种子和绘图风格
torch.manual_seed(42)
np.random.seed(42)
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class WeightDistributionAnalyzer:
    """LLM 权重分布分析器"""
    
    def __init__(self, model_names):
        """
        Args:
            model_names: dict, 模型名称映射，如 {'OPT-1.3B': 'facebook/opt-1.3b', ...}
        """
        self.model_names = model_names
        self.weight_stats = {}
        self.boundary_stats = {}
        
    def load_model(self, model_key):
        """加载模型"""
        print(f"Loading {model_key}...")
        model_path = self.model_names[model_key]
        
        # --- 修改开始 ---
        # 1. 检查 CUDA 是否可用
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # 2. 设置 dtype 为 float16 以节省显存 (7B 模型 FP32 需~28GB, FP16 需~14GB)
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,          # 修改：根据设备选择精度
            device_map="auto",          # 修改：自动分配到 GPU
            low_cpu_mem_usage=True,
            # max_memory={0: "20GB"}    # 可选：限制单卡显存使用，防止 OOM
        )
        # --- 修改结束 ---
        
        print(f"✓ {model_key} loaded successfully on {device}")
        return model
    
    def extract_weights(self, model):
        """提取所有线性层权重"""
        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 确保权重在 CPU 上以便转换为 numpy (scipy/numpy 不支持 CUDA tensor)
                weight = module.weight.data.flatten().cpu().numpy() 
                weights[name] = weight
        return weights
    
    def compute_boundary_coverage(self, weights, bit_width=4):
        """
        计算边界覆盖率 - 权重靠近量化边界的比例
        
        Args:
            weights: 权重数组
            bit_width: 量化位宽
            epsilon: 边界邻域范围
            
        Returns:
            boundary_coverage: 边界覆盖率
            boundary_distances: 到最近边界的距离分布
        """
        # 对称均匀量化的边界点 (半整数)
        # 对于 4-bit, 范围是 [-8, 7], 边界点在 ±0.5, ±1.5, ..., ±7.5
        max_int = 2 ** (bit_width - 1)  # 8 for 4-bit
        boundaries = np.array([i + 0.5 for i in range(-max_int, max_int)])
        
        # 计算每个权重到最近边界的距离
        distances = np.zeros_like(weights)
        for i, w in enumerate(weights):
            dists = np.abs(w - boundaries)
            distances[i] = np.min(dists)
        
        # 边界邻域阈值 (量化步长的 10%)
        epsilon = 0.1  # 可调整
        
        # 边界覆盖率
        boundary_coverage = np.mean(distances < epsilon)
        
        return boundary_coverage, distances
    
    def compute_statistics(self, weights_dict):
        """计算权重的统计指标"""
        all_weights = np.concatenate(list(weights_dict.values()))
        
        stats_dict = {
            'mean': np.mean(all_weights),
            'std': np.std(all_weights),
            'skewness': stats.skew(all_weights),
            'kurtosis': stats.kurtosis(all_weights),  # 超额峰度
            'min': np.min(all_weights),
            'max': np.max(all_weights),
            'median': np.median(all_weights),
            'q1': np.percentile(all_weights, 25),
            'q3': np.percentile(all_weights, 75),
            'outlier_ratio': self._compute_outlier_ratio(all_weights),
        }
        
        # 边界覆盖率
        boundary_cov, boundary_dists = self.compute_boundary_coverage(all_weights)
        stats_dict['boundary_coverage'] = boundary_cov
        stats_dict['boundary_dist_mean'] = np.mean(boundary_dists)
        stats_dict['boundary_dist_std'] = np.std(boundary_dists)
        
        # 层间方差分析
        layer_stds = [np.std(w) for w in weights_dict.values()]
        stats_dict['layer_std_mean'] = np.mean(layer_stds)
        stats_dict['layer_std_cv'] = np.std(layer_stds) / (np.mean(layer_stds) + 1e-8)  # 变异系数
        
        return stats_dict
    
    def _compute_outlier_ratio(self, weights, threshold=3.0):
        """计算离群值比例 (超过 3 倍标准差)"""
        mean, std = np.mean(weights), np.std(weights)
        outlier_mask = np.abs(weights - mean) > threshold * std
        return np.mean(outlier_mask)
    
    def analyze_all_models(self):
        """分析所有模型"""
        for model_key, model_path in self.model_names.items():
            print(f"\n{'='*60}")
            print(f"Analyzing {model_key}")
            print(f'{'='*60}')
            
            # 加载模型
            model = self.load_model(model_key)
            
            # 提取权重
            weights = self.extract_weights(model)
            print(f"Extracted {len(weights)} linear layers")
            
            # 计算统计指标
            stats_dict = self.compute_statistics(weights)
            self.weight_stats[model_key] = stats_dict
            
            # 打印结果
            self._print_stats(model_key, stats_dict)
            
            # 清理内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 强制清理 CUDA 缓存
                torch.cuda.synchronize()  # 确保操作完成            
        
        return self.weight_stats
    
    def _print_stats(self, model_key, stats_dict):
        """打印统计结果"""
        print(f"\n📊 {model_key} Weight Statistics:")
        print(f"  Mean:              {stats_dict['mean']:.6f}")
        print(f"  Std:               {stats_dict['std']:.6f}")
        print(f"  Skewness:          {stats_dict['skewness']:.4f}")
        print(f"  Kurtosis (excess): {stats_dict['kurtosis']:.4f}")
        print(f"  Outlier Ratio:     {stats_dict['outlier_ratio']:.4%}")
        print(f"  Boundary Coverage: {stats_dict['boundary_coverage']:.4%}")
        print(f"  Layer Std CV:      {stats_dict['layer_std_cv']:.4f}")
    
    def plot_comparison(self, save_path='weight_distribution_comparison.png'):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.weight_stats)))
        
        # 1. 峰度对比
        ax = axes[0, 0]
        models = list(self.weight_stats.keys())
        kurtosis_vals = [self.weight_stats[m]['kurtosis'] for m in models]
        bars = ax.bar(models, kurtosis_vals, color=colors)
        ax.set_ylabel('Kurtosis (Excess)')
        ax.set_title('Weight Distribution Kurtosis\n(Higher = More Outliers)', fontsize=12)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Normal')
        ax.legend()
        for bar, val in zip(bars, kurtosis_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 边界覆盖率对比
        ax = axes[0, 1]
        boundary_vals = [self.weight_stats[m]['boundary_coverage'] for m in models]
        bars = ax.bar(models, boundary_vals, color=colors)
        ax.set_ylabel('Boundary Coverage Rate')
        ax.set_title('Weights Near Quantization Boundaries\n(Higher = More Sensitive)', fontsize=12)
        for bar, val in zip(bars, boundary_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.2%}', ha='center', va='bottom', fontsize=10)
        
        # 3. 层间方差变异系数
        ax = axes[0, 2]
        cv_vals = [self.weight_stats[m]['layer_std_cv'] for m in models]
        bars = ax.bar(models, cv_vals, color=colors)
        ax.set_ylabel('Layer Std Coefficient of Variation')
        ax.set_title('Layer-wise Variance Heterogeneity\n(Higher = More Variation)', fontsize=12)
        for bar, val in zip(bars, cv_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 偏度对比
        ax = axes[1, 0]
        skew_vals = [self.weight_stats[m]['skewness'] for m in models]
        bars = ax.bar(models, skew_vals, color=colors)
        ax.set_ylabel('Skewness')
        ax.set_title('Weight Distribution Skewness\n(0 = Symmetric)', fontsize=12)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, skew_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 5. 离群值比例
        ax = axes[1, 1]
        outlier_vals = [self.weight_stats[m]['outlier_ratio'] for m in models]
        bars = ax.bar(models, outlier_vals, color=colors)
        ax.set_ylabel('Outlier Ratio')
        ax.set_title('Weight Outliers (>3σ)\n(Higher = More Extreme Values)', fontsize=12)
        for bar, val in zip(bars, outlier_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.2%}', ha='center', va='bottom', fontsize=10)
        
        # 6. 边界距离均值
        ax = axes[1, 2]
        dist_vals = [self.weight_stats[m]['boundary_dist_mean'] for m in models]
        bars = ax.bar(models, dist_vals, color=colors)
        ax.set_ylabel('Mean Distance to Boundary')
        ax.set_title('Average Distance to Quantization Boundary\n(Lower = Closer to Boundaries)', fontsize=12)
        for bar, val in zip(bars, dist_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to {save_path}")
        plt.show()
    
    def plot_weight_histograms(self, model_key, save_path=None):
        """绘制单个模型的权重直方图"""
        if model_key not in self.weight_stats:
            print(f"Error: {model_key} not analyzed yet")
            return
        
        # 需要重新加载模型获取权重
        model = self.load_model(model_key)
        weights = self.extract_weights(model)
        all_weights = np.concatenate(list(weights.values()))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 整体分布
        ax = axes[0]
        ax.hist(all_weights, bins=100, density=True, alpha=0.7, color='steelblue')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_key} - Overall Weight Distribution')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # 2. 边界附近分布
        ax = axes[1]
        boundaries = np.array([i + 0.5 for i in range(-8, 8)])
        boundary_mask = np.zeros_like(all_weights, dtype=bool)
        for b in boundaries:
            boundary_mask |= (np.abs(all_weights - b) < 0.2)
        ax.hist(all_weights[boundary_mask], bins=50, density=True, alpha=0.7, color='coral')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_key} - Weights Near Boundaries')
        for b in boundaries:
            ax.axvline(x=b, color='r', linestyle=':', alpha=0.3)
        
        # 3. QQ 图 (正态性检验)
        ax = axes[2]
        stats.probplot(all_weights, dist="norm", plot=ax)
        ax.set_title(f'{model_key} - Q-Q Plot')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Histogram saved to {save_path}")
        plt.show()
        
        del model
    
    def generate_report(self, save_path='weight_analysis_report.txt'):
        """生成分析报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LLM WEIGHT DISTRIBUTION ANALYSIS REPORT\n")
            f.write("For ZOE-Grad Framework - Model Dependency Analysis\n")
            f.write("="*80 + "\n\n")
            
            # 统计对比表
            f.write("📊 STATISTICAL COMPARISON\n")
            f.write("-"*80 + "\n")
            
            headers = ['Metric', 'OPT-1.3B', 'OPT-6.7B', 'LLaMA-2-7B']
            f.write(f"{headers[0]:<25} {headers[1]:<18} {headers[2]:<18} {headers[3]:<18}\n")
            f.write("-"*80 + "\n")
            
            metrics = [
                ('Kurtosis', 'kurtosis', '.4f'),
                ('Skewness', 'skewness', '.4f'),
                ('Boundary Coverage', 'boundary_coverage', '.4%'),
                ('Outlier Ratio', 'outlier_ratio', '.4%'),
                ('Layer Std CV', 'layer_std_cv', '.4f'),
                ('Boundary Dist Mean', 'boundary_dist_mean', '.4f'),
            ]
            
            for metric_name, stat_key, fmt in metrics:
                row = [metric_name]
                for model in ['OPT-1.3B', 'OPT-6.7B', 'LLaMA-2-7B']:
                    if model in self.weight_stats:
                        val = self.weight_stats[model][stat_key]
                        row.append(f"{val:{fmt}}")
                    else:
                        row.append("N/A")
                f.write(f"{row[0]:<25} {row[1]:<18} {row[2]:<18} {row[3]:<18}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("🔍 ANALYSIS & HYPOTHESES\n")
            f.write("="*80 + "\n\n")
            
            # 基于论文的分析
            f.write("Based on ZOE-Grad Framework (Section 6.2.2 & 6.3):\n\n")
            
            f.write("1. BOUNDARY COVERAGE & DISTRIBUTION CHOICE:\n")
            f.write("   - Higher boundary coverage → More weights near quantization boundaries\n")
            f.write("   - Requires smoother surrogate gradients (e.g., HTGE/Normal)\n")
            f.write("   - Lower boundary coverage → Uniform distribution may suffice\n\n")
            
            f.write("2. KURTOSIS & PERTURBATION TAILS:\n")
            f.write("   - Higher kurtosis → More outliers in weight distribution\n")
            f.write("   - May benefit from heavy-tailed perturbation distributions\n")
            f.write("   - Lower kurtosis → Light-tailed distributions (Uniform) work well\n\n")
            
            f.write("3. LAYER HETEROGENEITY & δ SELECTION:\n")
            f.write("   - Higher Layer Std CV → More variation across layers\n")
            f.write("   - May require layer-adaptive δ values\n")
            f.write("   - Reference: Single-boundary condition 2δC ≈ 1 (Section 6.3)\n\n")
            
            f.write("4. MODEL-SPECIFIC OBSERVATIONS (from Paper Table 1):\n")
            f.write("   - OPT models: Prefer Uniform/Normal distributions\n")
            f.write("   - LLaMA models: Prefer HTGE distribution\n")
            f.write("   - This analysis aims to explain WHY through weight statistics\n\n")
            
            f.write("="*80 + "\n")
            f.write("Generated by WeightDistributionAnalyzer\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Report saved to {save_path}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    
    # --- 新增：检查环境 ---
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA not available, falling back to CPU")

    # 1. 定义模型路径 (根据实际下载路径修改)
    model_names = {
        'OPT-1.3B': 'facebook/opt-1.3b',
        'OPT-6.7B': 'facebook/opt-6.7b',
        'LLaMA-2-7B': 'meta-llama/Llama-2-7b-hf',  # 需要 HuggingFace 认证
    }
    
    # 2. 创建分析器
    analyzer = WeightDistributionAnalyzer(model_names)
    
    # 3. 分析所有模型
    stats = analyzer.analyze_all_models()
    
    # 4. 绘制对比图
    analyzer.plot_comparison(save_path='weight_distribution_comparison.png')
    
    # 5. 可选：绘制单个模型的详细直方图
    # analyzer.plot_weight_histograms('OPT-1.3B', save_path='opt1.3b_histogram.png')
    # analyzer.plot_weight_histograms('LLaMA-2-7B', save_path='llama2_7b_histogram.png')
    
    # 6. 生成分析报告
    analyzer.generate_report(save_path='weight_analysis_report.txt')
    
    # 7. 打印关键发现
    print("\n" + "="*80)
    print("🔑 KEY FINDINGS FOR REBUTTAL")
    print("="*80)
    
    if 'OPT-1.3B' in stats and 'LLaMA-2-7B' in stats:
        opt_boundary = stats['OPT-1.3B']['boundary_coverage']
        llama_boundary = stats['LLaMA-2-7B']['boundary_coverage']
        
        opt_kurtosis = stats['OPT-1.3B']['kurtosis']
        llama_kurtosis = stats['LLaMA-2-7B']['kurtosis']
        
        print(f"\n1. Boundary Coverage Difference:")
        print(f"   OPT-1.3B: {opt_boundary:.4%} | LLaMA-2-7B: {llama_boundary:.4%}")
        print(f"   Difference: {abs(opt_boundary - llama_boundary):.4%}")
        print(f"   → Higher coverage suggests need for smoother gradients (HTGE)")
        
        print(f"\n2. Kurtosis Difference:")
        print(f"   OPT-1.3B: {opt_kurtosis:.4f} | LLaMA-2-7B: {llama_kurtosis:.4f}")
        print(f"   → Higher kurtosis indicates more outliers, may prefer heavy-tailed perturbations")
        
        print(f"\n3. Recommendation for Rebuttal:")
        if llama_boundary > opt_boundary:
            print("   LLaMA has higher boundary coverage → Explains preference for HTGE")
            print("   OPT has lower boundary coverage → Explains preference for Uniform/Normal")
        else:
            print("   Further analysis needed on layer-wise statistics")
        
        print("\n" + "="*80)