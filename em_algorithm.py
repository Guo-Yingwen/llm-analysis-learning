import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components=2, max_iter=100, tol=1e-5):
        """
        初始化高斯混合模型
        
        参数:
        n_components: 高斯分量的数量
        max_iter: 最大迭代次数
        tol: 收敛阈值
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
        # 模型参数
        self.weights = None    # 混合权重
        self.means = None      # 均值
        self.covs = None       # 协方差矩阵
        
    def _initialize_parameters(self, X):
        """初始化模型参数"""
        n_samples, n_features = X.shape
        
        # 随机初始化权重
        self.weights = np.ones(self.n_components) / self.n_components
        
        # 随机选择数据点作为初始均值
        random_idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_idx]
        
        # 使用数据协方差的一部分作为初始协方差
        self.covs = np.array([np.cov(X.T) for _ in range(self.n_components)])
        
    def _e_step(self, X):
        """E步：计算后验概率（责任）"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # 计算每个样本对应每个高斯分量的概率
        for k in range(self.n_components):
            gaussian = multivariate_normal(mean=self.means[k], cov=self.covs[k])
            responsibilities[:, k] = self.weights[k] * gaussian.pdf(X)
            
        # 归一化
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """M步：更新模型参数"""
        n_samples = X.shape[0]
        
        # 更新权重
        Nk = responsibilities.sum(axis=0)
        self.weights = Nk / n_samples
        
        # 更新均值
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        # 更新协方差
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            
    def fit(self, X):
        """训练模型"""
        self._initialize_parameters(X)
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            # E步
            responsibilities = self._e_step(X)
            
            # M步
            self._m_step(X, responsibilities)
            
            # 计算对数似然
            log_likelihood_new = self._compute_log_likelihood(X)
            
            # 检查收敛
            if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                print(f"收敛于第 {iteration + 1} 次迭代")
                break
                
            log_likelihood_old = log_likelihood_new
            
    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            gaussian = multivariate_normal(mean=self.means[k], cov=self.covs[k])
            likelihood[:, k] = self.weights[k] * gaussian.pdf(X)
            
        return np.sum(np.log(np.sum(likelihood, axis=1)))
    
    def predict(self, X):
        """预测样本所属的簇"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    
    # 生成两个高斯分布的数据
    n_samples = 300
    
    # 第一个高斯分布
    X1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples // 2)
    
    # 第二个高斯分布
    X2 = np.random.multivariate_normal([4, 4], [[1.5, -0.5], [-0.5, 1.5]], n_samples // 2)
    
    # 合并数据
    X = np.vstack([X1, X2])
    
    # 创建并训练模型
    gmm = GaussianMixtureModel(n_components=2, max_iter=100, tol=1e-5)
    gmm.fit(X)
    
    # 预测
    labels = gmm.predict(X)
    
    # 打印结果
    print("\n估计的均值:")
    print(gmm.means)
    print("\n估计的权重:")
    print(gmm.weights)
    
    # 可视化结果
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], color='red', marker='x', s=200, linewidth=3)
    plt.title('高斯混合模型聚类结果')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('gmm_clustering.png')
    plt.close() 