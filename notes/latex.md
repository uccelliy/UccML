好，实话实说：**你不需要“大全”，你需要的是一套“够用、不乱、可扩展”的 LaTeX 基本公式骨架**。下面这份我按**数学分析 + 机器学习**的真实记笔记需求给你整理，目标是：

> 看到公式就能直接抄、改、扩，不会卡在 LaTeX 本身。

---

## 一、LaTeX 记笔记的最低配置

```latex
\documentclass[12pt]{article}

\usepackage{amsmath, amssymb, amsthm}
\usepackage{bm}        % 粗体数学符号
\usepackage{mathrsfs}  % 花体
\usepackage{geometry}
\geometry{margin=2.5cm}

\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\mathrm{Var}}
\newcommand{\Cov}{\mathrm{Cov}}

\begin{document}
```

这套组合**覆盖 99% 数学分析 + ML 笔记**，不用再折腾。

---

## 二、数学分析必备公式写法

### 1️⃣ 极限与无穷小

```latex
\lim_{x \to 0} \frac{\sin x}{x} = 1
```

```latex
x_n \to x \quad (n \to \infty)
```

```latex
f(x) = o(g(x)), \quad f(x) = O(g(x))
```

---

### 2️⃣ 导数与微分

```latex
f'(x), \quad \frac{df}{dx}
```

```latex
\frac{\partial f}{\partial x}, \quad \nabla f
```

```latex
df = f_x dx + f_y dy
```

```latex
\nabla f(\bm{x}) =
\begin{pmatrix}
\frac{\partial f}{\partial x_1} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{pmatrix}
```

---

### 3️⃣ 积分（定积分 / 多重积分）

```latex
\int_a^b f(x)\, dx
```

```latex
\iint_D f(x,y)\, dx\,dy
```

```latex
\int_{\R^n} f(\bm{x}) \, d\bm{x}
```

---

### 4️⃣ 级数

```latex
\sum_{n=1}^\infty a_n
```

```latex
\sum_{n=0}^\infty \frac{x^n}{n!}
```

---

### 5️⃣ 收敛性（分析里高频）

```latex
a_n \xrightarrow[n\to\infty]{} a
```

```latex
\sum a_n \text{ converges}
```

```latex
f_n \to f \quad \text{uniformly}
```

---

## 三、线性代数（机器学习核心）

### 1️⃣ 向量与矩阵

```latex
\bm{x} \in \R^n
```

```latex
\bm{X} \in \R^{n \times d}
```

```latex
\bm{x}^\top \bm{y}
```

```latex
\|\bm{x}\|_2, \quad \|\bm{x}\|_1
```

---

### 2️⃣ 矩阵运算

```latex
\bm{X}^\top \bm{X}
```

```latex
\bm{X}^{-1}
```

```latex
\det(\bm{X}), \quad \mathrm{rank}(\bm{X})
```

```latex
\mathrm{Tr}(\bm{X})
```

---

### 3️⃣ 特征值 / SVD

```latex
\bm{A}\bm{v} = \lambda \bm{v}
```

```latex
\bm{X} = \bm{U}\bm{\Sigma}\bm{V}^\top
```

---

## 四、概率论（机器学习必备）

### 1️⃣ 随机变量

```latex
X \sim \mathcal{N}(\mu, \sigma^2)
```

```latex
P(X \le x)
```

```latex
p(x), \quad p(x \mid y)
```

---

### 2️⃣ 期望与方差

```latex
\E[X]
```

```latex
\Var(X)
```

```latex
\Cov(X,Y)
```

```latex
\E[f(X)] = \int f(x)p(x)\,dx
```

---

### 3️⃣ 常见分布

```latex
X \sim \mathrm{Bernoulli}(p)
```

```latex
X \sim \mathrm{Binomial}(n,p)
```

```latex
X \sim \mathcal{N}(0,1)
```

---

## 五、机器学习常用公式

### 1️⃣ 损失函数

```latex
\mathcal{L}(y, \hat{y})
```

```latex
\mathcal{L} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
```

```latex
\mathcal{L} = -\sum_i y_i \log \hat{y}_i
```

---

### 2️⃣ 经验风险最小化

```latex
\min_\theta \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i)
```

---

### 3️⃣ 梯度下降

```latex
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
```

---

### 4️⃣ 正则化

```latex
\lambda \|\theta\|_2^2
```

```latex
\lambda \|\theta\|_1
```

---

### 5️⃣ 概率模型 / 贝叶斯

```latex
p(\theta \mid X) = \frac{p(X \mid \theta)p(\theta)}{p(X)}
```

```latex
\log p(X \mid \theta)
```

---

## 六、记笔记时**非常实用**的排版技巧

### ✔ 对齐多行推导

```latex
\begin{align}
\mathcal{L}(\theta)
&= \sum_{i=1}^n (y_i - \bm{x}_i^\top \theta)^2 \\
&= \|\bm{y} - \bm{X}\theta\|_2^2
\end{align}
```

---

### ✔ 条件说明

```latex
\quad \text{subject to } \|\theta\|_2 \le 1
```

---
