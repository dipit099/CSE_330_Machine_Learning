# Online 4: Unsupervised Learning, EM & GMM - Q&A Guide

**Focus**: Theory + Hand-based Calculations + Algorithm Understanding (No Coding)

---

## Table of Contents
1. [Unsupervised Learning Basics](#unsupervised-learning-basics)
2. [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
3. [Expectation-Maximization (EM) Algorithm](#expectation-maximization-em-algorithm)
4. [Hand-Calculated Examples](#hand-calculated-examples)
5. [Common Exam Questions](#common-exam-questions)

---

## Unsupervised Learning Basics

### Q1: What is Unsupervised Learning? How is it different from Supervised Learning?

**Answer:**

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|------------------|---------------------|
| **Labels** | Requires labeled data (X, y) | No labels; only data X |
| **Goal** | Learn mapping X → y | Find patterns/structure in X |
| **Example** | Classification, Regression | Clustering, Dimensionality Reduction |
| **Evaluation** | Accuracy, MSE, F1-score | Silhouette score, Davies-Bouldin index |

**Intuition:**
- **Supervised**: Learning from a teacher who provides correct answers
- **Unsupervised**: Learning patterns without anyone telling you what's right

### Q2: Name common Unsupervised Learning algorithms.

**Answer:**
1. **Clustering**
   - K-Means
   - Hierarchical Clustering
   - Gaussian Mixture Models (GMM)
   - DBSCAN
   
2. **Dimensionality Reduction**
   - PCA (Principal Component Analysis)
   - t-SNE
   - UMAP

3. **Anomaly Detection**
   - Isolation Forest
   - Local Outlier Factor (LOF)

---

## Gaussian Mixture Models (GMM)

### Q3: What is a Gaussian Distribution? Explain with an example.

**Answer:**

A **Gaussian (Normal) Distribution** is a bell-shaped probability curve defined by:

$$P(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Where:
- **$\mu$ (mu)**: Mean (center of the bell) - where most data is concentrated
- **$\sigma$ (sigma)**: Standard deviation (width of the bell) - how spread out the data is

**Example:** Heights of students in a class
- $\mu = 170$ cm (average height)
- $\sigma = 10$ cm (most students are between 160-180 cm)
- Very tall (200 cm) or very short (140 cm) students have low probability

**Visual:**
```
                    |
                   /|\
                  / | \     ← Peak at μ
                 /  |  \
                /   |   \   ← Width determined by σ
               /    |    \
    ──────────────────────────  x
              μ-σ  μ  μ+σ
```

### Q4: What is a Gaussian Mixture Model (GMM)?

**Answer:**

A **GMM** assumes your data is generated from $k$ different Gaussian distributions mixed together:

$$P(x) = \sum_{i=1}^{k} w_i \cdot P(x|C=i)$$

Where:
- **$k$**: Number of components (clusters)
- **$w_i$**: Weight (mixture coefficient) of component $i$ - how much of the data comes from component $i$
- **$P(x|C=i)$**: Probability of $x$ given it came from component $i$

**Intuition:**
If you have data of heights from a mixed group of adults and children:
- Component 1: Adults' heights (mean=170 cm, σ=8 cm, weight=60%)
- Component 2: Children's heights (mean=130 cm, σ=10 cm, weight=40%)
- **Any height** = 60% chance it's from Adults + 40% chance it's from Children

### Q5: What's the key difference between GMM and K-Means?

**Answer:**

| Aspect | K-Means | GMM |
|--------|---------|-----|
| **Assignment** | Hard assignment (0 or 1) | Soft assignment (probability 0-1) |
| **Example** | Point belongs to cluster 1 only | Point has 70% in cluster 1, 30% in cluster 2 |
| **Flexibility** | All clusters have same size/shape | Each cluster can have different variance |
| **Output** | Cluster labels | Probability distribution |

**Visualization:**
```
K-Means (Hard Assignment):        GMM (Soft Assignment):
Point A → Cluster 1 (100%)         Point A → Cluster 1 (80%)
Point B → Cluster 2 (100%)                  Cluster 2 (20%)
```

---

## Expectation-Maximization (EM) Algorithm

### Q6: What does the EM algorithm do?

**Answer:**

The **EM algorithm** is an iterative method to find the best parameters of a GMM when **we don't know which data point belongs to which component**.

**The Problem:** 
- If I told you "50 points are from Adults, 30 from Children", finding their means is easy
- But without knowing this assignment, it's hard to calculate parameters

**The Solution (EM):**
1. **E-step**: Guess the probability of each point belonging to each component
2. **M-step**: Update the component parameters using these probabilities
3. **Repeat**: Until parameters stabilize

---

### Q7: Explain the E-Step (Expectation Step)

**Answer:**

**Goal**: Calculate probability that each data point belongs to each component

$$p_{ij} = \frac{w_i \cdot P(x_j|C=i)}{\sum_{k=1}^{k} w_k \cdot P(x_k|C=k)}$$

Where:
- **$p_{ij}$**: Probability that point $x_j$ belongs to component $i$
- **Numerator**: Component $i$'s contribution to generating $x_j$
- **Denominator**: Total probability (normalization to make it 0-1)

**Intuition:** This is **Bayes' Rule** in action:
$$P(\text{Component}|\text{Data}) = \frac{P(\text{Data}|\text{Component}) \times P(\text{Component})}{P(\text{Data})}$$

**Example:** For a height of 160 cm with 2 components:
- Adult component: low probability (mean=170)
- Child component: high probability (mean=130)
- Result: 90% child, 10% adult

---

### Q8: Explain the M-Step (Maximization Step)

**Answer:**

**Goal**: Update component parameters using probabilities from E-step

Given probabilities $p_{ij}$ from E-step, update:

**1. Effective Count:**
$$n_i = \sum_{j=1}^{N} p_{ij}$$
(How many "weighted points" belong to component $i$)

**2. New Mean:**
$$\mu_i \leftarrow \frac{\sum_{j=1}^{N} p_{ij} \cdot x_j}{n_i}$$
(Weighted average of all points)

**3. New Covariance:**
$$\Sigma_i \leftarrow \frac{\sum_{j=1}^{N} p_{ij} \cdot (x_j - \mu_i)(x_j - \mu_i)^T}{n_i}$$
(Weighted spread of points around new mean)

**4. New Weight:**
$$w_i \leftarrow \frac{n_i}{N}$$
(Proportion of total data in component $i$)

**Key Insight:** Each point contributes to the mean/covariance based on its probability ($p_{ij}$), not just 0 or 1.

---

### Q9: When does EM algorithm stop?

**Answer:**

The algorithm stops when **log-likelihood converges**:

$$|L_{new} - L_{old}| < \epsilon$$

Where:
- **$L$**: Log-likelihood = $\sum_{j=1}^{N} \log(P(x_j))$
- **$\epsilon$**: Small threshold (e.g., 0.001)

**What it means:**
- If log-likelihood isn't improving much, parameters are stable
- Algorithm has found a local optimum

**Important:** EM finds **local optima**, not global. Run multiple times with different initializations to find the best solution.

---

## Hand-Calculated Examples

### Example 1: Simple GMM with 1D Data

**Scenario:** 
Heights of 4 animals (unknown mixture of 2 species)
Data: [10, 11, 60, 61] cm

**Assumption:** 2 components (species), each is Gaussian

---

#### **Iteration 0: Initialization**

**Initial Parameters (Random Guess):**
- **Component 1**: $\mu_1 = 15$, $\sigma_1 = 5$, $w_1 = 0.5$
- **Component 2**: $\mu_2 = 65$, $\sigma_2 = 5$, $w_2 = 0.5$

---

#### **Iteration 1: E-Step**

Calculate $P(x_j|C=i)$ using Gaussian formula:

$$P(x|C=i) = \frac{1}{\sigma_i\sqrt{2\pi}} e^{-\frac{(x-\mu_i)^2}{2\sigma_i^2}}$$

**For $x_1 = 10$:**

$P(10|C=1) = \frac{1}{5\sqrt{2\pi}} e^{-\frac{(10-15)^2}{2(5)^2}} = \frac{1}{12.566} e^{-0.1} = 0.0782 \times 0.905 = 0.0708$

$P(10|C=2) = \frac{1}{5\sqrt{2\pi}} e^{-\frac{(10-65)^2}{2(5)^2}} = 0.0782 \times e^{-121} \approx 0$ (extremely small)

Now calculate soft assignments using Bayes:
$$p_{11} = \frac{0.5 \times 0.0708}{0.5 \times 0.0708 + 0.5 \times 0} = \frac{0.0354}{0.0354} = 1.0$$

$p_{21} = 0$ (point 1 is almost certainly from component 1)

**For $x_2 = 11$:**
Similarly: $p_{12} = 0.999$, $p_{22} \approx 0$

**For $x_3 = 60$:**
$P(60|C=1) \approx 0$ (very far from mean 15)
$P(60|C=2) = 0.0782 \times e^{-0.1} = 0.0708$ (close to mean 65)

$$p_{13} = \frac{0.5 \times 0}{0.5 \times 0 + 0.5 \times 0.0708} \approx 0$$
$p_{23} = 1.0$ (point 3 is almost certainly from component 2)

**For $x_4 = 61$:**
Similarly: $p_{14} \approx 0$, $p_{24} = 1.0$

**E-Step Summary:**
```
Point 1 (10):  [1.0, 0.0]  ← Belongs to Component 1
Point 2 (11):  [0.999, 0.001] ← Mostly Component 1
Point 3 (60):  [0.001, 0.999] ← Mostly Component 2
Point 4 (61):  [0.0, 1.0]  ← Belongs to Component 2
```

---

#### **Iteration 1: M-Step**

**Calculate Effective Counts:**
$$n_1 = 1.0 + 0.999 + 0.001 + 0.0 = 2.0$$
$$n_2 = 0.0 + 0.001 + 0.999 + 1.0 = 2.0$$

**Calculate New Means:**
$$\mu_1 = \frac{p_{11} \cdot 10 + p_{12} \cdot 11 + p_{13} \cdot 60 + p_{14} \cdot 61}{n_1}$$
$$= \frac{1.0 \times 10 + 0.999 \times 11 + 0.001 \times 60 + 0.0 \times 61}{2.0}$$
$$= \frac{10 + 10.989 + 0.06 + 0}{2.0} = \frac{21.049}{2.0} = 10.52$$

$$\mu_2 = \frac{0.0 \times 10 + 0.001 \times 11 + 0.999 \times 60 + 1.0 \times 61}{2.0}$$
$$= \frac{0 + 0.011 + 59.94 + 61}{2.0} = \frac{120.951}{2.0} = 60.48$$

**Calculate New Weights:**
$$w_1 = \frac{n_1}{N} = \frac{2.0}{4} = 0.5$$
$$w_2 = \frac{n_2}{N} = \frac{2.0}{4} = 0.5$$

**Updated Parameters after Iteration 1:**
- **Component 1**: $\mu_1 = 10.52$ (moved closer to data), $w_1 = 0.5$
- **Component 2**: $\mu_2 = 60.48$ (moved closer to data), $w_2 = 0.5$

---

#### **Convergence:**
After more iterations, means would stabilize:
- **Component 1**: $\mu_1 \approx 10.5$ (small animals)
- **Component 2**: $\mu_2 \approx 60.5$ (large animals)

---

### Example 2: Understanding Covariance Update

**Problem:** Calculate covariance for Component 1 after E-step

Given:
- Data points: [10, 11]
- Probabilities: $p_{11} = 1.0, p_{12} = 0.999$
- New mean: $\mu_1 = 10.52$

**Formula:**
$$\Sigma_1 = \frac{\sum_j p_{1j}(x_j - \mu_1)^2}{n_1}$$

**Calculation:**

Term 1: $p_{11}(x_1 - \mu_1)^2 = 1.0 \times (10 - 10.52)^2 = 1.0 \times 0.2704 = 0.2704$

Term 2: $p_{12}(x_2 - \mu_1)^2 = 0.999 \times (11 - 10.52)^2 = 0.999 \times 0.2304 = 0.2301$

$$\Sigma_1 = \frac{0.2704 + 0.2301}{2.0} = \frac{0.5005}{2.0} = 0.25$$

---

### Example 3: Posterior Probability Calculation

**Problem:** A new point $x = 35$ arrives. What's the probability it belongs to each component?

**Given (after convergence):**
- **Component 1**: $\mu_1 = 10.5$, $\sigma_1 = 0.5$, $w_1 = 0.5$
- **Component 2**: $\mu_2 = 60.5$, $\sigma_2 = 0.5$, $w_2 = 0.5$

**E-Step for new point:**

$P(35|C=1) = \frac{1}{0.5\sqrt{2\pi}} e^{-\frac{(35-10.5)^2}{2(0.5)^2}}$
$= \frac{1}{1.253} e^{-\frac{600.25}{0.5}} = 0.798 \times e^{-1200.5} \approx 0$ (extremely small)

$P(35|C=2) = \frac{1}{0.5\sqrt{2\pi}} e^{-\frac{(35-60.5)^2}{2(0.5)^2}}$
$= 0.798 \times e^{-\frac{650.25}{0.5}} = 0.798 \times e^{-1300.5} \approx 0$ (even smaller)

**Result:** New point 35 doesn't fit either component well → **Anomaly detected!**

---

## Common Exam Questions

### Q10: What is Log-Likelihood in GMM context?

**Answer:**

$$L = \sum_{j=1}^{N} \log(P(x_j))$$

Where $P(x_j) = \sum_i w_i \cdot P(x_j|C=i)$ is the total probability of observing data point $x_j$.

**Why it matters:**
- **Goal**: Maximize log-likelihood (find parameters that make data most probable)
- **Convergence criterion**: When $L$ stops increasing significantly, algorithm has converged
- **Higher $L$** = Better model fit

**Example:**
```
Iteration 0: L = -15.3
Iteration 1: L = -12.8  ← Improved
Iteration 2: L = -12.5  ← Still improving
Iteration 3: L = -12.4  ← Very small improvement
Iteration 4: L = -12.39 ← Stop (converged)
```

---

### Q11: What is the difference between Hard and Soft Clustering?

**Answer:**

**Hard Clustering (K-Means):**
```
Point A belongs to Cluster 1 with probability 100%
Point A belongs to Cluster 2 with probability 0%
→ Binary decision
```

**Soft Clustering (GMM):**
```
Point A belongs to Component 1 with probability 60%
Point A belongs to Component 2 with probability 40%
→ Probabilistic (all components get a say)
```

**Advantage of Soft:**
- More realistic for overlapping clusters
- Provides uncertainty quantification
- Better for probabilistic models

---

### Q12: Why might EM get stuck at local optima?

**Answer:**

The log-likelihood surface is **non-convex** (has multiple peaks):

```
Log-Likelihood
    |     ╱╲        ╱╲
    |    ╱  ╲      ╱  ╲  ← Multiple local peaks
    |   ╱    ╲____╱    ╲
    |  ╱                ╲
    └────────────────────→ Iterations
         ↑          ↑
    Local Optima  Global Optimum (best)
```

**Solution:** Run EM multiple times with different random initializations and pick the best result.

---

### Q13: What are the key limitations of GMM?

**Answer:**

1. **Must specify $k$ beforehand**: How many components? (Use AIC/BIC to select)
2. **Local optima problem**: EM can get stuck in local maxima
3. **Computational cost**: Covariance matrix inversion is expensive for high dimensions
4. **Assumes Gaussian distributions**: May not fit all data types well
5. **Sensitive to initialization**: Bad initial guesses lead to bad results

---

### Q14: When should you use GMM vs K-Means?

**Answer:**

| Use Case | Algorithm |
|----------|-----------|
| Need cluster membership probabilities | GMM |
| Need interpretable, simple clusters | K-Means |
| Clusters have very different sizes | GMM |
| Want speed and simplicity | K-Means |
| Need probabilistic model | GMM |
| Large dataset (>1M points) | K-Means |

---

### Q15: Explain Bayes' Rule in GMM context

**Answer:**

**Bayes' Rule:**
$$P(C=i|x) = \frac{P(x|C=i) \cdot P(C=i)}{P(x)}$$

**In GMM terms:**
$$P(\text{Component }i|\text{ Data point }) = \frac{P(\text{Data}|\text{Component }i) \times w_i}{\sum_k P(\text{Data}|\text{Component }k) \times w_k}$$

**Interpretation:**
- **Prior** ($w_i$): How common is component $i$ overall?
- **Likelihood** ($P(x|C=i)$): How likely is this data given component $i$?
- **Posterior** ($P(C=i|x)$): Given the data, what's the probability it came from component $i$?

**Example:**
```
Component 1 (Adults): Mean height = 170, σ = 8, weight = 60%
Component 2 (Children): Mean height = 130, σ = 10, weight = 40%

New person's height: 150 cm

P(Adult | 150) = (P(150|Adult) × 0.6) / P(150)
                = (very small × 0.6) / denominator

P(Child | 150) = (P(150|Child) × 0.4) / P(150)
                = (small × 0.4) / denominator

Result: ~95% Child, ~5% Adult
```

---

## Algorithm Flow Summary

### GMM + EM Complete Workflow

```
1. INITIALIZATION
   ├─ Choose k (number of components)
   ├─ Randomly initialize μ_i, Σ_i, w_i for each component
   └─ Set iteration counter = 0

2. REPEAT (until convergence):
   
   a) E-STEP (Calculate soft assignments)
      ├─ For each data point x_j:
      │  ├─ Calculate P(x_j | C=i) for each component i
      │  └─ Calculate p_ij = (w_i × P(x_j|C=i)) / Σ_k(w_k × P(x_j|C=k))
      └─ Calculate log-likelihood L
   
   b) M-STEP (Update parameters)
      ├─ For each component i:
      │  ├─ Calculate n_i = Σ_j p_ij
      │  ├─ Update μ_i = (Σ_j p_ij × x_j) / n_i
      │  ├─ Update Σ_i = (Σ_j p_ij × (x_j - μ_i)(x_j - μ_i)ᵀ) / n_i
      │  └─ Update w_i = n_i / N
      └─ Increment iteration counter
   
   c) CHECK CONVERGENCE
      ├─ If |L_new - L_old| < ε: BREAK
      └─ Else: Go to step 2a

3. OUTPUT
   ├─ Learned parameters: μ_i, Σ_i, w_i for each component
   ├─ Soft assignments p_ij for each point
   └─ Final log-likelihood L
```

---

## Quick Reference: Formulas

| Formula | Purpose |
|---------|---------|
| $P(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | Gaussian PDF (1D) |
| $P(x) = \sum_i w_i \cdot P(x\|C=i)$ | GMM total probability |
| $p_{ij} = \frac{w_i \cdot P(x_j\|C=i)}{\sum_k w_k \cdot P(x_j\|C=k)}$ | E-step: Soft assignment |
| $n_i = \sum_j p_{ij}$ | M-step: Effective count |
| $\mu_i = \frac{\sum_j p_{ij} x_j}{n_i}$ | M-step: New mean |
| $\Sigma_i = \frac{\sum_j p_{ij}(x_j - \mu_i)(x_j - \mu_i)^T}{n_i}$ | M-step: New covariance |
| $w_i = \frac{n_i}{N}$ | M-step: New weight |
| $L = \sum_j \log(P(x_j))$ | Log-likelihood (convergence criterion) |

---

## Notes for Simulation/Exam

✅ **Focus on:**
- Understanding algorithm steps conceptually
- Hand-calculating E-step and M-step with small examples
- Recognizing convergence conditions
- Choosing between GMM/K-Means based on problem

❌ **Don't memorize:**
- Exact implementations (won't be asked)
- Complex matrix calculations (focus on intuition)
- Code details

💡 **Exam Tips:**
1. Draw diagrams showing how soft assignments work
2. Show calculations step-by-step with real numbers
3. Explain why EM finds local optima (non-convex surface)
4. Compare GMM vs K-Means with concrete examples
5. Use the animal height example to explain concepts
