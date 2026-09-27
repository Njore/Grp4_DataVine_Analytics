# Grp4 DataVine Analytics

## 1. Business Understanding

**Classification.** Predicting a wine's class from its chemical properties shows how measurable attributes can reliably distinguish categories without manual inspection. Useful for quality control or product categorization.

**Recommendation System.** Recommending similar feed types based on chicken weight outcomes shows how products can be grouped by measured performance rather than by label alone, useful for guiding substitutions or comparisons.

**Clustering.** Grouping U.S. states by violent crime severity shows how natural groupings can be surfaced in unlabeled data. Useful for regional risk profiling, policy, or resource allocation.

## 2. File Structure
Grp4_DataVine_Analytics/
├── README.md
├── LICENSE
├── classification/
│ ├── KNN_Classification.ipynb
│ ├── wine.KNN_classification.ipynb
│ └── wine.csv
├── clustering/
│ ├── clustering.ipynb
│ └── USArrests.csv
└── recommendation_systems/
├── Chickwts_recommendationsystems.ipynb
├── Dorcas_Chickwts_Recommendation.ipynb
└── chickwts.csv


Each folder contains the notebook(s) and dataset for that task. Where a task was completed independently by two group members, both notebooks are kept side by side rather than merged, to preserve each person's full approach.

## 3. Summaries

### Classification (KNN)

Two independent approaches tackled KNN classification on the wine dataset, using standardization, PCA-based dimensionality reduction (13 features reduced to 10 components, 95%+ variance retained), and GridSearchCV to tune hyperparameters. Both converged on the same conclusion despite being built separately: 100% test accuracy and ~98% cross-validation accuracy, with a perfect classification report across all 3 wine classes, and the two runs settled on slightly different "best" hyperparameter combinations at the same CV score, indicating a fairly flat accuracy landscape near the optimum rather than one clearly superior configuration.

### Recommendation System


Two independent approaches built a feed recommender from the chickwts dataset, using the same core pipeline: group weight by feed, standardize, reduce to a single PCA component, and recommend similar feeds via cosine similarity. Both produced comparable groupings, agreeing closely on which feeds are most similar to one another, though the two approaches differed in what was fed into PCA — one used only the average weight per feed as input, while the other used a fuller profile per feed (mean, median, standard deviation, min, max).

Both approaches share a structural limitation: reducing to one PCA component means cosine similarity can only return +1 or -1, giving a binary similar/dissimilar split rather than a graded similarity score. Despite the difference in input features, both agree that feeds separate into a clear high-weight group and a low-weight group, with recommendations landing consistently across both approaches.

### Clustering

Clustering was handled as a single shared notebook rather than two independent versions; the group discussed the intended methodology (standardization, feature selection, PCA, K-Means and GMM) beforehand to align on approach and clear up blockers ahead of the work being assigned


 Violent crime features (Murder, Assault, Rape) were standardized and reduced via PCA to 2 components, and K-Means, tuned via the elbow method, settled on 4 clusters, producing a finer severity breakdown from low to high crime (silhouette score 0.46).
GMM, tuned via BIC, settled on 2 clusters, producing a coarser high/low split with a higher silhouette score (0.56) and the added benefit of soft membership probabilities, which flagged a handful of low-confidence states sitting at the extreme edge of the low-crime group rather than genuinely ambiguous between groups. The two methods agree on the overall structure while offering different levels of granularity and confidence information.