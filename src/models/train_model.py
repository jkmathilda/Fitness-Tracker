import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib


if __name__ != '__main__':
    raise ImportError("This script should be run directly, not imported.")

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle('../../data/interim/03_data_features.pkl')


# --------------------------------------------------------------
# Create a training and test set (set-based split)
# --------------------------------------------------------------

# Set-based split prevents temporal leakage between adjacent samples
sets = df['set'].unique()
rng = np.random.default_rng(42)
rng.shuffle(sets)
split_idx = int(len(sets) * 0.75)
train_sets = sets[:split_idx]
test_sets = sets[split_idx:]

df_train_full = df[df['set'].isin(train_sets)].copy()
df_test_full = df[df['set'].isin(test_sets)].copy()

# Drop metadata and features that leak information
# - duration: proxy for category (heavy=5reps shorter, medium=10reps longer)
# - pca_*: fitted on entire dataset in build_features.py
# - cluster: fitted on entire dataset in build_features.py
leaked_cols = [c for c in df.columns if c.startswith('pca_')] + ['cluster', 'duration']
drop_cols = ['participant', 'category', 'set'] + leaked_cols
drop_cols = [c for c in drop_cols if c in df.columns]

df_train = df_train_full.drop(columns=drop_cols)
df_test = df_test_full.drop(columns=drop_cols)

# Re-fit PCA on training data only (prevents data leakage)
predictor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
pca_scaler = StandardScaler()
pca = SklearnPCA(n_components=3)

train_pca_input = pca_scaler.fit_transform(df_train[predictor_columns])
test_pca_input = pca_scaler.transform(df_test[predictor_columns])
train_pca_result = pca.fit_transform(train_pca_input)
test_pca_result = pca.transform(test_pca_input)

for i in range(3):
    df_train[f'pca_{i+1}'] = train_pca_result[:, i]
    df_test[f'pca_{i+1}'] = test_pca_result[:, i]

# Re-fit KMeans on training data only
cluster_columns = ['acc_x', 'acc_y', 'acc_z']
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
df_train['cluster'] = kmeans.fit_predict(df_train[cluster_columns])
df_test['cluster'] = kmeans.predict(df_test[cluster_columns])

# Prepare X, y
X_train = df_train.drop('label', axis=1)
X_test = df_test.drop('label', axis=1)
y_train = df_train['label']
y_test = df_test['label']

fig, ax = plt.subplots(figsize=(10, 5))
y_train.value_counts().plot(kind='bar', ax=ax, color='dodgerblue', label='Train')
y_test.value_counts().plot(kind='bar', ax=ax, color='royalblue', label='Test')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
square_features = ['acc_r', 'gyr_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
time_features = [f for f in X_train.columns if '_temp_' in f]
frequency_features = [f for f in X_train.columns if (('_freq' in f) or ('_pse' in f))]
cluster_features = ['cluster']

print('Basic features:', len(basic_features))
print('Square features:', len(square_features))
print('PCA features:', len(pca_features))
print('Time features:', len(time_features))
print('Frequency features:', len(frequency_features))
print('Cluster features:', len(cluster_features))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

# Note: duration removed (leaks target info), re-run forward_selection for updated list
selected_features = [
    'acc_y_freq_0.0_Hz_ws_14',
    'acc_x_freq_0.0_Hz_ws_14',
    'pca_2',
    'gyr_x',
    'acc_y_pse',
    'gyr_y_freq_1.786_Hz_ws_14',
    'gyr_x_freq_1.071_Hz_ws_14',
    'gyr_y_freq_2.5_Hz_ws_14',
    'gyr_y_freq_2.143_Hz_ws_14'
]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set_1, 
    feature_set_2, 
    feature_set_3, 
    feature_set_4,
    selected_features
]

feature_names = [
    'Feature Set 1',
    'Feature Set 2',
    'Feature Set 3',
    'Feature Set 4',
    'Selected Features'
]

iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    # Scale features for distance-based models (KNN, NN)
    # Tree-based models (RF, DT) and NB are invariant to scaling
    feature_scaler = StandardScaler()
    selected_train_X = pd.DataFrame(
        feature_scaler.fit_transform(X_train[possible_feature_sets[i]]),
        columns=possible_feature_sets[i],
        index=X_train.index,
    )
    selected_test_X = pd.DataFrame(
        feature_scaler.transform(X_test[possible_feature_sets[i]]),
        columns=possible_feature_sets[i],
        index=X_test.index,
    )

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by='accuracy', ascending=False, inplace=True)

plt.figure(figsize=(10,10))
sns.barplot(x='model', y='accuracy', hue='feature_set', data=score_df)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1)
plt.legend(loc='lower right')
plt.show()


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
best_scaler = StandardScaler()
best_train_X = pd.DataFrame(
    best_scaler.fit_transform(X_train[feature_set_4]),
    columns=feature_set_4, index=X_train.index,
)
best_test_X = pd.DataFrame(
    best_scaler.transform(X_test[feature_set_4]),
    columns=feature_set_4, index=X_test.index,
)
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y
) = learner.random_forest(
    best_train_X,
    y_train,
    best_test_X,
    gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)
print(f"Best model accuracy (set-based split): {accuracy:.4f}")

# Re-train best model directly to save the fitted estimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV as GS
best_rf_gs = GS(
    RandomForestClassifier(),
    [{'min_samples_leaf': [2, 10, 50, 100, 200], 'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}],
    cv=5, scoring='accuracy'
)
best_rf_gs.fit(best_train_X, y_train.values.ravel())
best_rf_model = best_rf_gs.best_estimator_

# Save the fitted model, scaler, and feature list for prediction
joblib.dump(best_rf_model, '../../models/best_model_rf.pkl')
joblib.dump(best_scaler, '../../models/best_scaler.pkl')
joblib.dump(pca_scaler, '../../models/pca_scaler.pkl')
joblib.dump(pca, '../../models/pca_model.pkl')
joblib.dump(kmeans, '../../models/kmeans_model.pkl')
joblib.dump(feature_set_4, '../../models/feature_set.pkl')
print("Models saved to ../../models/")

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(columns=['set', 'category'] + leaked_cols, errors='ignore')

p_train_df = participant_df[participant_df['participant'] != 'A'].copy()
p_test_df = participant_df[participant_df['participant'] == 'A'].copy()

p_train_df = p_train_df.drop('participant', axis=1)
p_test_df = p_test_df.drop('participant', axis=1)

# Re-fit PCA on participant training data only
p_pca_scaler = StandardScaler()
p_pca = SklearnPCA(n_components=3)
p_train_pca = p_pca_scaler.fit_transform(p_train_df[predictor_columns])
p_test_pca = p_pca_scaler.transform(p_test_df[predictor_columns])
p_train_pca_result = p_pca.fit_transform(p_train_pca)
p_test_pca_result = p_pca.transform(p_test_pca)

for i in range(3):
    p_train_df[f'pca_{i+1}'] = p_train_pca_result[:, i]
    p_test_df[f'pca_{i+1}'] = p_test_pca_result[:, i]

# Re-fit KMeans on participant training data only
p_kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
p_train_df['cluster'] = p_kmeans.fit_predict(p_train_df[cluster_columns])
p_test_df['cluster'] = p_kmeans.predict(p_test_df[cluster_columns])

X_train = p_train_df.drop('label', axis=1)
y_train = p_train_df['label']
X_test = p_test_df.drop('label', axis=1)
y_test = p_test_df['label']

fig, ax = plt.subplots(figsize=(10, 5))
y_train.value_counts().plot(kind='bar', ax=ax, color='dodgerblue', label='Train')
y_test.value_counts().plot(kind='bar', ax=ax, color='royalblue', label='Test')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
p_scaler = StandardScaler()
p_train_scaled = pd.DataFrame(
    p_scaler.fit_transform(X_train[feature_set_4]),
    columns=feature_set_4, index=X_train.index,
)
p_test_scaled = pd.DataFrame(
    p_scaler.transform(X_test[feature_set_4]),
    columns=feature_set_4, index=X_test.index,
)
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y
) = learner.random_forest(
    p_train_scaled,
    y_train,
    p_test_scaled,
    gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
nn_scaler = StandardScaler()
nn_train_X = pd.DataFrame(
    nn_scaler.fit_transform(X_train[selected_features]),
    columns=selected_features, index=X_train.index,
)
nn_test_X = pd.DataFrame(
    nn_scaler.transform(X_test[selected_features]),
    columns=selected_features, index=X_test.index,
)
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y
) = learner.feedforward_neural_network(
    nn_train_X,
    y_train,
    nn_test_X,
    gridsearch=False
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
