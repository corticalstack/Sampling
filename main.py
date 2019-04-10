"""
==========================================================
Example sampling techniques using KDD Cup 1999 IDS dataset
==========================================================

The following examples demonstrate various under and over-sampling techniques
for a dataset in which column is extremely imbalanced different over-sampling algorithms available in the imbalanced-learn package.

Visualisations using both linear and radial functions to illustrate separability

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from collections import Counter
import itertools


class Original:
    def fit_resample(self, x, y):
        return x, y


class Sampling:
    def __init__(self):
        print(__doc__)
        self.dataset = None
        self.x_encoded = None
        self.x = None
        self.y = None
        self.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
        self.attack_category_int = [0, 1, 2, 3, 4]
        self.attack_category = ['normal', 'dos', 'u2r', 'r2l', 'probe']
        self.weight = 0.2  # Full run 1, testing 0.1, 0.3 etc
        self.class_colours = np.array(["red", "green", "blue", "orange", "yellow"])
        self.ac_count = {}
        self.level_00 = ['attack_category']
        self.level_01 = ['attack_category', 'label']
        self.random_state = 20
        self.folds = 10
        self.scores = {}


        # Load data then set column names
        self.load_data()
        self.set_columns()

        # Drop large number of duplicates in dataset
        self.drop_duplicates()

        # Clean up the label column data
        self.clean()

        # Set binary target label
        self.set_binary_label()

        # Set attack_category to more clearly see the majority/minority classes - there are 5 "classes"
        self.set_attack_category()

        # Set original count by attack category
        self.set_attack_category_count()

        self.row_target_count_by_group(self.level_00, ['target'])  # Count by attack category majority/minority classes
        self.row_target_count_by_group(self.level_01, ['target'])  # More detailed row count by attack category/label
        self.dataset.attack_category.value_counts().plot(kind='bar', title='Count (attack_category)')
        plt.show()

        self.dataset = self.weight_attack_category(self.weight)
        self.row_target_count_by_group(self.level_00, ['target'])
        self.row_target_count_by_group(self.level_01, ['target'])
        self.dataset.attack_category.value_counts().plot(kind='bar', title='Count (attack_category)')
        plt.show()

        self.set_x_y(self.dataset)
        self.x = self.encode(self.x)

        # Reduce dimensionality with PCA for visualisation
        #self.x = self.pca(self.x)

        # Sampling options
        # for sampler in (Original(),
        #                 RandomOverSampler(),
        #                 SMOTE(random_state=0),
        #                 ADASYN(random_state=self.random_state),
        #                 BorderlineSMOTE(random_state=self.random_state, kind='borderline-1'),
        #                 BorderlineSMOTE(random_state=self.random_state, kind='borderline-2'),
        #                 SVMSMOTE(random_state=self.random_state)):
        self.sample(Original())

        #self.smote_nc()

        self.show_scores()

    @staticmethod
    def pca(ds):
        pca = PCA(n_components=2)
        ds = pca.fit_transform(ds)
        return ds

    # Plot classifier decision function to illustrate characteristics and sampling
    @staticmethod
    def plot_decision_function(x, y, clf, ax):
        plot_step = 0.02
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        ax.contourf(xx, yy, z, alpha=0.4)
        ax.scatter(x[:, 0], x[:, 1], alpha=0.8, c=y, edgecolor='k')

    def load_data(self):
        self.dataset = pd.read_csv('kddcup.data_10_percent')
        print('--- Original Shape')
        print('\tRow count:\t', '{}'.format(self.dataset.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

    def set_columns(self):
        self.dataset.columns = self.columns

    def drop_duplicates(self):
        print('\n--- Shape after duplicated dropped')
        self.dataset.drop_duplicates(keep='first', inplace=True)
        print('\tRow count:\t', '{}'.format(self.dataset.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.dataset.shape[1]))

    def clean(self):
        self.dataset['label'] = self.dataset['label'].str.rstrip('.')

    def set_binary_label(self):
        conditions = [
            (self.dataset['label'] == 'normal'),
            (self.dataset['label'] == 'back') | (self.dataset['label'] == 'buffer_overflow') |
            (self.dataset['label'] == 'ftp_write') | (self.dataset['label'] == 'guess_passwd') |
            (self.dataset['label'] == 'imap') | (self.dataset['label'] == 'ipsweep') |
            (self.dataset['label'] == 'land') | (self.dataset['label'] == 'loadmodule') |
            (self.dataset['label'] == 'multihop') | (self.dataset['label'] == 'neptune') |
            (self.dataset['label'] == 'nmap') | (self.dataset['label'] == 'perl') |
            (self.dataset['label'] == 'phf') | (self.dataset['label'] == 'pod') |
            (self.dataset['label'] == 'portsweep') | (self.dataset['label'] == 'rootkit') |
            (self.dataset['label'] == 'satan') | (self.dataset['label'] == 'smurf') |
            (self.dataset['label'] == 'spy') | (self.dataset['label'] == 'teardrop') |
            (self.dataset['label'] == 'warezclient') | (self.dataset['label'] == 'warezmaster')
        ]
        choices = [0, 1]
        self.dataset['target'] = np.select(conditions, choices, default=0)

    def set_attack_category(self):
        conditions = [
            (self.dataset['label'] == 'normal'),
            (self.dataset['label'] == 'back') | (self.dataset['label'] == 'land') |
            (self.dataset['label'] == 'neptune') | (self.dataset['label'] == 'pod') |
            (self.dataset['label'] == 'smurf') | (self.dataset['label'] == 'teardrop'),
            (self.dataset['label'] == 'buffer_overflow') | (self.dataset['label'] == 'loadmodule') |
            (self.dataset['label'] == 'perl') | (self.dataset['label'] == 'rootkit'),
            (self.dataset['label'] == 'ftp_write') | (self.dataset['label'] == 'guess_passwd') |
            (self.dataset['label'] == 'imap') | (self.dataset['label'] == 'multihop') |
            (self.dataset['label'] == 'phf') |  (self.dataset['label'] == 'spy') |
            (self.dataset['label'] == 'warezclient') | (self.dataset['label'] == 'warezmaster'),
            (self.dataset['label'] == 'ipsweep') | (self.dataset['label'] == 'nmap') |
            (self.dataset['label'] == 'portsweep') | (self.dataset['label'] == 'satan')
        ]
        self.dataset['attack_category'] = np.select(conditions, self.attack_category, default='na')
        self.dataset['attack_category_int'] = np.select(conditions, self.attack_category_int, default=0)

    def set_attack_category_count(self):
        ac = self.dataset['attack_category'].value_counts()
        for key, value in ac.items():
            self.ac_count[key] = value

    def weight_attack_category(self, weight):
        if weight == 1.0:
            print('\nNo initial under-sampling as weight {}'.format(weight))
            return self.dataset

        print('\nReducing dataset by under-sampling attack categories with weight {}'.format(weight))
        df = pd.DataFrame()
        for key, value in self.ac_count.items():
            samples = int(value * weight)
            df_sample = self.dataset[self.dataset.attack_category == key].sample(samples, random_state=self.random_state)
            df = df.append(df_sample)
        return df

    def set_x_y(self, ds):
        self.x = ds.iloc[:, :-4]
        self.y = ds.iloc[:, -1].values

    def encode(self, ds):
        le = preprocessing.LabelEncoder()
        ds = ds.apply(le.fit_transform)

        for col in ds.columns:
            if ds[col].dtype == np.int64:
                ds[col] = ds[col].astype(np.float64)

        sc = StandardScaler()
        ds = pd.DataFrame(sc.fit_transform(ds), columns=ds.columns)
        return ds

    def row_target_count_by_group(self, level, by):
        print('\n--- Row count by {}'.format(level))
        df = self.dataset.groupby(level)[by].count()
        df = df.rename(columns={by[0]: 'Count'})
        df['Percent'] = (df['Count'] / self.dataset.shape[0]) * 100
        df_flat = df.reset_index()
        print(df_flat)

    def sample(self, sampler):
        x, y = sampler.fit_resample(self.x, self.y)
        x = self.pca(x)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 11))

        # Build title with attack category sample counts
        title_suffix = ''
        class_sample_count = Counter(y)
        for cls, samples in class_sample_count.items():
            title_suffix = title_suffix + str(cls) + ':' + str(samples) + '  '

        # Linear
        fn_linear = 'Linear'
        clf_linear_svc = LinearSVC(dual=False).fit(x, y)
        self.plot_decision_function(x, y, clf_linear_svc, ax1)
        linear_title = sampler.__class__.__name__ + ' ' + fn_linear + ' y={}'.format(title_suffix)
        ax1.set_title(linear_title, size=16)

        # SVC with RBF
        fn_rbf = 'RBF'
        clf_rbf_svc = SVC(kernel='rbf', gamma=2, random_state=self.random_state).fit(x, y)
        self.plot_decision_function(x, y, clf_rbf_svc, ax2)
        rbf_title = sampler.__class__.__name__ + ' ' + fn_rbf + ' y={}'.format(title_suffix)
        ax2.set_title(rbf_title, size=16)

        fig.tight_layout()
        plt.savefig(fname='plots/' + 'DF - ' + sampler.__class__.__name__, dpi=300, format='png')
        plt.show()

        y_pred = clf_linear_svc.predict(x)
        self.show_cm_multiclass(y, y_pred, linear_title)
        self.register_score(sampler, fn_linear, clf_linear_svc, x, y, y_pred)

        y_pred = clf_rbf_svc.predict(x)
        self.show_cm_multiclass(y, y_pred, rbf_title)
        self.register_score(sampler, fn_rbf, clf_rbf_svc, x, y, y_pred)

    # Unlike SMOTE, SMOTENC can handle mix continuous/categorical features
    def smote_nc(self):
        self.set_x_y(self.dataset)
        smote_nc = SMOTENC(categorical_features=[1, 2, 3], random_state=self.random_state)
        x_res, y_res = smote_nc.fit_resample(self.x, self.y)
        print('Dataset after resampling:')
        print(sorted(Counter(y_res).items()))
        print('SMOTE-NC will generate categories for the categorical features:')
        print(x_res[-5:])

    def register_score(self, sampler, fn, clf, x, y, y_pred):
        # JP - change to ordered dict to remember insert order
        prefix = sampler.__class__.__name__ + '_' + fn + '_'
        self.scores[prefix + 'recall'] = recall_score(y, y_pred, average=None)
        self.scores[prefix + 'precision'] = precision_score(y, y_pred,  average=None)
        self.scores[prefix + 'f1'] = f1_score(y, y_pred,  average=None)
        self.scores[prefix + 'accuracy'] = cross_val_score(clf, x, y, scoring='accuracy', cv=self.folds)

    def show_scores(self):
        for sid, score in self.scores.items():
            print('ID: {}'.format(sid))
            print('\t\tScore{}'.format(score))

    def show_cm_multiclass(self, y, y_pred, title):
        cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4])
        self.plot_confusion_matrix(cm, classes=[0, 1, 2, 3, 4], title=title)

    def plot_confusion_matrix(self, cm, classes, title='Confusion matrix'):
        cmap = plt.cm.Blues
        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(fname='plots/' + 'CM - ' + title, dpi=300, format='png')
        plt.show()


sampling = Sampling()
