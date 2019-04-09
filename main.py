"""
==========================================================
Example sampling techniques using KDD Cup 1999 IDS dataset
==========================================================

The following examples demonstrate various under and over-sampling techniques
for a dataset in which column is extremely imbalanced different over-sampling algorithms available in the imbalanced-learn package.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.base import BaseSampler
from collections import Counter


# Make an identity sampler
class FakeSampler(BaseSampler):

    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y


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
        self.weight = 0.3
        self.class_colours = np.array(["red", "green", "blue", "orange", "yellow"])
        self.ac_count = {}
        self.level_00 = ['attack_category']
        self.level_01 = ['attack_category', 'label']
        self.random_state = 20
        self.folds = 10
        # create score dictionary
        self.scores = ['recall', 'precision', 'f1', 'roc_auc', 'accuracy']
        self.values = [np.array([]) for i in range(0, len(self.scores))]
        self.score_dict = OrderedDict(zip(self.scores, self.values))


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
        self.x = self.pca(self.x)

        # Sampling options
        self.random_oversampling()
        #self.adasyn_smote()
        #self.smote_alt()
        #self.smote_nc()

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
        sc = StandardScaler()
        ds = pd.DataFrame(sc.fit_transform(ds), columns=ds.columns)
        return ds

    def pca(self, ds):
        pca = PCA(n_components=2)
        ds = pca.fit_transform(ds)
        return ds

    def row_target_count_by_group(self, level, by):
        print('\n--- Row count by {}'.format(level))
        df = self.dataset.groupby(level)[by].count()
        df = df.rename(columns={by[0]: 'Count'})
        df['Percent'] = (df['Count'] / self.dataset.shape[0]) * 100
        df_flat = df.reset_index()
        print(df_flat)

    ###############################################################################
    # The following function will be used to plot the sample space after resampling
    # to illustrate the characterisitic of an algorithm.
    @staticmethod
    def plot_resampling(x, y, sampling, ax):
        x_res, y_res = sampling.fit_resample(x, y)
        ax.scatter(x_res[:, 0], x_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
        # make nice plotting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        return Counter(y_res)

    ###############################################################################
    # The following function will be used to plot the decision function of a
    # classifier given some data.
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

    def random_oversampling(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        clf = LinearSVC().fit(self.x, self.y)

        y_pred = clf.predict(self.x)
        self.add_score(clf, self.x, self.y, y_pred)

        self.plot_decision_function(self.x, self.y, clf, ax1)
        ax1.set_title('Linear SVC with y={}'.format(Counter(self.y)))
        pipe = make_pipeline(RandomOverSampler(random_state=0), LinearSVC())
        pipe.fit(self.x, self.y)
        self.plot_decision_function(self.x, self.y, pipe, ax2)
        ax2.set_title('Decision function for RandomOverSampler')
        fig.tight_layout()
        plt.show()

    def adasyn_smote(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        sampler = FakeSampler()
        clf = make_pipeline(sampler, LinearSVC())
        self.plot_resampling(self.x, self.y, sampler, ax1)
        ax1.set_title('Original data - y={}'.format(Counter(self.y)))

        ax_arr = (ax2, ax3, ax4)
        for ax, sampler in zip(ax_arr, (RandomOverSampler(random_state=0),
                                        SMOTE(random_state=0),
                                        ADASYN(random_state=0))):
            clf = make_pipeline(sampler, LinearSVC())
            clf.fit(self.x, self.y)
            self.plot_resampling(self.x, self.y, sampler, ax)
            ax.set_title('Resampling using {}'.format(sampler.__class__.__name__))
        fig.tight_layout()
        plt.show()

        # Plot decision boundaries
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        clf = LinearSVC().fit(self.x, self.y)
        self.plot_decision_function(self.x, self.y, clf, ax1)
        ax1.set_title('Linear SVC with y={}'.format(Counter(self.y)))
        sampler = SMOTE()
        clf = make_pipeline(sampler, LinearSVC())
        clf.fit(self.x, self.y)
        self.plot_decision_function(self.x, self.y, clf, ax2)
        ax2.set_title('Decision function for {}'.format(sampler.__class__.__name__))
        sampler = ADASYN()
        clf = make_pipeline(sampler, LinearSVC())
        clf.fit(self.x, self.y)
        self.plot_decision_function(self.x, self.y, clf, ax3)
        ax3.set_title('Decision function for {}'.format(sampler.__class__.__name__))
        fig.tight_layout()
        plt.show()

    #Alternative SMOTE samppling methods
    def smote_alt(self):
        fig, ((ax1, ax2), (ax3, ax4),
              (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15, 30))
        ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8))
        for ax, sampler in zip(ax_arr,
                               (SMOTE(random_state=0),
                                BorderlineSMOTE(random_state=0, kind='borderline-1'),
                                BorderlineSMOTE(random_state=0, kind='borderline-2'),
                                SVMSMOTE(random_state=0))):
            clf = make_pipeline(sampler, LinearSVC())
            clf.fit(self.x, self.y)
            self.plot_decision_function(self.x, self.y, clf, ax[0])
            ax[0].set_title('Decision function for {}'.format(
                sampler.__class__.__name__))
            self.plot_resampling(self.x, self.y, sampler, ax[1])
            ax[1].set_title('Resampling using {}'.format(sampler.__class__.__name__))
        fig.tight_layout()
        plt.show()

    # Unlike SMOTE, SMOTENC can handle mix continuous/categorical features
    def smote_nc(self):
        self.set_x_y(self.dataset)
        smote_nc = SMOTENC(categorical_features=[1, 2, 3], random_state=0)
        x_res, y_res = smote_nc.fit_resample(self.x, self.y)
        print('Dataset after resampling:')
        print(sorted(Counter(y_res).items()))
        print('SMOTE-NC will generate categories for the categorical features:')
        print(x_res[-5:])

    def add_score(self, clf, x, y, y_pred):
        self.score_dict['recall'] = np.append(self.score_dict['recall'], recall_score(y, y_pred))
        self.score_dict['precision'] = np.append(self.score_dict['precision'], precision_score(y, y_pred))
        self.score_dict['f1'] = np.append(self.score_dict['f1'], f1_score(y, y_pred))
        self.score_dict['roc_auc'] = np.append(self.score_dict['roc_auc'], roc_auc_score(y, y_pred))
        self.score_dict['accuracy'] = np.append(self.score_dict['accuracy'],
                                           cross_val_score(clf, x, y, scoring='accuracy', cv=self.folds))


sampling = Sampling()
