import pandas as pd
import numpy as np
import os
import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('TkAgg')
import re
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
from hdtree import HundredQuantileSplit
# HDTreeClassifier, SmallerThanSplit, HundredQuantileSplit, TwentyQuantileRangeSplit, SingleCategorySplit, FixedValueSplit
# TwentyQuantileSplit,FiftyQuantileSplit, SmallerThanSplit,
from hdtree import HDTreeClassifier, EntropyMeasure, RelativeAccuracyMeasure
#, MisclassificationMeasure
import json
##
# SUBDATA TYPE CHECKS
def check_equal_data_types(dataa):
    for i_f in range(0, len(dataa.dtypes)):
        # check for type "object" -> that means there are mixed types in that column
        if pd.api.types.infer_dtype(dataa.iloc[:, i_f]) == "mixed":
            #print(pd.api.types.infer_dtype(dataa.iloc[:, i_f]))
            #print(dataa.columns[i_f])
            raise ValueError("Column " + dataa.columns[i_f] + " has 'mixed' datatype! Careful! To check!!")
    return


# Data Type Conversion
def data_type_conversion(data):
    # convert "datetime.datetime" objects to strings in the Column: Company Size
    for i_row in range(0, len(data)):
        if isinstance(data.loc[i_row, 'Company Size'], datetime.datetime):
            mon = data.loc[i_row, 'Company Size'].month
            day = data.loc[i_row, 'Company Size'].day
            if mon > day:
                cs_str = str(day) + '-' + str(mon)
            elif mon < day:
                cs_str = str(mon) + '-' + str(day)
            data.loc[i_row, 'Company Size'] = cs_str
    return data


# CHECK IF PATH/FILE EXISTS, IF YES -> do nothing, don't overwrite!
def save_files(direc_to, fig, what_to_save, graph=None):
    if os.path.exists(direc_to):
        print("directory \n ...'" + direc_to + "'... \nexists. If you want to save, please rename.")
    else:
        if what_to_save == "graphviz-tree":
            graph.render(direc_to)
        elif what_to_save == "fig":
            fig.savefig(direc_to)
    return


# variety index
def variety_index(x):
    unique_values = np.unique(x)
    res = 0
    n = len(unique_values)
    b = 0
    if n == 2:
        b = 6
    elif n == 3:
        b = 8
    elif n == 4:
        b = 11
    elif n == 5:
        b = 12.8
    elif n == 6:
        b = 14.67
    elif n == 7:
        b = 16
    if n > 1:
        pairwise_diff = [np.abs(a - b) for i, a in enumerate(unique_values) for j, b in enumerate(unique_values) if
                         i < j]
        y = np.mean(pairwise_diff)
        #res = y / (len(x)-1)

        res = y*2 / b
    return res


def plot_features(df_features, plot_feat="importances",
                  sort_by='importances_original', save_to=".",
                  show=True, saving=True):
    # Sorting data from highest to lowest
    #features_df_sorted = features_df.sort_values(by='importances_Step1', ascending=False)
    df_features_sorted = df_features.sort_values(by=sort_by, ascending=False)

    # Barplot of the result without borders and axis lines
    if show or saving:
        plt.figure()
        g = sns.barplot(data=df_features_sorted, x=plot_feat, y='features', palette="rocket")
        sns.despine(bottom=True, left=True)
        g.set_title('Feature importances')
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set(xticks=[])
        for value in g.containers:
            g.bar_label(value, padding=2)
        fig = g.figure
        plt.tight_layout()
        if saving:
            save_files(save_to, fig, what_to_save="fig")
        if not show:
            plt.close()

    return df_features_sorted


def plot_cv_result(x_label, y_label, xticklabels, plot_title, train_data,
                   val_data, save_to, show=True):
    '''Function to plot a grouped bar chart showing the training and validation
      results of the ML model in each fold after applying K-fold cross-validation.
     Parameters
     ----------
     x_label: str,
        Name of the algorithm used for training e.g 'Decision Tree'
     y_label: str,
        Name of metric being visualized e.g 'Accuracy'
     plot_title: str,
        This is the title of the plot e.g 'Accuracy Plot'
     train_result: list, array
        This is the list containing either training precision, accuracy, or f1 score.
     val_result: list, array
        This is the list containing either validation precision, accuracy, or f1 score.
     Returns
     -------
     The function returns a Grouped Barchart showing the training and validation result
     in each fold.
    '''

    # Set size of plot
    plt.figure(figsize=(9, 6))
    mean_val = np.mean(val_data)

    X_axis = np.arange(len(xticklabels))
    ax = plt.gca()
    plt.ylim(0, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='olive', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='khaki', label='Validation')
    plt.axhline(y=mean_val, color='goldenrod', linewidth=2, label="Mean Validation")
    plt.title(plot_title, fontsize=20)
    plt.xticks(X_axis, xticklabels)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_to + "PLOT_CV_" + y_label + str(len(xticklabels)) + ".png", bbox_inches='tight')
    if not show:
        plt.close()


# UPSAMPLING
def upsampling(agg_data):
    """
    :param agg_data:
    :param number_to_upsample: len(cat_3)
    :return:
    """
    cat_1 = agg_data[agg_data["Expert Opinion (Overall Performance)", "mean"] == 1]
    cat_2 = agg_data[agg_data["Expert Opinion (Overall Performance)", "mean"] == 2]
    cat_3 = agg_data[agg_data["Expert Opinion (Overall Performance)", "mean"] == 3]
    cat_4 = agg_data[agg_data["Expert Opinion (Overall Performance)", "mean"] == 4]

    number_to_upsample = len(cat_2)
    cat1_upsample = resample(cat_1,
                             replace=True,
                             n_samples=number_to_upsample,
                             random_state=42)
    cat4_upsample = resample(cat_4,
                             replace=True,
                             n_samples=number_to_upsample,
                             random_state=42)
    agg_data = pd.concat([cat1_upsample, cat_2, cat_3, cat4_upsample])
    return agg_data


# is the splitting of the dataset balanced?
def plot_barplot_numberCategoryOccurances(y_all, y_train, y_test, nametosave, save_to_folder):
    # number of category 1
    df_num_categ = pd.DataFrame(index=["y_all", "y_train",  "y_test"], columns=[1,2,3,4])
    for i_c in range(1, 5):
        df_num_categ.loc["y_all", i_c] = y_all.tolist().count(i_c)
        df_num_categ.loc["y_train", i_c] = y_train.tolist().count(i_c)
        df_num_categ.loc["y_test", i_c] = y_test.tolist().count(i_c)

    plt.figure()
    plt.bar(np.array([1,2,3,4])-0.3, df_num_categ.loc["y_all", :], 0.3, label="y_all", color="indianred")
    plt.bar(np.array([1,2,3,4]), df_num_categ.loc["y_train", :], 0.3, label="y_train", color="peru")
    plt.bar(np.array([1,2,3,4])+0.3, df_num_categ.loc["y_test", :], 0.3, label="y_test", color="khaki")
    plt.xticks(np.array([1,2,3,4]))
    plt.title(nametosave)
    plt.xlabel("category")
    plt.ylabel("number of occurances")
    plt.legend()
    plt.savefig(save_to_folder + "BARPLOT_numberCategoryOccurances.png")


def plot_pieplot_numberCategoryOccurances(agg_data_orig_sample, save_to_folder):
    df_y = agg_data_orig_sample.loc[:, 'Expert Opinion (Overall Performance)']
    df_y.groupby('mean').size().plot(kind='pie', y="mean",
                                     label="Categories",
                                     autopct='%1.1f%%')
    plt.savefig(save_to_folder + "PIEPLOT_numberCategoryOccurances_AFTERUPSAMPLING.png")


# GRIDSEARCH
def do_gridsearch(x_train, y_train, param_grid):

    clf = DecisionTreeClassifier(random_state=42)
    grid_cv = GridSearchCV(clf, param_grid, scoring='accuracy', cv=3, verbose=1).fit(x_train, y_train)
    grid_cv = GridSearchCV(clf, param_grid, scoring='roc_auc_ovo', cv=3, verbose=1).fit(x_train, y_train)
    print(grid_cv.best_params_)
    print("CV score for GS", grid_cv.best_score_)
    return grid_cv


def convert_float_to_int(column):
    # Function to transform float columns with essentially integer values to integers
    if column.dtype == 'float64' and all(value.is_integer() for value in column.dropna()):
        return column.astype(int)
    return column


def convert_obj_to_int(subdata):
    # Convert type <object> columns to type <Integer>
    for col in subdata.columns:
        if subdata[col].dtype == 'object':
            # Convert values to numeric and then to int
            subdata[col] = pd.to_numeric(subdata[col], errors='coerce').astype('Int64')
    return subdata


def read_data(dir_data, save_to_folder,
              feature_set="Personality",
              ylabel="Expert Opinion (Overall Performance)",
              company_id_label = "Index",
              sheet_name="Table_SCCH",
              header_column_in_row=1):
    """
    read excel data (Foundercheck data)
    save ...
    :param dir_data: str: direction to the excel sheet
    :param save_to_fold: str: direction, where things shall be saved to
    :param feature_set: str or list:
        list of individual features, which should be taken into account
        string: "Personality" -> Features, that belong to category "Personality"
            are taken into account
            possible options: "Personality", "Motivation", "Interests",
                              "ThinkingStyle", "Combined" (-> all features)
    :param ylabel: str: column name of labels
    :return:
    """

    # check if dir exist
    if not os.path.isdir(os.path.split(save_to_folder)[0]):
        os.makedirs(os.path.split(save_to_folder)[0])

    # read data
    data = pd.read_excel(open(dir_data, 'rb'),
                         header=header_column_in_row-1,
                         sheet_name=sheet_name)

    # polish column names, ensure that column names do net end with an empty space,
    # otherwise it does not recognize the given feature-set ("Numerical literacy ")
    # Rename columns with trailing spaces
    data.columns = data.columns.str.rstrip()

    # Data Type Conversion
    # convert "datetime.datetime" objects to strings in the Column: Company Size
    data = data_type_conversion(data)

##

    mandatory_cols = [company_id_label, ylabel]
    if type(feature_set) == str:
        if feature_set == "Motivation":
            feature_set = ['Recognition', 'Material Award', 'Affiliation', 'Autonomy',
                           'Flexibility_Motivation', 'Influence', 'Achievement',
                           'Personal Development','Meaning and values']  # , # Motivation
        elif feature_set == "Interests":
            feature_set = ['Physical', 'Practical', 'Analytical', 'Technical',
                           'Art and Culture', 'Creative', 'Helping/supporting',
                           'Networking', 'Leading', 'Commercial activities',
                           'Administrating processes', 'Transforming Processes'] # Interests
        elif feature_set == "ThinkingStyle":
            feature_set = ['Action', 'Reflection', 'Flexibility_Thinking', 'Structure']  # Thinking style
        elif feature_set == "Personality":
            feature_set = ['Resilience', 'Self-Discipline', 'Assertiveness',
                           'Activity level', 'Decisiveness', 'Sociability',
                           'Trust', 'Orientation on norms', 'Willingness to compromise',
                           'Emotionality', 'Intellectual curiosity']  # Personality
        elif feature_set == "Combined":
            feature_set = ['Material Award', 'Personal Development', 'Influence',
                           'Autonomy', 'Recognition', 'Meaning and values',
                           'Affiliation', 'Flexibility_Motivation', # Motivation
                           'Decisiveness', 'Orientation on norms', 'Trust', 'Self-Discipline',
                           'Willingness to compromise', # Personality
                           'Transforming Processes', 'Administrating processes', 'Physical',
                           'Practical', 'Technical', 'Leading', 'Creative', 'Networking', # Interests
                           'Flexibility_Thinking', 'Action', 'Reflection', 'Structure' #ThinkingStyle
                            ]
        else:
            print("Please define feature-set! "
                  "Either: 'Motivation', 'Interests', 'ThinkingStyle', 'Personality', or "
                  "'Combined'. Or define a list of strings with the features of interest.")
    elif type(feature_set) == list:

        feature_set = feature_set

    subdata = data[mandatory_cols + feature_set]

    if "Company Size" in subdata.columns:
        subdata['Company Size'] = pd.to_numeric([str(r).split('-')[0] for r in subdata['Company Size']],
                                                errors='coerce')
        subdata['Company Size'] = subdata['Company Size'].fillna(1).astype(int)
    if "Median Tenure" in subdata.columns:
        subdata['Median Tenure'] = pd.to_numeric(subdata['Median Tenure'].astype(str).str.replace(',', ''),
                                                 errors='coerce').fillna(0).astype(float)
    # TODO: only drop nans of ylabel-column?
    subdata = subdata.dropna()
    # Convert type <object>  columns to <integer> columns
    subdata = convert_obj_to_int(subdata)
    # Convert <float> type to <integer> type. Apply the function to each column in the DataFrame
    subdata = subdata.apply(convert_float_to_int)
    subdata = subdata[subdata[ylabel] >= 0]

    # check: do we have mixed datatypes within in some columns? If no error arises, everything is fine!
    check_equal_data_types(subdata)

    # Get feature-names
    features = [col for col in subdata.columns if col not in mandatory_cols]
    features_wCompanyID = features.copy()
    features_wCompanyID.append(company_id_label)

    # K_featnames_wCompanyID = K_featnames.copy()
    # arrays = [['Company ID'], ['']]
    # tuples = list(zip(*arrays))
    # index = pd.MultiIndex.from_tuples(tuples)
    #
    # K_featnames_wCompanyID = K_featnames_wCompanyID.append(index)

    return subdata, features, features_wCompanyID


def read_simulated_data(json_dir):
    """
    Wrapper Function to call and read simulated data.
    :param json_dir: str - Directory to json file with excel
                           file specifications.
    :return: agg_data: pd.Dataframe: data
             data_specs: dict
    """
    with open(json_dir, 'r') as f:
        data_specs = json.load(f)

    dir_data = data_specs["dir_data"]

    agg_data = pd.read_excel(dir_data,
                             index_col=0)#, header=[0])#, 1])
    agg_data = agg_data.rename(columns={'Unnamed: 2_level_1': ''})
    # if len(agg_data.columns[0]) > 1:
    #     new_cols = []
    #     for i_c in agg_data.columns:
    #         new_cols.append(i_c[0])
    #     agg_data.columns = new_cols

    K_featnames = agg_data.columns[2:]
    K_featnames_wCompanyID = agg_data.columns[1:]
    ylabel_agg_data = agg_data.columns[0]

    data_specs["K_featnames"] = K_featnames
    data_specs["K_featnames_wCompanyID"] = K_featnames_wCompanyID
    data_specs["ylabel_agg_data"] = ylabel_agg_data

    save_to_folder = os.path.join(data_specs["dir_to"],
                                  data_specs["save_to_folder"],
                                  data_specs["name_to_save"])
    data_specs["save_to_folder"] = save_to_folder

    if not os.path.exists(save_to_folder):
        # If it doesn't exist, create it
        os.makedirs(save_to_folder)
        print(f"Directory created: '{save_to_folder}' ")

    return agg_data, data_specs


def read_foundercheck_data(json_dir):
    """
    Wrapper function to call and to read foundercheck data, with given
    json-directory, where all parameters of excel sheet are defined.
    :param json_dir: str
    :return: agg_data: pd.Dataframe: data
             data_specs: dict
    """
    with open(json_dir, 'r') as f:
        data_specs = json.load(f)

    dir_to = data_specs["dir_to"]
    dir_data = data_specs["dir_data"]
    save_to_fold = data_specs["save_to_folder"]
    excel_sheet_name = data_specs["excel_sheet_name"]
    which_feature_set = data_specs["which_feature_set"]
    ylabel = data_specs["ylabel"]
    person_id_label = data_specs["person_id_label"]
    header_column_in_row = data_specs["header_column_in_row"]

    name_to_save = which_feature_set[0] + '_' + which_feature_set[1]
    save_to_folder = os.path.join(dir_to,
                                  save_to_fold,
                                  name_to_save)
    data_specs["save_to_folder"] = save_to_folder
    data_specs["name_to_save"] = name_to_save

    if not os.path.exists(save_to_folder):
        # If it doesn't exist, create it
        os.makedirs(save_to_folder)
        print(f"Directory created: '{save_to_folder}' ")

    agg_data, K_featnames, K_featnames_wCompanyID = \
        read_data(dir_data, save_to_folder,
                  feature_set=which_feature_set,
                  ylabel=ylabel,
                  company_id_label=person_id_label,
                  sheet_name=excel_sheet_name,
                  header_column_in_row=header_column_in_row)

    data_specs["K_featnames"] = K_featnames
    data_specs["K_featnames_wCompanyID"] = K_featnames_wCompanyID

    return agg_data, data_specs


def aggregate_data(data, company_id_label):
    # todo try to replace NAs of company size and median tenure with another values
    # TODO: Check if there are NAs in the data
    agg_data = data.groupby(company_id_label).aggregate(
        {k: (np.mean if k == 'Expert Opinion (Overall Performance)'
             else np.mean if k == "Material Award"
        else np.mean if k == "Recognition"
        else variety_index if k == "Personal Development"
        else variety_index if k == "Influence"
        else variety_index if k == "Autonomy"
        else np.mean if k == "Decisiveness"
        else np.mean if k == "Orientation on norms"
        else np.mean if k == "Trust"
        else np.mean if k == "Self-Discipline"
        else np.mean if k == "Transforming Processes"
        else np.mean if k == "Physical"
        else np.mean if k == "Technical"
        else variety_index if k == "Practical"
        else np.mean if k == "Action"
        else np.mean if k == "Leading"
        else np.mean if k == "Creative"
        else np.mean if k == "Networking"
        else variety_index if k == "Meaning and values"
        else variety_index if k == "Flexibility_Motivation"
        else variety_index if k == "Willingness to compromise"
        else np.mean if k == "Reflection"
        else np.mean if k == "Structure"

        else np.sum if k == 'Company Size'
        else {np.mean, variety_index}) for k in data.columns[1:]}
    ).reset_index()
    return agg_data


def select_best_features(agg_data,
                         ylabel='Expert Opinion (Overall Performance)',
                         n_best_feat=12):
    # selecting the best 12 features using slectkbest & chi2
    x = agg_data.iloc[:, 2:].round(2)
    y = agg_data[ylabel]
    if len(x.columns) <= n_best_feat:
        kfeat = "all"
    else:
        kfeat = n_best_feat
    select = SelectKBest(chi2, k=kfeat)
    select.fit_transform(x, y)
    selectK_mask = select.get_support()
    feature_names = x.columns
    best_features = feature_names[selectK_mask]

    return best_features


def split_train_test(agg_data, dec_tree_parameters,
                     K_featnames, K_featnames_wCompanyID,
                     K_col_categories,
                     company_id_label):
    """

    :param agg_data: pandas dataframe
    :param dec_tree_parameters: dict with parameters / variables
    :param K_featnames: featurenames
    :param K_featnames_wCompanyID: featurenames + companyID
    :param K_col_categories: column which defines categories/labels of the dataset (y=)
    :return:
    """
    randomstate = dec_tree_parameters["randomstate"]
    perc_trainingset = dec_tree_parameters["trainingset"]
    do_upsamplin = dec_tree_parameters["do_upsamplin"]

    agg_data_orig_sample0 = agg_data.sample(frac=1, random_state=randomstate).reset_index(drop=True)

    ind0 = int(len(agg_data) * perc_trainingset / 100)

    if do_upsamplin:
        # upsampling only on training set
        train_orig = agg_data_orig_sample0.loc[0:ind0, :]
        test_orig = agg_data_orig_sample0.loc[ind0 + 1:, :]

        agg_data_upsampled_train = upsampling(train_orig)
        ind = len(agg_data_upsampled_train)
        agg_data_orig_sample = pd.concat([agg_data_upsampled_train, test_orig],
                                         axis=0, ignore_index=True)
    else:
        ind = ind0
        agg_data_orig_sample = agg_data_orig_sample0

    x_all = agg_data_orig_sample.loc[:, K_featnames].round(2)
    y_all_label = agg_data_orig_sample.loc[:, [K_col_categories]]
    y_all = y_all_label.values.flatten()
    x_train = agg_data_orig_sample.loc[0:ind, K_featnames].round(2)
    x_train_wCompanyID = agg_data_orig_sample.loc[0:ind, K_featnames_wCompanyID].round(2)
    y_train_label = agg_data_orig_sample.loc[0:ind, [K_col_categories]]
    y_train = y_train_label.values.flatten()
    if isinstance(K_col_categories, tuple):
        y_train_wCompanyID = agg_data_orig_sample.loc[0:ind,
                             [K_col_categories[0], company_id_label]]
    elif isinstance(K_col_categories, str):
        y_train_wCompanyID = agg_data_orig_sample.loc[0:ind,
                             [K_col_categories, company_id_label]]
    else:
        raise ValueError("K_col_categories is either string nor tuple. "
                         "To check!! K_col_categories = " +
                         str(K_col_categories))
    x_test = agg_data_orig_sample.loc[ind + 1:, K_featnames].round(2)
    x_test.reset_index(inplace=True, drop=True)
    x_test_wCompanyID = agg_data_orig_sample.loc[ind + 1:, K_featnames_wCompanyID].round(2)
    x_test_wCompanyID.reset_index(inplace=True, drop=True)
    y_test_label = agg_data_orig_sample.loc[ind + 1:, [K_col_categories]]
    y_test_label.reset_index(inplace=True, drop=True)
    y_test = y_test_label.values.flatten()
    if isinstance(K_col_categories, tuple):
        y_test_wCompanyID = agg_data_orig_sample.loc[ind + 1:,
                            [K_col_categories[0], company_id_label]].reset_index(drop=True)
    elif isinstance(K_col_categories, str):
        y_test_wCompanyID = agg_data_orig_sample.loc[ind + 1:,
                            [K_col_categories, company_id_label]].reset_index(drop=True)
    x_y_dict = {"x_all": x_all,
                "x_train": x_train,
                "x_train_wCompanyID": x_train_wCompanyID,
                "x_test": x_test,
                "x_test_wCompanyID": x_test_wCompanyID,
                "y_all_label": y_all_label,
                "y_all": y_all,
                "y_train_label": y_train_label,
                "y_train": y_train,
                "y_train_wCompanyID": y_train_wCompanyID,
                "y_test_label": y_test_label,
                "y_test": y_test,
                "y_test_wCompanyID": y_test_wCompanyID,
                "agg_data_orig_sample": agg_data_orig_sample
                }
    return x_y_dict
##


def transform_y_to_0_1(agg_data, ylabel):
    # Transform y to a scala: [0,1], instead [2,3]

    # Check if it is already [0,1]
    if len(agg_data.loc[:, ylabel].unique()) == 2 and \
            set(agg_data.loc[:, ylabel].unique()) == {0, 1}:
        print(set(agg_data.loc[:, ylabel].unique()))

    # Check if unique values are [2, 3]
    elif len(agg_data.loc[:, ylabel].unique()) == 2 and \
            set(agg_data.loc[:, ylabel].unique()) == {2, 3}:
        agg_data.loc[:, ylabel] = agg_data.loc[:,ylabel] - 2

    # transform y to a scala: [0,1], instead [1,2]
    elif len(agg_data.loc[:, ylabel].unique()) == 2 and \
            set(agg_data.loc[:, ylabel].unique()) == {1, 2}:
        agg_data.loc[:, ylabel] = agg_data.loc[:, ylabel] - 1

    elif len(agg_data.loc[:, ylabel].unique()) == 4 and \
            set(agg_data.loc[:, ylabel].unique()) == {1, 2, 3, 4}:
        agg_data[ylabel + "_original"] = agg_data[ylabel]
        # Define a mapping of values
        value_mapping = {1: 0, 2: 0, 3: 1, 4: 1}
        # Use the mapping to create the new column
        agg_data[ylabel] = agg_data[ylabel].map(value_mapping)

    else:
        raise ValueError("To check! y has , "
                         "other values than {1,2} or {2,3} or {1,2,3,4}. "
                         "To check!!!")
    return agg_data


def create_df_accuracies(y_train, y_test, predicted_train, predicted_test):
    df_accuracies = pd.DataFrame(columns=["DT1", "DT1 + AI", "DT3"],
                                 index=["acc_test", "acc_train", "MAE_test", "MAE_train", "MAE_AI"])
    predicted_test = predicted_test.astype(int)
    predicted_train = predicted_train.astype(int)
    y_test = y_test.astype(int)
    y_train = y_train.astype(int)

    df_accuracies.loc["acc_test", "DT1"] = np.round(accuracy_score(y_test, predicted_test), 3)
    df_accuracies.loc["acc_train", "DT1"] = np.round(accuracy_score(y_train, predicted_train), 3)
    df_accuracies.loc["MAE_test", "DT1"] = np.around(mean_absolute_error(y_test, predicted_test), 3)
    df_accuracies.loc["MAE_train", "DT1"] = np.around(mean_absolute_error(y_train, predicted_train), 3)
    return df_accuracies


def create_df_y(y_train_wCompanyID, ylabel, x_train, company_id_label):
    df_y = y_train_wCompanyID.copy()
    df_y.rename(columns={ylabel: "y_train"}, inplace=True)
    #df_y.rename(columns={company_id_label: "Company_ID"}, inplace=True)
    df_y = pd.concat([df_y, x_train], axis=1)
    return df_y


def step1(dec_tree_params,
          x_train, y_train,
          x_test, y_test,
          df_y, df_y_test,
          which_step):
    """
    Step1 of BAPC. Train Decision tree (sklearn) on given training-dataset.
    - split dataset into training and test datasets
    - Option: do upsampling of trainingset, if do_upsamplin = True
    Parameters:
    ----------
        dec_tree_params,
        x_train,
        y_train,
        x_test,
        y_test,
        df_y,
        df_y_test,
        which_step
    Return:
    -------
        dt_y: dataframe of prediction summaries (y-labels) for different steps
    """

    randomstate = dec_tree_params["randomstate"]
    trainingset = dec_tree_params["trainingset"]
    max_depth = dec_tree_params["max_depth"]
    min_samples_leaf = dec_tree_params["min_samples_leaf"]
    min_samples_split = dec_tree_params["min_samples_split"]
    ccpalpha = dec_tree_params["ccp_alpha"]
    nametosave = dec_tree_params["nametosave"]
    save_to_folder = dec_tree_params["save_to_folder"]
    person_id_label = dec_tree_params["person_id_label"]

    clf = DecisionTreeClassifier(splitter="best",
                                 max_depth=max_depth, random_state=42,
                                 min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,
                                 min_impurity_decrease=0.01,
                                 ccp_alpha=ccpalpha, criterion='gini')

    clf.fit(x_train, y_train)
    predicted_train = clf.predict(x_train)
    predicted_test = clf.predict(x_test)

    #df_accuracies = create_df_accuracies(y_train, y_test, predicted_train, predicted_test)

    #print(classification_report(y_test, predicted_test))
    #print('base model accuracy test data', df_accuracies.loc["acc_test", "DT1"])
    #print('base model accuracy train data', df_accuracies.loc["acc_train", "DT1"])
    #print(y_train - predicted_train)

    # dataframe of prediction summaries (y-labels) for different steps
    df_y["y_predicted_hdtree" + which_step] = predicted_train.tolist()
    df_y_test["y_predicted_hdtree" + which_step] = predicted_test.tolist()

    # leaf ids
    node_indicator = clf.decision_path(x_train)
    leaf_id = clf.apply(x_train)
    df_y["leaf_id_DT1"] = leaf_id

    return clf, df_y, df_y_test


def step1_hdtree(x_train, y_train, x_test, y_test,
                 dec_tree_parameters, df_y, df_y_test,
                 which_step="1"):
    """
    Run Decision Tree Classification with hdtree.
    :param x_train: pd Dataframe
    :param y_train: ndarray
    :param x_test: pd Dataframe
    :param y_test: ndarray
    :param dec_tree_parameters: dictionary with parameters for decision tree
    :param df_y: pd.Dataframe: Summary of all results for training set
    :param df_y_test: pd.Dataframe: Summary of all results for test set
    :return:
        hdtree_linear: hdtree object
        df_y: pd.Dataframe: Updated with prediction results of hdtree
                            decision tree for training set
        df_y_test: pd.Dataframe: Updated with prediction results of hdtree
                                decision tree for test set
    """
    if dec_tree_parameters["which_cost_fct"] == "RelativeAccuracyMeasure":
        inf_measure = RelativeAccuracyMeasure()
    elif dec_tree_parameters["which_cost_fct"] == "EntropyMeasure":
        inf_measure = EntropyMeasure()
    else:
        raise ValueError("Using HDtree: Choose between: which_cost_fct = 'RelativeAccuracyMeasure' or "
                         "'EntropyMeasure', or define new Cost-function-Information-Measures. ")

    hdtree_linear = HDTreeClassifier(allowed_splits=[HundredQuantileSplit.build()],
                                     information_measure=inf_measure,# EntropyMeasure(), #RelativeAccuracyMeasure()
                                     # #MisclassificationMeasure()
                                     attribute_names=x_train.columns.tolist(),
                                     max_levels=dec_tree_parameters["max_depth"],
                                     min_samples_at_leaf=dec_tree_parameters["min_samples_leaf"]
                                     )
    hdtree_linear.fit(x_train.values, y_train)
    hdtree_linear.generate_dot_graph()
    hdtree_linear.score(x_test.values, y_test)
    #
    df_y["y_predicted_hdtree" + which_step] = hdtree_linear.predict(x_train.values).astype(float)
    df_y_test["y_predicted_hdtree" + which_step] = hdtree_linear.predict(x_test.values).astype(float)

    # Assign to each datapoint the leaf node number
    counter = [0]
    df_y["leaf_id_DT" + which_step] = None
    assign_data_traverse_tree(hdtree_linear._node_head, counter,
                              df_y, which_step)

    return hdtree_linear, df_y, df_y_test


def name_nametosave(dec_tree_parameters):
    if dec_tree_parameters["which_cost_fct"] == "RelativeAccuracyMeasure":
        cost_fct = "RelAccur"
    elif dec_tree_parameters["which_cost_fct"] == "EntropyMeasure":
        cost_fct = "Entropy"
    elif dec_tree_parameters["which_cost_fct"] == "Gini":
        cost_fct = "Gini"
    else:
        cost_fct = ''
    nametosave = "_train" + str(dec_tree_parameters["trainingset"]) + \
                 "_rs" + str(dec_tree_parameters["randomstate"]) + \
                 "_d" + str(dec_tree_parameters["max_depth"]) + \
                 "_ml" + str(dec_tree_parameters["min_samples_leaf"]) + \
                 "_ms" + str(dec_tree_parameters["min_samples_split"]) + "_" + \
                 cost_fct + '_'
    return nametosave


def name_rowname(params_grid, i_t, i_rs, i_md, i_ml, i_ms, i_ccp):
    rowname = "perc_train" + str(params_grid["trainingset"][i_t]) + \
              "_rs" + str(params_grid["randomstate"][i_rs]) + \
              "_depth" + str(params_grid["max_depth"][i_md]) + \
              "_minleaf" + str(params_grid["min_samples_leaf"][i_ml]) + \
              "_minsampl" + str(params_grid["min_samples_split"][i_ms]) + \
              "ccp" + str(params_grid["ccp_alpha"][i_ccp]).replace(".", "_")
    return rowname


def fix_features_of_step1(fix_step1, features_df_sorted,
                          x_train, x_test):
    if fix_step1:
        add_name = "_fixS1"
    else:
        add_name = ""

    # FIX FEATURES OF STEP 1
    if fix_step1:
        # drop 0-features
        df_new_feat_step1 = features_df_sorted[features_df_sorted.loc[:, 'importances1'] != 0]
        # rename features: remove "(", ")" , to match the columnames of x_train
        df_new_feat_step1["features"] = df_new_feat_step1["features"].str.replace('(', '/')
        df_new_feat_step1["features"] = df_new_feat_step1["features"].str.replace(')', '')

        new_feat_step1 = df_new_feat_step1["features"].tolist()
        x_train.columns = x_train.columns.get_level_values(0) + '/' + x_train.columns.get_level_values(1)
        x_test.columns = x_test.columns.get_level_values(0) + '/' + x_test.columns.get_level_values(1)

        x_train_new = x_train[new_feat_step1]
        x_test_new = x_test[new_feat_step1]
    else:
        df_new_feat_step1 = features_df_sorted
        x_train_new = x_train
        x_test_new = x_test

    return x_train_new, x_test_new, df_new_feat_step1, add_name


def get_splitting_values(hdtree_linear):
    """
    # Create Dataframe: Summary of split values
    :param hdtree_linear:
    :return:
        pd.Dataframe with columns: ["Level", "Feature", "feat", "value"]
            "Level": Tree-Level (Level 0 -  head node, Level 1 - first level in the tree ,...)
            "Feature": Full feature of the node: e.g. "feat2 < 2.02"
            "feat": only feature name: "feat2"
            "value": only the value: 2.02
    """
    # Get Tree Information, with splitting information as a string
    tree_string = hdtree_linear.__str__()

    # Define a regular expression pattern to match the relevant information
    # Retrieve splitting information "Level0", "feat2<2.02" ,...
    pattern = r'-*(Level \d+).*?split rule "(.*?) \(=.*?\' ([\d.]+)'

    # Use regular expression to find all matches in the string
    matches = re.findall(pattern, tree_string)

    # Save splitting information in a Dataframe.
    data = pd.DataFrame(columns=["Level", "Feature"], index=range(0, len(matches)))

    for index, (level, feat, value) in enumerate(matches):
        data.loc[index, "Level"] = level
        data.loc[index, "Feature"] = feat

        # Check: Does data["Feature"] always exist of "<" or is it
        # also sometimes expressed with ">" ?
        # if "<" in feat:
        #     print("String contains '<'")
        # elif ">" in feat:
        #     print("Hmmmm....... TO CHECK!!!!!! data['Feature'] contains '>'")
        # elif "no split rule" in feat:
        #     print("Okay, 'no split rule' in data['Feature'].")
        # else:
        #     print("Hmmmm....... TO CHECK!!!!!! data['Feature'] neither "
        #           "contains '>' nor '<' nor 'no split rule' ..... ???????? ")

    data[['feat', 'value']] = data['Feature'].str.split(' < ', expand=True)
    data['value'] = data['value'].astype(float)
    # pd_splits = pd.DataFrame(columns=["feature","split_value"], index=range(0,len(hdtree_linear._cached_predictions)))
    #
    # # Parent NODE
    # i_node=0
    # pd_splits.loc[i_node,"split_value"] = hdtree_linear._node_head._split_rule._state["split_value"]
    # pd_splits.loc[i_node,"feature"] = hdtree_linear._node_head._split_rule._state["split_attribute_indices"][0]
    #
    # hdtree_node_head = hdtree_linear._node_head
    #
    # # Child NODES
    # i_node=1
    # for ii in range(0, int(len(hdtree_linear._cached_predictions)/2-1)):
    #     for i_child in range(0,2):
    #         if hdtree_node_head._split_rule._child_nodes[i_child]._split_rule:
    #             pd_splits.loc[i_node,"split_value"] = \
    #                 hdtree_node_head._split_rule._child_nodes[i_child]._split_rule._state["split_value"]
    #             pd_splits.loc[i_node, "feature"] = \
    #                 hdtree_node_head._split_rule._child_nodes[i_child]._split_rule._state["split_attribute_indices"][0]
    #
    #         else:
    #             pd_splits.loc[i_node, "split_value"] = np.nan
    #             pd_splits.loc[i_node, "feature"] = np.nan
    #         i_node = i_node + 1
    #     # update latest child node
    #     hdtree_node_head = hdtree_node_head._split_rule._child_nodes[i_child]

    return data


def step2(x_train_new, x_test_new, epsilon,
          y_train, y_test,
          predicted_train, predicted_test,
          dec_tree_parameters,
          df_accuracies,
          df_y, df_y_test):
    """
    Step 2 of BAPC. AI model (RandomForestClassifier) on new corrected data.
    Trainingdata - Prediction of DT1 of training data
    Parameters
    ----------
    x_train_new: new x-trainingset
    x_test_new: new x-test set
    epsilon: new y-labels training set
    y_train: original y-labels trainingset
    y_test: original y-labels testset
    predicted_train: predicted y-labels from step 1 - trainingset -> for calculating surrogate : AI+DT1
    predicted_test: predicted y-labels from step 1 - testset
    dec_tree_parameters: dict - hyperparameters
    df_accuracies: pd.Dataframe - summary for accuracies & MAE for test&trainingdata
    df_y: pd.Dataframe - summary of y-labels & predictions:
            here add: surrogate_prediction
    df_y_test: pd.Dataframe - summary of y-predictions test set: here add: surrogate_prediction
    :return:
    """
    # split the previous test set into a new train and test for step 2
    # target here would be the difference vector calculated above
    # ind_new = int(len(x_test)*0.50)
    # x_train_new = x_test.loc[0:ind_new, K_featnames].round(2)
    # y_train_new = y_test[0:ind_new+1]
    # x_test_new = x_test.loc[ind_new+1:, K_featnames].round(2)
    # y_test_new = y_test[ind_new+1:]
    # x_train_new = np.array(x_train_new.values)

    # model = SVC(kernel='linear')
    ##
    # #model = DecisionTreeClassifier(max_depth=4, random_state=4, min_samples_split=3)#
    model = RandomForestClassifier(random_state=42,
                                   ccp_alpha=dec_tree_parameters["ccp_alpha"])
    # model = RandomForestClassifier(n_estimators=100, max_depth=4,
    #                                random_state=1,
    #                                ccp_alpha=dec_tree_parameters["ccp_alpha"])
    # #model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)    # Train the model on training data
#
    model.fit(x_train_new, epsilon)

    predicted_rf_test = model.predict(x_test_new)
    predicted_rf_train = model.predict(x_train_new)

    # For regression problems the metrics used to evaluate an algorithm are mean absolute error (MAE),
    # mean squared error (MSE), and root mean squared error (RMSE).
    df_accuracies.loc["MAE_AI", "DT1 + AI"] = np.around(mean_absolute_error(epsilon, predicted_rf_train), 3)

    print('Mean Absolute Error epsilon-epsilon_tilde:', df_accuracies.loc["MAE_AI", "DT1 + AI"])

    surr_predict = predicted_rf_train + predicted_train
    surr_predict_test = predicted_rf_test + predicted_test

    df_accuracies.loc["acc_test", "DT1 + AI"] = np.around(accuracy_score(y_test, surr_predict_test), 3)
    df_accuracies.loc["acc_train", "DT1 + AI"] = np.around(accuracy_score(y_train, surr_predict), 3)
    df_accuracies.loc["MAE_test", "DT1 + AI"] = np.around(mean_absolute_error(y_test, surr_predict_test), 3)
    df_accuracies.loc["MAE_train", "DT1 + AI"] = np.around(mean_absolute_error(y_train, surr_predict), 3)
    print("accuracy test data y_test & surrogate-predict-test " + str(df_accuracies.loc["acc_test", "DT1 + AI"]))
    print("accuracy training data y_train & surrogate-predict " + str(df_accuracies.loc["acc_train", "DT1 + AI"]))
##
    df_y["y_predicted_DT1+AI=y_pred_DT1+epsilon_hat"] = surr_predict

    df_y["epsilon"] = epsilon
    df_y["epsilon_hat"] = predicted_rf_train

    df_y_test["y_pred_DT1+AI"] = surr_predict_test
    df_y_test["epsilon_hat"] = predicted_rf_test

    return df_y, df_y_test, df_accuracies, \
        predicted_rf_train, predicted_rf_test,\
        surr_predict, surr_predict_test, model


def get_epsilon_value_for_person_id(df_y, person_id_label, person_id_to_check):
    check_for_eps_value = df_y[df_y[person_id_label] == person_id_to_check]["epsilon"].values[0]
    return check_for_eps_value


def error_amplification(company_datapoint_to_check,
                        x_train, hdtree_linear,
                        check_for_eps_value,
                        df_y,
                        circle_radius_factor,
                        circle_around_one_point=True):
    """

    :param company_datapoint_to_check:
    :param x_train:
    :param hdtree_linear:
    :param check_for_eps_value:
    :param df_y:
    :param circle_radius_factor:
    :param circle_around_one_point: Bool:
            True: Circle only around datapoint of interest
            False: Circle around all datapoints in the neighborhood,
                    belonging to the same category.

    :return:
        df_y: updated with error amplification
        circle_radius.
    """
    # Add new column "extended_eps"
    df_y.loc[:, "extended_eps"] = df_y.loc[:, "epsilon_hat"].copy()

    # Get values of decision boundaries (on which x-axis-values are there
    # decision boundaries? -> rec_regions_x_uni)
    # (on which y-axis-values are there
    # decision boundaries? -> rec_regions_y_uni)
    rec_regions_x_uni, rec_regions_y_uni, _, _, _ = get_dec_boundaries(x_train, hdtree_linear)

    # -----------------------------------------------------------
    # --------Neighborhood of rectangle of interest! ------------
    # -----------------------------------------------------------
    # Identify the rectangle boundaries of interest (in which rectangle is the
    # company-datapoint-of-interest located )
    rec_oi_x, rec_oi_y = rectangle_of_interest2(company_datapoint_to_check,
                                                rec_regions_x_uni,
                                                rec_regions_y_uni,
                                                x_train)



    # get subset of data with epsilon = 1
    df_y_sub = df_y[df_y["epsilon_hat"] == check_for_eps_value]

    idx = df_y_sub.index
    #
    # circle radius relative to range of datapoints (xlim, ylim)
    xymin = min(x_train.min())
    xymax = max(x_train.max())

    circle_radius = circle_radius_factor * (xymax - xymin) / 10

    for i_d_subb in idx:  # idx = indices of epsilon values
        i_d_centers = x_train.index.get_loc(i_d_subb)
        for i_data in x_train.index:
            # Is the datapoint within rectangle of interest?

            if is_within_range(rec_oi_x, x_train.loc[i_data, x_train.columns[0]]) and \
                    is_within_range(rec_oi_y, x_train.loc[i_data, x_train.columns[1]]):

                if circle_around_one_point:
                    # Circle only around datapoint of interest
                    xx = company_datapoint_to_check[0]
                    yy = company_datapoint_to_check[1]
                else:
                    # Circle around all datapoints in neighborhood (on boundary side),
                    # belonging to the same category
                    xx = x_train.loc[i_d_centers, x_train.columns[0]]
                    yy = x_train.loc[i_d_centers, x_train.columns[1]]

                # if is_within_circle(x_train.iloc[i_d_centers, 0], x_train.iloc[i_d_centers, 1],
                #                     x_train.iloc[i_data, 0], x_train.iloc[i_data, 1],
                #                     radius=circle_radius) and \
                #         does_need_relabeling(df_y, i_d_centers, i_data):
                # if is_within_circle(company_datapoint_to_check[0], company_datapoint_to_check[1],
                #                     x_train.iloc[i_data, 0], x_train.iloc[i_data, 1],
                #                     radius=circle_radius) and \
                #         does_need_relabeling(df_y, i_d_centers, i_data):
                if is_within_circle(xx, yy,
                                    x_train.loc[i_data, x_train.columns[0]],
                                    x_train.loc[i_data, x_train.columns[1]],
                                    radius=circle_radius) and \
                        does_need_relabeling(df_y, i_d_centers, i_data):
                    # is_above_below_boundary(x_train, i_data, i_d_subb, pd_splitting_vals,
                    #                        df_y, new_cols_df_y):
                    # print(i_data, i_d_sub)
                    # Reassign a new value -> [ 2*value(epsilon) ] to the datapoints within the circle
                    if i_data != i_d_centers:
                        if df_y.loc[i_data, "extended_eps"] == 0:
                            # df_y.loc[i_data, "extended_eps"] = 2 * df_y.loc[i_d_subb, "epsilon"]
                            df_y.loc[i_data, "extended_eps"] = df_y.loc[i_d_subb, "epsilon_hat"]

    # # ........TESTING............
    # # x_train_sorted = x_train.sort_values(by=("feat1","mean"))
    # x_train_sorted = x_train.sort_values(by=("feat2","mean"), ascending=False)
    #
    # # Target row values
    # target_values = [1.54, 3.34]
    # target_values = [2.64, 6.31]
    #
    # # Find the index of the row with target values
    # index = x_train.index[(x_train[('feat1','mean')] == target_values[0]) &
    #                    (x_train[('feat2','mean')] == target_values[1])].tolist()
    #
    # # .........For --Testing--- purposes:..............
    # i_data = 21
    # i_d_centers = index
    # #is_within_circle(x_train.iloc[i_d_sub, 0], x_train.iloc[i_d_sub, 1],
    # #                            x_train.iloc[i_data, 0], x_train.iloc[i_data ,1],
    # #                            radius=0.5)
    # print(is_above_below_boundary(x_train, i_data, i_d_subb, pd_splitting_vals,
    #                                         df_y, new_cols_df_y))
    # print(x_train.loc[i_data,:])
    # print(new_cols_df_y)
    # print(x_train.iloc[i_d_centers, :])


    # Set all data of epsilon back to their original values
    # (they got assigned to 2, if they were in the circle
    # neighborhood of a neighboring data point)
    df_y.loc[idx, "extended_eps"] = check_for_eps_value

    # Calculate new epsilon hat (epsilon - eps_extended), which shall be used for subtracting it from
    # original data, to get new dataset for STEP3
    df_y["eps_EA"] = df_y["epsilon_hat"] - df_y["extended_eps"]

    # df_y["x"] = x_train.iloc[:,0]
    # df_y["y"] = x_train.iloc[:,1]
    # df_y["eps_rect"] = 0
    # df_y["eps_x"] = np.where((df_y['x'] < 3.5) & (df_y['y'] < 4.5),
    #                          df_y['epsilon'].copy(), df_y['eps_rect'] )
    # #
    # df_y["Y-eps"] = df_y["y_train"] - df_y["eps_x"]

    return df_y, circle_radius, rec_oi_x, rec_oi_y


def correcting_wrong_prediction_min1_2(df_y):
    """
    Check that "y_epsilon_hat_ea" does not have values {-1} or {2}.
    This can be due to a wrong/missing AI prediction, then the error amplification can
    relabel the wrong predicted point and then y_train-eps_EA can lead to {-1, 2}
    :param df_y: pd.Dataframe
    :return:
    """
    for i_val in [-1,2]:
        if i_val in df_y["y_epsilon_hat_ea"].values:
            # Replace -1 with the corresponding values from "y_train"
            df_y.loc[df_y["y_epsilon_hat_ea"] == i_val, "y_epsilon_hat_ea"] = df_y.loc[
                df_y["y_epsilon_hat_ea"] == i_val, "y_train"]
            warnings.warn("We replace wrong AI prediction, and therefore wrong"
                    " value of " + str(i_val) + " in y_epsilon_hat_ea = "
                    "(ytrain - eps_EA), with original value of ytrain.")
    return df_y


def step3_4(dec_tree_params, x_train_new, eps_hut_int, x_test_new,
          y_test, y_train,
          df_accuracies, add_name,
          df_y, df_y_test,
          which_step="DT3"):

    randomstate = dec_tree_params["randomstate"]
    trainingset = dec_tree_params["trainingset"]
    max_depth = dec_tree_params["max_depth"]
    min_samples_leaf = dec_tree_params["min_samples_leaf"]
    min_samples_split = dec_tree_params["min_samples_split"]
    ccp_alpha = dec_tree_params["ccp_alpha"]
    nametosave = dec_tree_params["nametosave"]
    save_to_folder = dec_tree_params["save_to_folder"]

    clf3 = DecisionTreeClassifier(splitter="best",
                                  max_depth=max_depth, random_state=42,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  min_impurity_decrease=0.01,
                                  ccp_alpha=dec_tree_params["ccp_alpha"],
                                  criterion='gini')
    clf3.fit(x_train_new, eps_hut_int)
    # clf3.fit(x_train, eps_hut_int)

    predicted3_train = clf3.predict(x_train_new)
    predicted3_test = clf3.predict(x_test_new)
    # predicted3_train = clf3.predict(x_train)
    # predicted3_test = clf3.predict(x_test)

    df_accuracies.loc["acc_test", which_step] = np.around(accuracy_score(y_test, predicted3_test), 3)
    # TODO: to check: sollte hier nicht anstatt y_train -> eps_hut_int stehen?
    df_accuracies.loc["acc_train", which_step] = np.around(accuracy_score(y_train, predicted3_train), 3)
    df_accuracies.loc["MAE_test", which_step] = np.around(mean_absolute_error(y_test, predicted3_test), 3)
    df_accuracies.loc["MAE_train", which_step] = np.around(mean_absolute_error(y_train, predicted3_train), 3)

    df_accuracies.to_excel(save_to_folder + 'df_accuracies' + add_name + '.xlsx', index=True)

    print('base model accuracy test data', df_accuracies.loc["acc_test", which_step])
    print('base model accuracy train data', df_accuracies.loc["acc_train", which_step])

    df_y.loc[:,"y_predicted_" + which_step] = predicted3_train
    df_y_test["y_predicted_test_" + which_step] = predicted3_test

    # leaf ids
    node_indicator = clf3.decision_path(x_train_new)
    leaf_id = clf3.apply(x_train_new)
    df_y.loc[:,"leaf_id_" + which_step] = leaf_id

    df_y_copy = df_y.copy()
    df_y_copy["DT1-DT3_" + add_name] = df_y_copy["y_predicted_DT1"] - df_y_copy["y_predicted_DT3"]

    #df_y.loc[:, "DT1-DT3_"+ add_name] = df_y["y_predicted_DT1"] - df_y["y_predicted_DT3"]
    df_y = df_y_copy.copy()
    df_y.to_excel(save_to_folder + 'df_y_' + add_name + '.xlsx', index=True)

    return df_accuracies, df_y, df_y_test, clf3


def get_person_id_with_given_datapoint(df_y_test,
                                       which_feature_set,
                                       datapoint_of_interest):
    df_y_test_sub = df_y_test[(df_y_test[which_feature_set[0]] == datapoint_of_interest[0]) &
                                 (df_y_test[which_feature_set[1]] == datapoint_of_interest[1])]
    return df_y_test_sub


def get_datapoint_oi_given_person_id(person_id_to_check,
                                     agg_data, company_id_label,
                                     which_feature_set):
    n_feat = len(which_feature_set)
    # Get datapoint_of_interest, with given companyID
    if person_id_to_check is not None:
        datapoint_of_interest = []
        for i in range(n_feat):
            a = agg_data[agg_data[company_id_label] == person_id_to_check][which_feature_set[i]]
            datapoint_of_interest.append(a.values[0])
        # b = agg_data[agg_data[company_id_label] == person_id_to_check][which_feature_set[1]]
        # datapoint_of_interest.append(b.values[0])

    return datapoint_of_interest


def get_y_train_label_for_person_id(person_id_to_check,
                                    person_id_label,
                                    df_y_big):
    """
    Get the corresponding y_train-label for the given Person_id of interest.
    :param person_id_to_check: int (given in Index column of agg_data)
    :param df_y_big: pandas dataframe
    :return: y_train_oi
    """
    #if person_id_to_check is not None:
    y_train_oi = df_y_big[df_y_big[person_id_label]==person_id_to_check]["y_train"].values[0]
        # if y_train_oi != y_true_c_oi:
        #     warnings.warn("for check_for_eps_value = ..." + " , we"
        #                   "set y_true_c_oi = " + str(y_true_c_oi) + ". But this does not "
        #                   "coincide with y_train of our person_id_to_check: y_train = " + str(y_train_oi) +
        #                   ". TO CHECK!!!!")
    # elif person_id_to_check is None:
    #     subset_df_datapoint_oi = df_y_big[(df_y_big[which_feature_set[0]] == datapoint_of_interest[0]) &
    #                          (df_y_big[which_feature_set[1]] == datapoint_of_interest[1])]
    #     if len(subset_df_datapoint_oi) == 1:
    #         y_train_oi = subset_df_datapoint_oi["y_train"].values[0]
    #     elif len(subset_df_datapoint_oi) > 1:
    #         warnings.warn("Please set a Person_ID in 'person_id_to_check', \n because "
    #                       "we have multiple persons on the datapoint_of_interest."
    #                       "\n Choose between \n" + person_id_label + " = [" +
    #                       str(subset_df_datapoint_oi[person_id_label].values) + "], with"
    #                      "\ny_train values = [" + str(str(subset_df_datapoint_oi["y_train"].values) + "]."))
    #         print("y_true_c_oi set to = " + str(y_true_c_oi))
    return y_train_oi


def fix_features_step3(fix_features_of_step3,
                       fix_features_s1_s3,
                       fix_step1,
                       features_df_sorted3,
                       x_train,
                       x_test):
    x_train_copy = x_train.copy()
    x_test_copy = x_test.copy()

    # FIX FEATURES OF STEP 3
    if fix_features_of_step3:
        # drop 0-features
        df_sub_feat_step3 = features_df_sorted3[features_df_sorted3.loc[:, 'importances3'] != 0]

        # rename features: remove "(", ")" , to match the columnames of x_train
        df_sub_feat_step3["features"] = df_sub_feat_step3["features"].str.replace('(', '/')
        df_sub_feat_step3["features"] = df_sub_feat_step3["features"].str.replace(')', '')

        sub_feat_step3 = df_sub_feat_step3["features"].tolist()
        # sub_feat_step3 = ['Company Size/sum', 'Willingness to compromise/variety_index',
        #                  'Orientation on norms/mean', 'Assertiveness/mean', 'Technical/mean']
        # rename column names

        x_train_copy.columns = x_train_copy.columns.get_level_values(0) + '/' + x_train_copy.columns.get_level_values(1)
        x_test_copy.columns = x_test_copy.columns.get_level_values(0) + '/' + x_test_copy.columns.get_level_values(1)

        # sub_feat_step3 = ['Assertiveness/mean', 'Technical/mean',
        #                  'Administrating processes/mean',
        #                  'Administrating processes/variety_index',
        #                  'Self-Discipline/mean',
        #                  'Company Size/sum']
        # sample_weigth = ['Assertiveness/mean', 'Technical/mean','Administrating processes/mean']
        x_train_new = x_train_copy[sub_feat_step3]
        x_test_new = x_test_copy[sub_feat_step3]
    if fix_features_s1_s3:
        # drop 0-features
        df_sub_feat_step3 = features_df_sorted3[features_df_sorted3.loc[:, 'importances3'] != 0]
        df_sub_feat_step3 = df_sub_feat_step3.loc[:, 'importances3'].dropna()

        # rename features: remove "(", ")" , to match the columnames of x_train
        df_sub_feat_step3.index = df_sub_feat_step3.index.str.replace('(', '/')
        df_sub_feat_step3.index = df_sub_feat_step3.index.str.replace(')', '')

        sub_feat_step3 = df_sub_feat_step3.index.tolist()

        # sub_feat_step3 = ['Company Size/sum', 'Willingness to compromise/variety_index',
        #                  'Orientation on norms/mean', 'Assertiveness/mean', 'Technical/mean']
        # rename column names
        if not fix_step1:
            x_train_copy.columns = x_train_copy.columns.get_level_values(0) + '/' + x_train_copy.columns.get_level_values(1)
            x_test_copy.columns = x_test_copy.columns.get_level_values(0) + '/' + x_test_copy.columns.get_level_values(1)

        # sub_feat_step3 = ['Assertiveness/mean', 'Technical/mean',
        #                  'Administrating processes/mean',
        #                  'Administrating processes/variety_index',
        #                  'Self-Discipline/mean',
        #                  'Company Size/sum']
        # sample_weigth = ['Assertiveness/mean', 'Technical/mean', 'Administrating processes/mean']
        x_train_new = x_train_copy[sub_feat_step3]
        x_test_new = x_test_copy[sub_feat_step3]
    return x_train_new, x_test_new


def draw_graph(clsf, x_training, savetofolder, addname, whichstep):
    dot_data = tree.export_graphviz(clsf, feature_names=x_training.columns, filled=True)
    graph = graphviz.Source(dot_data, format="png")

    dir_to_dt3 = savetofolder + "_dt_step" + \
                 whichstep + addname
    save_files(dir_to_dt3, fig=None, what_to_save="graphviz-tree", graph=graph)

    #graph.view()
    return


def calc_local_epsilon_hat(df_y):#, leaf_nr):
    # for each leaf, take the values of epsilon_hat (AI predicted epsilon)
    # in the region of this leaf (datapoints within the decision boundaries),
    # for rest of the datapoints set new epsilon = 0.

    list_leaves = df_y["leaf_id_DT1"].unique()
    local_epsilon_hat = {}
    add_name_list = []

    for leaf_nr in list_leaves:
        df_y["leaf" + str(leaf_nr)] = df_y["leaf_id_DT1"] == leaf_nr
        add_name_list.append("_leaf" + str(leaf_nr))

        # new column: local epsilon_hat
        # (übernehme Werte von epsilon_hat für Datenpunkte im leaf, else 0)
        df_y["epsilon_leaf" + str(leaf_nr)] = 0
        for i in range(df_y.shape[0]):
            if df_y["leaf"+ str(leaf_nr)].iloc[i]:
                df_y.loc[i,"epsilon_leaf" + str(leaf_nr)] = df_y.loc[i, "epsilon_hat"]
                # TODO: !!! change epsilon to epsilon_hat!

        local_epsilon_hat["leaf" + str(leaf_nr)] = df_y["epsilon_leaf" + str(leaf_nr)].to_numpy()

    return local_epsilon_hat, df_y, add_name_list


def merge_df_y_diff_leaves(df_y, df_y_merged, i_leaves, ii,
                           save_to_folder, add_name):
    df_y_sub = df_y[df_y[i_leaves]].copy()
    df_y_sub.rename(columns={"DT1-DT3_"+i_leaves: "DT1-DT3"}, inplace=True)

    if ii == 0:
        df_y_merged = df_y_sub.copy()
    else:
        df_y_merged = pd.concat([df_y_merged, df_y_sub], axis=0)

    df_y_merged.to_excel(save_to_folder + 'df_y_MERGED_' + add_name + '.xlsx',
                         index=True)

    return df_y_merged


def feature_importance_per_leaf(clf, nodes):
    """
    Calculates Feature Importance per leaf. Considers only the features / nodes
    that lead to the specific leaf.
    :param clf: classifier object of DecisionTreeClassifier
    :param nodes: dict, keys: features of Decision Tree,
                        values: [[int(parent node), int(child_left), int(child_right)]]
    :return: pd_fi: pandas.DataFrame with feature importances
    """
    fi_raw = {}
    fi = {}
    list_nodes = list(nodes.keys())
    n_all_samples = clf.tree_.n_node_samples[0]

    # calculate raw feature importances:
    # gini_parent_node*sample_weigth - gini_left_child*sample_weight - gini_right_child*sample_weight
    for i_n in range(len(list_nodes)):
        fi_list = []
        for i_f1 in range(0, len(nodes[list_nodes[i_n]])):
            nodes_fi = nodes[list_nodes[i_n]][i_f1]
            fi_list.append(
                clf.tree_.impurity[nodes_fi[0]] * clf.tree_.n_node_samples[nodes_fi[0]] / n_all_samples - \
                clf.tree_.impurity[nodes_fi[1]] * clf.tree_.n_node_samples[nodes_fi[1]] / n_all_samples - \
                clf.tree_.impurity[nodes_fi[2]] * clf.tree_.n_node_samples[nodes_fi[2]] / n_all_samples)
        print("node impurities:")
        print(clf.tree_.impurity[nodes_fi[0]])
        print(clf.tree_.impurity[nodes_fi[1]])
        print(clf.tree_.impurity[nodes_fi[2]])
        fi_raw[list_nodes[i_n]] = fi_list

    # sum of the feature importances of all nodes
    sum_fi_all_nodes = 0
    for i_n in range(len(list_nodes)):
        for i_f1 in range(0, len(nodes[list_nodes[i_n]])):
            sum_fi_all_nodes = sum_fi_all_nodes + fi_raw[list_nodes[i_n]][i_f1]

    for i_n in range(len(list_nodes)):
        fi[list_nodes[i_n]] = sum(fi_raw[list_nodes[i_n]]) / sum_fi_all_nodes

    pd_fi = pd.DataFrame.from_dict(fi, orient='index')
    pd_fi.sort_values(0, inplace=True)
    return pd_fi


def plot_feature_importance_per_leaf(pd_fi, title,
                                     save_to_folder, color=["lavenderblush", "lightpink", "black"]):

    fig, ax = plt.subplots(figsize=[5, len(pd_fi)/2])
    width = 0.75  # the width of the bars
    ind = np.arange(len(pd_fi))  # the x locations for the groups

    cmap = mcolors.LinearSegmentedColormap.from_list("", color)#["red", "yellow", "green"]

    #plt.bar(df.index, df["x"], color=cmap(df.x.values / df.x.values.max()))

    bars = ax.barh(ind + 0.33, pd_fi.iloc[:,0], width,
                   color=cmap(pd_fi.iloc[:,0].values / pd_fi.iloc[:,0].values.max()))

    ax.bar_label(bars, fontsize=12)

    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)

    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(pd_fi.index, minor=False, fontsize=12)
    plt.title(title)

    plt.savefig(save_to_folder + "Plot_FeatureImportances_" +
                title + ".png", dpi=300, format='png',
                bbox_inches='tight')
    return ax



def count_decimals(num):
    """
    # Function to count decimals in a number
    :param num: float
    :return:
    """
    num_str = str(num)
    if '.' in num_str:
        return len(num_str.split('.')[1])
    return 0


def calc_grid_resolution(x_train):
    """
    Calculate grid resolution. How many datapoints with stepsize xsteps=0.01,
    ysteps=0.01 (resolution of x_train datapoints)
    :param x_train: pd.Dataframe
    :return: grid_res: int
    """
##
    # Get Accuracy of Datapoints (With how many decimals are they represented?)
    # Iterate over columns of x_train and calculate max accuracy/
    # number of decimal places for getting the stepsize (xsteps, ysteps)
    max_accuracy = {}
    for col in x_train.columns:
        decimals = [count_decimals(val) for val in x_train[col]]
        max_accuracy[col] = max(decimals)
    # Get Boundaries of dataset (Minimal values + Maximal values)
    feat1_x = x_train.iloc[:, 0]
    feat2_y = x_train.iloc[:, 1]
    xmin = np.min(feat1_x)
    xmax = np.max(feat1_x)
    xsteps = 1 * 10 ** -max_accuracy[x_train.columns[0]]  # xsteps = 0.01
    #xrange = np.arange(xmin, xmax, xsteps)
    ymin = np.min(feat2_y)
    ymax = np.max(feat2_y)
    ysteps = 1 * 10 ** -max_accuracy[x_train.columns[1]]
    #yrange = np.arange(ymin, ymax, ysteps)
##
    # TODO: at the moment only valid if xmin = ymin and xmax = ymax (if datapoints lay
    #  exactly in a squared dataspace)
    # The grid_resolution of DecisionBoundaryDisplay is calculated starting from
    # the minimum value of x - 1 and ending an the maximum value of x + 1
    # (probably the same for y-values, but here in this case, we have a squared dataspace.
    gridrange = np.arange(xmin - 1, xmax + 1 + xsteps, xsteps)

    # Grid resolution
    grid_res = len(gridrange)

    return grid_res

##
def plot_simdata(clf, x_train, colorlabels,
                 save_to_folder,
                 company_datapoint_to_check=[],
                 title="",
                 which_step="step1",
                 fs=12,
                 linecolor="rosa",
                 lw=2, alpha=1):
    """
    Plot simulated data, with decision boundaries, for different cases:
    Step1, Step2, Step3, errorAmplification

    :param clf: classifier
    :param x_train:
    :param colorlabels: y-labels
    :param save_to_folder: str
    :param company_datapoint_to_check: list of float: eg. = [3.5, 4]
    :param title: str
    :param which_step: str "step1"|"step2"|"step3"|"error_ampl"
    :param fs: int : fontsize
    :param linecolor:
    :param lw:
    :param alpha:
    :return:
    """

    if clf is not None:
        # ------------------------
        # Plot Decision Boundaries
        # ------------------------
        # Grid Resolution
        #
        grid_res = calc_grid_resolution(x_train)
        if linecolor == "rosa":
            cmap = plt.cm.coolwarm
        else:
            cmap = plt.cm.Spectral
#
        disp = DecisionBoundaryDisplay.from_estimator(clf,
                                                      x_train,
                                                      grid_resolution=grid_res,
                                                      response_method="predict",
                                                      xlabel=x_train.columns[0],
                                                      ylabel=x_train.columns[1],
                                                      alpha=0.5,
                                                      plot_method="contour",  # ,{‘contourf’, ‘contour’, ‘pcolormesh’}
                                                      cmap=cmap)
        fig = disp.figure_
        ax = disp.ax_
    else:
        fig, ax = plt.subplots()

    # ------------------
    # Plot Datapoints
    # ------------------
    # Define color mapping
    if which_step=="step1" or which_step=="step3":
        color_map = {0: 'darkblue', 1: 'gold'}#, 2:"red"}
    elif which_step=="step2":
        color_map = {-1: 'mediumvioletred',
                     0: 'papayawhip',
                     1: 'yellowgreen'}
    elif which_step=="error_ampl":
        color_map = {#-2: 'darkmagenta',
                     -1: 'mediumvioletred',
                     0: 'papayawhip',
                     1: 'yellowgreen'}#,
                     #2: 'olivedrab'}
    else:
        raise ValueError("colormap is not initialized properly,"
                         "because we need 'which_step' to be"
                         " either step1, step2, step3 in "
                         " the input of plot_simdata() ")

    # Assign colors based on y_label values
    colors = [color_map[y] for y in colorlabels]

    ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=colors,
               label=colors, alpha=0.5)  # , label=labels)
    # Add company-data-point-on interest
    if len(company_datapoint_to_check) > 0:
        ax.scatter(company_datapoint_to_check[0],
                   company_datapoint_to_check[1],
                   marker='x', color='crimson', s=100, linewidth=3)

    # Create legend entries
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=8)
                     for color in color_map.values()]
    legend_keys = color_map.keys()

    # Add legend
    ax.legend(legend_labels, legend_keys, loc="upper right")

    np.max(x_train)
    ax.set_xlim([np.min(x_train)-0.2, np.max(x_train) + 0.2])
    ax.set_ylim([np.min(x_train)-0.2, np.max(x_train) + 0.2])
    ax.set_xlabel(x_train.columns[0], fontsize=fs)
    ax.set_ylabel(x_train.columns[1], fontsize=fs)
    ax.set_xticks(np.arange(np.min(x_train), np.max(x_train)+0.5, 1))
    ax.set_yticks(np.arange(np.min(x_train), np.max(x_train)+0.5, 1))

    ax.set_title(title)
    fig.savefig(save_to_folder + "_plot_" + title + ".png")

    return
##


def transform_string_to_int(arr):
    # First, try converting the array to integers
    try:
        arr_int = arr.astype(int)
        return arr_int
    except ValueError:
        # If ValueError occurs, it means some elements have decimal points, so convert to float and then to int
        arr_float = arr.astype(float)
        arr_int = arr_float.astype(int)
        return arr_int


def change_ylabel_duplicated_datapoints_ambiguous_ylabel(agg_data,
                                                         which_feature_set,
                                                         ylabel):
    """
    If for the same datapoint, multiple ambiguous ylabel exist:
    Rename the ylabels, if for in one category we have 2 or more datapoints,
    then in the other category.
    E.g. ylabel = 0 -> 5 datapoints
         ylabel = 1 -> 2 datapoints  -> change their ylabel to 0.
    :return:
        agg_data,
        result: ylabel-counts per datapoint
        result2: ylabel-counts per datapoint after changing the ylabels.
    """
    # Combine datapoints (x=1, y=2 --> points_combined=12)
    agg_data["points_combined"] = agg_data[which_feature_set[0]].astype(str) + \
                                  agg_data[which_feature_set[1]].astype(str)
    agg_data["points_combined"] = agg_data["points_combined"].astype(int)
    #
    # How often does a combination of datapoints occur with value 0,
    # and 1 in the Leadership column.
    result = agg_data.groupby(['points_combined', ylabel]).size().reset_index(name='count')

    # Add new column, original ylabels to column: ylabel+'_original'
    agg_data[ylabel + '_original'] = agg_data[ylabel]
    #
    # Change ylabel if the same datapoint has 2 or more entries in one category of [0,1],
    # then in the other category
    for i_row in result["points_combined"].unique():
        if len(result[result["points_combined"] == i_row]["count"]) > 1:
            if np.abs(result[result["points_combined"] == i_row]["count"].iloc[0] -
                      result[result["points_combined"] == i_row]["count"].iloc[1]) >= 2:
                if result[result["points_combined"] == i_row]["count"].iloc[0] > \
                        result[result["points_combined"] == i_row]["count"].iloc[1]:
                    a = result[result["points_combined"] == i_row][ylabel].iloc[0].copy()
                    subset = agg_data.loc[agg_data["points_combined"] == i_row].copy()
                    subset[ylabel] = a
                    agg_data.loc[agg_data["points_combined"] == i_row] = subset

                elif result[result["points_combined"] == i_row]["count"].iloc[1] > \
                        result[result["points_combined"] == i_row]["count"].iloc[0]:
                    b = result[result["points_combined"] == i_row][ylabel].iloc[1].copy()
                    subset = agg_data.loc[agg_data["points_combined"] == i_row].copy()
                    subset[ylabel] = b
                    agg_data.loc[agg_data["points_combined"] == i_row] = subset

    # Test: How often does a combination of datapoints occur with value 0,
    # and 1 in the Leadership column.
    result2 = agg_data.groupby(['points_combined', ylabel]).size().reset_index(name='count')

    return agg_data, result, result2
##

def get_dec_boundaries(X, clf):
    """
    from classifier, build a meshgrid, and identify decision boundaries.
    Return: list of decision boundaries, in x-direction and in y-direction.
    :param X:
    :param clf:
    :return: rec_regions_x,rec_regions_y
    """
    h = 0.01
    x_min, x_max = X.iloc[:, 0].min() - 10 * h, X.iloc[:, 0].max() + 10 * h
    y_min, y_max = X.iloc[:, 1].min() - 10 * h, X.iloc[:, 1].max() + 10 * h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    #
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #
    #
    # If entries of the ndarray are of type string:
    # transform to integer
    if np.any(np.issubdtype(Z.dtype, np.str_)):
        Z = transform_string_to_int(Z)

    # INFER RECTANGULAR REGIONS / BOUNDARIES

    # Rectangular regions
    rectangular_regions_x = []
    rectangular_regions_y = []
    rectangular_regions_x.append(0)
    rectangular_regions_y.append(0)
    # Iterate through the rows and columns of Z
    for i in range(Z.shape[0] - 1):
        for j in range(Z.shape[1] - 1):
            # Check if the class labels at the four corners of the square differ
            if Z[i, j] != Z[i + 1, j]:
                # If they differ, this indicates a decision boundary
                y_b = yy[i, j]
                # Append the rectangular region
                rectangular_regions_y.append(y_b)
            elif Z[i, j] != Z[i, j + 1]:
                x_b = xx[i, j]
                rectangular_regions_x.append(x_b)

    rectangular_regions_x.append(yy[i, j])  # last value
    rectangular_regions_y.append(xx[i, j])  # last value

    rectangular_regions_x_unique = np.unique(rectangular_regions_x)
    rectangular_regions_y_unique = np.unique(rectangular_regions_y)
    print("regions/boundaries x-axis:")
    print(rectangular_regions_x_unique)
    print("regions/boundaries y-axis:")
    print(rectangular_regions_y_unique)

    if len(rectangular_regions_x_unique) > 3:
        warnings.warn("TODO: We have more than 1 boundary in x-direction. "
                      "TO CHECK...")
    if len(rectangular_regions_y_unique) > 3:
        warnings.warn("TODO: We have more than 1 boundary in y-direction. "
                      "TO CHECK...")

    return rectangular_regions_x_unique, \
        rectangular_regions_y_unique, \
        xx, yy, Z


def build_pairs_of_region_boundaries(rectangular_regions_x, rectangular_regions_y):
    # Build pairs of rectangular region boundaries: rec_regions_x = [[0, 1.54],[1.54,7]]
    # input: rectangular_regions_x = [0, 1.54, 7]

    rec_regions_x = []
    for i in range(len(rectangular_regions_x) - 1):
        pair = [rectangular_regions_x[i], rectangular_regions_x[i + 1]]
        rec_regions_x.append(pair)
    rec_regions_y = []
    for i in range(len(rectangular_regions_y) - 1):
        pairy = [rectangular_regions_y[i], rectangular_regions_y[i + 1]]
        rec_regions_y.append(pairy)
    print("pairs of rec_regions_x:")
    print(rec_regions_x)
    print("pairs of rec_regions_y:")
    print(rec_regions_y)

    return rec_regions_x, rec_regions_y
##

def plot_decision_boundary(clf, X, Y_colorlabels,
                           save_to_folder,
                           company_datapoint_to_check=[],
                           edgecolor=[],
                           circle_radius=None,
                           plot_color_area=True,
                           which_step="step1",
                           fs=12,
                           title="",
                           cmap='Set3',
                           alpha=0.5):
    """
    Plot Decision Boundaries for HDTree case.
    Based on predictions of the classifier.
    :param clf: classifier
    :param X: pd.Dataframe
    :param Y_colorlabels: ndarray, of Y-vector
    :param save_to_folder: str
    :param company_datapoint_to_check: list of float, e.g. = [4.5, 3], plot one datapoint
    :param plot_color_area: Boolean, Fill the areas of decision boundaries with color?
    :param which_step: str "step1", "step3", difines which colormaps to use
    :param fs: int, fontsize
    :param title: str "title of the Plot"
    :param cmap: str, colormap for area-coloring for decision boundaries
    :return: rec_regions_x, rec_regions_y:
                list of lists, with ranges of boundaries [[0, 1.54], [1.54, 7]]
    """
    # Get Decision boundaries
    rec_regions_x, rec_regions_y, xx, yy, Z = get_dec_boundaries(X, clf)
#
    #
    #  ----  Colors Scatterplot  ------------
    # Define color mapping
    if which_step == "step1" or which_step == "step3":
        color_map = {0: 'darkblue', 1: 'gold'}#, #-1: 'green'}  # , 2:"red"}
    elif which_step == "step2":
        color_map = {-1: 'mediumvioletred',
                     0: 'papayawhip',
                     1: 'yellowgreen'}
    elif which_step == "error_ampl":
        color_map = {  # -2: 'darkmagenta',
            #-2: 'violet',
            -1: 'mediumvioletred',
            0: 'papayawhip',
            1: 'yellowgreen'}#,
        #2: 'green'}  # ,
        # 2: 'olivedrab'}
    else:
        raise ValueError("colormap is not initialized properly,"
                         "because we need 'which_step' to be"
                         " either step1, step2, step3 in "
                         " the input of plot_simdata() ")
    # Assign colors based on y_label values
    colors = [color_map[y] for y in Y_colorlabels]


    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # ----------------------------------------
    # Plot Decision Boundaries
    # -----------------------------------------
    if plot_color_area:
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    ax.contour(xx, yy, Z, colors='salmon', linewidths=1)

    # ----------------------------------------
    # Scatter-Plot
    # ----------------------------------------
    if len(edgecolor)>0:
        ax.scatter(X.iloc[:,0], X.iloc[:,1], c=colors,
                     edgecolor=edgecolor, lw=2)
    else:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors,
                   alpha=alpha)


    # Plot datapoint of interest
    if len(company_datapoint_to_check) > 0:
        ax.scatter(company_datapoint_to_check[0],
                   company_datapoint_to_check[1],
                   marker='x', color='crimson', s=100, linewidth=3)

    # Add Circle for Fidelity Neighborhood
    if circle_radius is not None:
        # Create a circle using patches.Circle
        circle = patches.Circle((company_datapoint_to_check[0],
                                 company_datapoint_to_check[1]),
                                circle_radius, fill=False,
                                color='black', linestyle='dotted')
        # Add the circle to the Axes object
        ax.add_patch(circle)

    # Create legend entries
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=8)
                     for color in color_map.values()]
    legend_keys = color_map.keys()
    # Add legend
    ax.legend(legend_labels, legend_keys, loc="upper right")
    ax.set_xlim([np.min(X)-0.2,np.max(X)+0.2])
    ax.set_ylim([np.min(X)-0.2,np.max(X)+0.2])
    ax.set_xlabel(X.columns[0], fontsize=fs)
    ax.set_ylabel(X.columns[1], fontsize=fs)
    ax.set_xticks(np.arange(np.min(X), np.max(X)+0.5, 1))
    ax.set_yticks(np.arange(np.min(X), np.max(X)+0.5, 1))

    ax.set_title(title)

    fig.savefig(save_to_folder + "_plot_HDTREE_" + title +
                 ".png")
#
    return rec_regions_x, rec_regions_y

##

def plot_decision_boundary_two_dbound(clf1, clf2,
                                      X, Y_colorlabels,
                                      save_to_folder,
                                      edgecolor=[],
                                      company_datapoint_to_check=[],
                                      circle_radius=None,
                                      plot_color_area=True,
                                      which_step="step1",
                                      fs=12,
                                      title="",
                                      cmap='Set3'):
    """
    Plot Decision Boundaries for HDTree case.
    Based on predictions of the classifier.
    :param clf: old classifier, old decision boundaries
    :param clf: new classifier, new decision boundaries of step3
    :param X: pd.Dataframe
    :param Y_colorlabels: ndarray, of Y-vector
    :param save_to_folder: str
    :param edgecolor:
    :param company_datapoint_to_check:
    :param plot_color_area: Boolean, Fill the areas of decision boundaries with color?
    :param which_step: str "step1", "step3", difines which colormaps to use
    :param fs: int, fontsize
    :param title: str "title of the Plot"
    :param cmap: str, colormap for area-coloring for decision boundaries
    :return:
    """

    h = 0.01
    x_min, x_max = X.iloc[:,0].min() - 10*h, X.iloc[:,0].max() + 10*h
    y_min, y_max = X.iloc[:,1].min() - 10*h, X.iloc[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)
    Z2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z2 = Z2.reshape(xx.shape)

    #
    # If entries of the ndarray are of type string:
    # transform to integer
    if np.any(np.issubdtype(Z1.dtype, np.str_)):
        Z1 = transform_string_to_int(Z1)
    if np.any(np.issubdtype(Z2.dtype, np.str_)):
        Z2 = transform_string_to_int(Z2)

    #  ----  Colors Scatterplot  ------------
    # Define color mapping
    if which_step == "step1" or which_step == "step3":
        color_map = {0: 'darkblue', 1: 'gold'}#, 2: 'tomato'}  # , 2:"red"}
    elif which_step == "step2":
        color_map = {-1: 'mediumvioletred',
                     0: 'papayawhip',
                     1: 'yellowgreen'}
    elif which_step == "error_ampl":
        color_map = {  # -2: 'darkmagenta',
            -1: 'mediumvioletred',
            0: 'papayawhip',
            1: 'yellowgreen'}  # ,
        # 2: 'olivedrab'}
    else:
        raise ValueError("colormap is not initialized properly,"
                         "because we need 'which_step' to be"
                         " either step1, step2, step3 in "
                         " the input of plot_simdata() ")
    # Assign colors based on y_label values
    colors = [color_map[y] for y in Y_colorlabels]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # ----------------------------------------
    # Plot Decision Boundaries
    # -----------------------------------------
    ax.contour(xx, yy, Z1, colors='salmon', linewidths=1, alpha=0.7)
    if plot_color_area:
        ax.contourf(xx, yy, Z2, cmap=cmap, alpha=0.25)
    ax.contour(xx, yy, Z2, colors='green', linewidths=1.4)

    # ----------------------------------------
    # Scatter-Plot
    # ----------------------------------------
    if len(edgecolor)>0:
        ax.scatter(X.iloc[:,0], X.iloc[:,1], c=colors,
                     edgecolor=edgecolor, lw=2)
    else:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors)
    # Add datapoint of interest
    if len(company_datapoint_to_check)>0:
        ax.scatter(company_datapoint_to_check[0],
                   company_datapoint_to_check[1],
                   marker='x', color='crimson', s=100, linewidth=3)

    # Add Circle for Fidelity Neighborhood
    if circle_radius is not None:
        # Create a circle using patches.Circle
        circle = patches.Circle((company_datapoint_to_check[0],
                                company_datapoint_to_check[1]),
                                circle_radius, fill=False,
                                color='black', linestyle='dotted')
        # Add the circle to the Axes object
        ax.add_patch(circle)

    # Create legend entries
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=8)
                     for color in color_map.values()]
    legend_keys = color_map.keys()
    # Add legend
    ax.legend(legend_labels, legend_keys, loc="upper right")
    ax.set_xlim([np.min(X)-0.2,np.max(X)+0.2])
    ax.set_ylim([np.min(X)-0.2,np.max(X)+0.2])
    ax.set_xlabel(X.columns[0], fontsize=fs)
    ax.set_ylabel(X.columns[1], fontsize=fs)
    ax.set_xticks(np.arange(np.min(X), np.max(X)+0.5, 1))
    ax.set_yticks(np.arange(np.min(X), np.max(X)+0.5, 1))

    ax.set_title(title)

    # Rename title, if there are more than two "-" in the title.
    if len(title.split('-')) > 2:
        title_l = title.split('-')[0:2]
        title = ''.join(title_l).replace(' ', '_')

    fig.savefig(save_to_folder + "_plot_HDTREE_" + title +
                 ".png")
#
    return

##
def map_epsilon_to_color(value):
    """ Maps values 0,1 to "white" or "black" """
    if value == 0:
        return 'none'
    elif value == 1 or value == -1:
        return 'black'
    else:
        return 'unknown'  # You can specify a default value for other cases if needed



##

def plot_simdata_two_dboundaries(clf, clf2,
                                 x_train,
                                 colorlabels,
                                 save_to_folder,
                                 edgecolor=[],
                                 company_datapoint_to_check=[],
                                 circle_radius=None,
                                 title='',
                                 which_step="step3",
                                 local_epsilon_hat=None,
                                 fs=12,
                                 linecolor="rosa",
                                 linecolor2="blue",
                                 lw=2, alpha=1):
    """
    Scatter-Plot with decision boundaries of two classifiers
    :param clf: Classifier 1 (old classifier for comparison)
    :param clf2: Classifier 2 (new classification after Step3)
    :param x_train: pd.Dataframe
    :param colorlabels:
    :param save_to_folder: str
    :param edgecolor: column of pd.Dataframe
    :param company_datapoint_to_check: list
    :param title: str
    :param which_step: str ("step3")
    :param local_epsilon_hat:
    :param fs: int, fontsize
    :param linecolor: str, "rosa" or something else (will take either coolwarm or PuOr colormaps)
    :param linecolor2: str, "blue" or sth else (will take either Set3, or RdYlGn colormaps)
    :param lw:
    :param alpha:
    :return:
    """

    if clf is not None:
        # ------------------------
        # Plot Decision Boundaries of two classifiers
        # ------------------------
        # Grid Resolution
        grid_res = calc_grid_resolution(x_train)
        if linecolor == "rosa":
            cmap = plt.cm.coolwarm
        else:
            cmap = plt.cm.PuOr
        if linecolor2 == "blue":
            cmap2 = plt.cm.Set3
        else:
            cmap2 = plt.cm.RdYlGn

        disp = DecisionBoundaryDisplay.from_estimator(clf,
                                                      x_train,
                                                      grid_resolution=grid_res,
                                                      response_method="predict",
                                                      xlabel=x_train.columns[0],
                                                      ylabel=x_train.columns[1],
                                                      alpha=1,
                                                      plot_method="contour",  # ,{‘contourf’, ‘contour’, ‘pcolormesh’}
                                                      cmap=cmap)

        if clf2 is not None:
            ##
            # contourf - area
            disp = DecisionBoundaryDisplay.from_estimator(clf2,
                                                          x_train,
                                                          grid_resolution=grid_res*2,
                                                          response_method="predict",
                                                          xlabel=x_train.columns[0],
                                                          ylabel=x_train.columns[1],
                                                          alpha=0.3,
                                                          plot_method="contourf",  # ,{‘contourf’, ‘contour’, ‘pcolormesh’}
                                                          cmap=cmap2, ax=disp.ax_)
            # contour - lines
            disp = DecisionBoundaryDisplay.from_estimator(clf2,
                                                          x_train,
                                                          grid_resolution=grid_res*2,
                                                          response_method="predict",
                                                          xlabel=x_train.columns[0],
                                                          ylabel=x_train.columns[1],
                                                          alpha=0.3,
                                                          plot_method="contour",
                                                          # ,{‘contourf’, ‘contour’, ‘pcolormesh’}
                                                          cmap=plt.cm.Spectral, ax=disp.ax_,
                                                          linewidths=3.0, levels=1)
            ##

        fig = disp.figure_
        ax = disp.ax_
    else:
        fig, ax = plt.subplots()


    # ------------------
    # Plot Datapoints
    # ------------------
    # Define color mapping
    if which_step=="step1" or which_step=="step3":
        color_map = {0: 'darkblue', 1: 'gold'}#, 2:"red"}
    elif which_step=="step2":
        color_map = {-1: 'mediumvioletred',
                     0: 'papayawhip',
                     1: 'yellowgreen'}
    elif which_step=="error_ampl":
        color_map = {#-2: 'darkmagenta',
                     -1: 'mediumvioletred',
                     0: 'papayawhip',
                     1: 'yellowgreen'}#,
                     #2: 'olivedrab'}
    else:
        raise ValueError("colormap is not initialized properly,"
                         "because we need 'which_step' to be"
                         " either step1, step2, step3 in "
                         " the input of plot_simdata() ")

    # Assign colors based on y_label values
    colors = [color_map[y] for y in colorlabels]

    # ----------------------------------------
    # Scatter-Plot
    # ----------------------------------------
    if len(edgecolor) > 0:
        ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=colors,
                   label=colors,
                   edgecolor=edgecolor, lw=2)  # , label=labels)
    else:
        ax.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=colors,
                   label=colors)

    # Add datapoint of interest
    if len(company_datapoint_to_check) > 0:
        ax.scatter(company_datapoint_to_check[0],
                   company_datapoint_to_check[1],
                   marker='x', color='crimson', s=100, linewidth=3)

    # Add Circle for Fidelity Neighborhood
    if circle_radius is not None:
        # Create a circle using patches.Circle
        circle = patches.Circle((company_datapoint_to_check[0],
                                 company_datapoint_to_check[1]),
                                circle_radius, fill=False,
                                color='black', linestyle='dotted')
        # Add the circle to the Axes object
        ax.add_patch(circle)

    # Create legend entries
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=8)
                     for color in color_map.values()]
    legend_keys = color_map.keys()

    # Add legend
    ax.legend(legend_labels, legend_keys, loc="upper right")

    ax.set_xlim([-0.2,np.max(x_train)+0.2])
    ax.set_ylim([-0.2,np.max(x_train)+0.2])
    ax.set_xlabel(x_train.columns[0], fontsize=fs)
    ax.set_ylabel(x_train.columns[1], fontsize=fs)
    ax.set_xticks(np.arange(0, np.max(x_train)+0.5, 1))
    ax.set_yticks(np.arange(0, np.max(x_train)+0.5, 1))

    ax.set_title(title)

    # Rename title if there are more than two "-" in the title
    if len(title.split('-')) > 2:
        title_l = title.split('-')[0:2]
        title = ''.join(title_l).replace(' ', '_')

    fig.savefig(save_to_folder + "_plot_" + title + ".png")

    return


# HDTree
def generate_dt_plot(hdtree_linear, save_to_folder, name_to_save):
    """
    generate Decision tree Plot for HDTree. Modify some strings
    for proper saving and later visualizing with http://webgraphviz.com/
    :param hdtree_linear: result from: HDTreeClassifier()
    :param save_to_folder: str
    :param name_to_save: str
    :return:
    """
    dot_data = hdtree_linear.generate_dot_graph()
    dot_data_s = dot_data.source

    # some string modifications
    new_string = dot_data_s.replace("✓", "x")
    cleaned_string = new_string.replace('≥', '>=')

    # remove newlines
    cleaned_string = cleaned_string.replace('\n', '        ')
    cleaned_string = cleaned_string.replace('// HDTree Export', '')
    cleaned_string = cleaned_string.replace('shape=box', 'shape=box,width=2.5,height=0.5')

    with open(save_to_folder + name_to_save + '_graph.dot', 'w') as dot_file:
        dot_file.write(cleaned_string)
    return


def assign_data_traverse_tree(node, counter, df_y, which_step="1"):
    # For HDTree
    # Recursively, go through the nodes, and check
    # if the node is an end leaf, if we are in the
    # end node, then we can get all datapoints
    # that are within this end leaf.
    if node._split_rule is None:
        # Process the attributes of the end leaf
        assign_data_to_leaf(node, counter[0], df_y, which_step)
        counter[0] += 1
    else:
        # Traverse the child nodes recursively
        for child_node in node._split_rule._child_nodes:
            # print(child_node)
            assign_data_traverse_tree(child_node, counter, df_y, which_step)


def assign_data_to_leaf(leaf, leaf_number, df_y, which_step="1"):
    # For HDTree
    # Assign to each datapoint in an end leaf, the number of the
    # belonging end leaf (starting from 0, 1 , 2 ,... just counting all
    # end leaves from left to right)
    # print(f"Leaf number: {leaf_number}")
    # print(leaf._assigned_data_indices)
    # print(len(leaf._assigned_data_indices))
    df_y.loc[leaf._assigned_data_indices, "leaf_id_DT" + which_step] = int(leaf_number)



def is_within_circle(x1, y1, x2, y2, radius):
    """
    used in HDTree
    Function to check if a point is within the circular neighborhood
    Check if data point (x2,y2) is within a circle with center (x1,y1)
    and radius
    Mittelpunkt Kreis: x1,y2
    :param x1: center - x
    :param y1: center - y
    :param x2: datapoint - x
    :param y2: datapoint -y
    :param radius: radius of circle
    :return: Boolean :  True: datapoint is within circle,
                        False: otherwise
    """
    is_in_circle = (x1 - x2) ** 2 + (y1 - y2) ** 2 <= radius ** 2
    return is_in_circle


def does_need_relabeling(df_y, i_d_centers, i_data):
    """
    Check, if the neighboring datapoint has the same label as the center point.
    (Then, it does not need to be relabeled. If it's not the same -> relabeling!)
    :param df_y:
    :param i_d_centers: index of center datapoint
    :param i_data: index of neighboring datapoint to check.
    :return: needs_relabeling: Boolean
    """
    # Is neighboring datapoint in the same labeling group as center point?
    ylab_center = df_y.loc[i_d_centers, "y_train"]
    ylab_neighb = df_y.loc[i_data, "y_train"]
    if ylab_center == ylab_neighb:
        needs_relabeling = False
    else:
        needs_relabeling = True
    return needs_relabeling



def check_pos_eps_boundary(df_y, x_train, pd_splitting_vals):
    """
    Used in HDtree
    Check on which side of the splitting boundaries the data is located.
    Add new column to df_y ["Level0-feat2<2.02",...]
    :param df_y: pd.Dataframe
    :param x_train: pd.Dataframe with the features in columns
    :param pd_splitting_vals: pd.Dataframe with columns: ["Level", "Feature", "feat", "value"]
            - generated with: get_splitting_values(hdtree_linear)
            "Level": Tree-Level (Level 0 -  head node, Level 1 - first level in the tree ,...)
            "Feature": Full feature of the node: e.g. "feat2 < 2.02"
            "feat": only feature name: "feat2"
            "value": only the value: 2.02
    :return:
        df_y: pd.Dataframe
        new_cols_df_y: list of new columns with Level-Splitting_Features
    """
    # Add New Columns of the splitting features ["Level0-feat2<2.02",...]
    new_cols_df_y = []
    for i_r in range(len(pd_splitting_vals)):
        new_cols_df_y.append(pd_splitting_vals.loc[i_r,"Level"] + ' - ' +
                             pd_splitting_vals.loc[i_r,"Feature"])

    # add new columns to df_y
    for i_r in range(len(pd_splitting_vals)):
        df_y[new_cols_df_y[i_r]] = None

    # First check on which side the data lies (smaller than boundary,
    # or larger than boundary?) -> Information into new columns of e.g. df_y["Level0-feat2>2.02"]
    # Go through all splitting boundaries
    for i_row in range(len(pd_splitting_vals)):
        # Get column-id of x_train for feature of interest (pd_splitting_vals) (splitting boundaries)
        which_col = x_train.columns.get_loc(pd_splitting_vals.loc[i_row, "feat"]).start
        feat_split_info = {"i_row_feat": i_row,
                           "which_col_xtrain": which_col}

        which_c = feat_split_info["which_col_xtrain"]   # column number in xtrain
        i_r_f = feat_split_info["i_row_feat"]           # row number in pd_splitting_vals

        for i_data in x_train.index:
            i_d_s = x_train.index.get_loc(i_data) # since we are using iloc here!
            if x_train.iloc[i_d_s, which_c] < pd_splitting_vals.loc[i_r_f, "value"]:
                df_y.loc[i_data, new_cols_df_y[i_r_f]] = 1
                # CHANGED HERE >=
            elif x_train.iloc[i_d_s, which_c] > pd_splitting_vals.loc[i_r_f, "value"]:
                df_y.loc[i_data, new_cols_df_y[i_r_f]] = 0
            # if data point is on the boundary, then set it to original y_train value
            #elif x_train.iloc[i_d_s, which_c] == pd_splitting_vals.loc[i_r_f, "value"]:
            #    df_y.loc[i_data, new_cols_df_y[i_r_f]] = df_y.loc[i_data, "y_train"]

    return df_y, new_cols_df_y


def is_above_below_boundary(x_train, i_data, i_d_sub,
                            pd_splitting_vals, df_y,
                            new_cols_df_y):
    """
    used in HDtree
    Go through the feature boundaries of pd_splitting and
    Check if the datapoint in x_train, with index i_data, is on the feature area side,
    as given in the columns of df_y["Level0-feat2<2.02"]
    :return: is_in_area: Boolean (True -> it is in the area, False -> datapoint is not to be considered)
    """
    #
    feat_split_info = pd.DataFrame(columns=["i_row_feat", "which_col_xtrain"],
                                   index=range(len(pd_splitting_vals)))
    #

    data_center = x_train.iloc[i_data]
    data_neighbrhd = x_train.iloc[i_d_sub]

    # which decision boundary is the closest one to the datapoint ?

    for i_row in range(len(pd_splitting_vals)):
        # Get column-id of x_train for feature of interest (pd_splitting_vals) (splitting boundaries)
        which_col = x_train.columns.get_loc(pd_splitting_vals.loc[i_row, "feat"]).start
        feat_split_info.loc[i_row, "i_row_feat"] = i_row
        feat_split_info.loc[i_row, "which_col_xtrain"] = which_col
    # print(feat_split_info)
        ##
    # which_c = feat_split_info["which_col_xtrain"]   # column number in xtrain
    # i_r_f = feat_split_info["i_row_feat"]
    # TODO: For now: only works for 2 features!
    # Check if entrys in df_y[Level-Feature] is the same for 2 datapoints (rows) ->
    # Then they are in the same area.
    if df_y.loc[i_data, new_cols_df_y[feat_split_info.loc[0, "i_row_feat"]]] == \
            df_y.loc[i_d_sub, new_cols_df_y[feat_split_info.loc[0, "i_row_feat"]]] and \
        df_y.loc[i_data, new_cols_df_y[feat_split_info.loc[1, "i_row_feat"]]] == \
            df_y.loc[i_d_sub, new_cols_df_y[feat_split_info.loc[1, "i_row_feat"]]]:
        is_in_area = True
    else:
        is_in_area = False
    # elif df_y.loc[i_data, new_cols_df_y[i_r_f]] == 1:
    #     if x_train.iloc[i_data, which_c] < pd_splitting_vals.loc[i_r_f, "value"]:
    #         is_in_area = True
    #     else:
    #         is_in_area = False
    # else:
    #     is_in_area = None
    return is_in_area


def rectangle_of_interest(company_datapoint_to_check,
                          pd_splitting_vals,
                          x_train):
    """
    Checks if the Datapoint_of_interest (company_datapoint_to_check) is laying in
    which rectangle of the decision boundaries.
    Returns x & y coordinates (rec_oi_x, rec_oi_y), within which rectangle the
    datapoint is located.
    :param company_datapoint_to_check:
    :param pd_splitting_vals:
    :param x_train:
    :return:
    """
    # TODO: change y_min / x_min, if we have multiple levels , this only works if
    #  we have 2 decision boundaries
    y_min = 0
    y_max = np.ceil(np.max(x_train.loc[:,"feat2"])) # maximal y-value
    rec_oi_y = [y_min, y_max]
    x_min = 0
    x_max = np.ceil(np.max(x_train.loc[:,"feat1"])) # maximal x-value
    rec_oi_x = [x_min, x_max]

    for i_levels in range(0,len(pd_splitting_vals)):
        if pd_splitting_vals.iloc[i_levels, 2] == "feat1":
            if company_datapoint_to_check[0] > pd_splitting_vals.iloc[i_levels, 3]:
                rec_oi_x = [pd_splitting_vals.iloc[i_levels, 3], x_max]
            elif company_datapoint_to_check[0] < pd_splitting_vals.iloc[i_levels, 3]:
                rec_oi_x = [x_min, pd_splitting_vals.iloc[i_levels, 3]]
            elif company_datapoint_to_check[0] == pd_splitting_vals.iloc[i_levels, 3]:
                raise ValueError("What do we do, if the company-data-point is located "
                                 "on the boundary??? Which rectangle to define then? TO CHECK!!")
        if pd_splitting_vals.iloc[i_levels, 2] == "feat2":
            if company_datapoint_to_check[1] > pd_splitting_vals.iloc[i_levels, 3]:# y_coord = feat2
                rec_oi_y = [pd_splitting_vals.iloc[i_levels, 3], y_max]
            elif company_datapoint_to_check[1] < pd_splitting_vals.iloc[i_levels, 3]:
                rec_oi_y = [y_min, pd_splitting_vals.iloc[i_levels, 3]]
            elif company_datapoint_to_check[1] == pd_splitting_vals.iloc[i_levels, 3]:
                raise ValueError("What do we do, if the company-data-point is located "
                                 "on the boundary??? Which rectangle to define then? TO CHECK!!")
    if len(pd_splitting_vals) > 2:
        print("rectangle_of_interest_x:")
        print(rec_oi_x)
        print("rectangle_of_interest_y:")
        print(rec_oi_y)
        warnings.warn("TO CHECK!! You need to check how the rectangle of interest is calculated! "
                      "It only works properly, if we have 2 decision boundaries."
                      "You would need to implement that x_min or y_min are overwritten if"
                      "the decision boundaries are overlaying each other. \n"
                      "!!!")
    return rec_oi_x, rec_oi_y

##

##

#dec_bound_x = rec_regions_x_uni.copy()
#dec_bound_y = rec_regions_y_uni.copy()

##
def rectangle_of_interest2(company_datapoint_to_check,
                          dec_bound_x, dec_bound_y,
                          x_train):
    """
    Checks in which rectangle of the decision boundaries
    the Datapoint_of_interest (company_datapoint_to_check, e.g. [4, 1]) is laying.
    Returns x & y coordinates (rec_oi_x, rec_oi_y), within which rectangle the
    datapoint is located.
    :param company_datapoint_to_check: [xval, yval]
    :param dec_bound_x: nd.array([0, 1.54, 7]) - decision boundaries on x-axis
    :param dec_bound_y: nd.array([0, 2.75, 7]) - decision boundaries on y-axis
    :param x_train:
    :return: rec_oi_x = [1.54, 7], rec_oi_y = [0, 2.75]
    """
    # initialize rec_oi_x, rec_oi_y
    y_min = dec_bound_y.min()
    y_max = dec_bound_y.max() # maximal y-value
    rec_oi_y = [y_min, y_max]
    x_min = dec_bound_x.min()
    x_max = dec_bound_x.max() # maximal x-value
    rec_oi_x = [x_min, x_max]

    # x-axis
    for i_dec_bound in range(1, len(dec_bound_x)-1):

        if company_datapoint_to_check[0] > dec_bound_x[i_dec_bound]:
            rec_oi_x[0] = dec_bound_x[i_dec_bound]
        elif company_datapoint_to_check[0] < dec_bound_x[i_dec_bound]:
            rec_oi_x[1] = dec_bound_x[i_dec_bound]
        elif company_datapoint_to_check[0] == dec_bound_x[i_dec_bound]:
            raise ValueError("X: What do we do, if the company-data-point is located "
                             "on the boundary??? Which rectangle to define then? TO CHECK!!")

    # y-axis
    for i_dec_bound in range(1, len(dec_bound_y)-1):
        if company_datapoint_to_check[1] > dec_bound_y[i_dec_bound]:
            rec_oi_y[0] = dec_bound_y[i_dec_bound]
        elif company_datapoint_to_check[1] < dec_bound_y[i_dec_bound]:
            rec_oi_y[1] = dec_bound_y[i_dec_bound]
        elif company_datapoint_to_check[1] == dec_bound_y[i_dec_bound]:
            raise ValueError("Y: What do we do, if the company-data-point is located "
                             "on the boundary??? Which rectangle to define then? TO CHECK!!")

    print("rectangle_of_interest_x:")
    print(rec_oi_x)
    print("rectangle_of_interest_y:")
    print(rec_oi_y)

    return rec_oi_x, rec_oi_y

##


def min_rectangle_around_xy_2dec_bounds(datapoint_to_check,
                                        dec_bound_x1, dec_bound_y1,
                                        dec_bound_x2, dec_bound_y2):
    """
    Given: 2 sets of decision boundaries, x-dec_bound_x1 = [0, 1.54] and y-dec_bound_y1 = [2,5];
     x-dec_bound_x2 = [1.33, 7] and y-dec_bound_y2 = [0,3].
    Prerequisite: datapoint_to_check lies within these given boundaries.
    Find the minimal rectangle that surrounds datapoint_to_check. What are the closest boundaries?

    Returns x & y coordinates (min_rec_around_x_oi, min_rec_around_y_oi), with the minimal rectangle around the
    datapoint.
    :param datapoint_to_check: [xval, yval]
    :param dec_bound_x1: list [0, 1.54] - decision boundaries on x-axis of DT1 (prerequisite: company_datapoint_to_check lies in between these boundaries!)
    :param dec_bound_x2: list [1.33, 7] - decision boundaries on x-axis of DT2
    :param dec_bound_y1: list [2, 5] - decision boundaries on y-axis of DT1 (prerequisite: company_datapoint_to_check lies in between these boundaries!)
    :param dec_bound_y2: list [0, 3] - decision boundaries on y-axis of DT2
    :return:
    min_rec_around_x_oi = [1.33, 1.54],
    min_rec_around_y_oi = [2, 3] ("Minimal rectangle around x-of interest")
    """
    # Minimal rectangle around datapoint of interest:
    min_rec_around_x_oi = [max(min(dec_bound_x1), min(dec_bound_x2)),
                           min(max(dec_bound_x1), max(dec_bound_x2))]
    min_rec_around_y_oi = [max(min(dec_bound_y1), min(dec_bound_y2)),
                           min(max(dec_bound_y1), max(dec_bound_y2))]

    # check if the company_datapoint_of_interest is located within rec_oi_x, rec_oi_y:
    if datapoint_to_check[0] < min_rec_around_x_oi[0] or \
        datapoint_to_check[0] > min_rec_around_x_oi[1]:
        raise ValueError("TO CHECK!! company_datapoint_to_check does not lie in between rec_oi_x: \n "
                         "company_datapoint_to_check-X = " +
                         str(datapoint_to_check[0]) + ",\n rec_oi_x = " + str(min_rec_around_x_oi))
    if datapoint_to_check[1] < min_rec_around_y_oi[0] or \
            datapoint_to_check[1] > min_rec_around_y_oi[1]:
        raise ValueError("TO CHECK!! company_datapoint_to_check does not lie in between rec_oi_y: \n "
                         "company_datapoint_to_check-Y = " +
                         str(datapoint_to_check[1]) + ",\n rec_oi_y = " + str(min_rec_around_y_oi))

    print("datapoint_of_interest:")
    print(datapoint_to_check)
    print("minimal rectangle around datapoint_oi-X:")
    print(str(min_rec_around_x_oi) + "  --> datapoint_oi-x-value = " + str(datapoint_to_check[0]))
    print("minimal rectangle around datapoint_oi-Y:")
    print(str(min_rec_around_y_oi) + "  --> datapoint_oi-y-value = " + str(datapoint_to_check[1]))

    return min_rec_around_x_oi, min_rec_around_y_oi


##
def is_within_range(range_l, x_train_val):
    """
    Checks if x_train_val is within the given range in range_l.
    :param range_l: list [xmin, xmax]
    :param x_train_val: float, one value to check
    :return:
    """
    if range_l[0] <= x_train_val <= range_l[1]:
        is_in_range = True
    else:
        is_in_range = False
    return is_in_range


def read_dec_tree_params(json_dec_tree_params):
    with open(json_dec_tree_params, 'r') as f:
        dec_tree_parameters = json.load(f)
    return dec_tree_parameters

def routine_get_person_id_to_datapoint_oi(json_dir,
                                          json_dec_tree,
                                          datapoint_of_interest=[]):
    """
    Read Data. Read decision tree parameters. Do Step1 (Prediction with DT1)
    Plot Data with datapoint of interest.
    Goal: Get person_index for the given datapoint.

    :return: df_y_sub: pd.Dataframe: Subdataframe with datapoint of interest.
                        And Index-Label.
    """

    datapoint_of_interest_test = []
    testdata = False

    # Read Simulation Data.
    if "simulated_data" in json_dir:
        agg_data, data_specs = read_simulated_data(json_dir)

    # Read Foundercheck Data.
    else:
        agg_data, data_specs = read_foundercheck_data(json_dir)

    # ------------------------
    # Decision Tree Parameters
    # ------------------------
    # Read decision tree parameters

    with open(json_dec_tree, 'r') as f:
        dec_tree_parameters = json.load(f)

    # ***********************
    # Define variables
    alpha = 0.5  # for plotting
    dir_to = data_specs["dir_to"]
    dir_data = data_specs["dir_data"]
    save_to_folder = data_specs["save_to_folder"]
    name_to_save = data_specs["name_to_save"]
    which_feature_set = data_specs["which_feature_set"]
    ylabel = data_specs["ylabel"]
    person_id_label = data_specs["person_id_label"]

    K_featnames = data_specs["K_featnames"]
    K_featnames_wCompanyID = data_specs["K_featnames_wCompanyID"]

    dec_tree_parameters["person_id_label"] = person_id_label

    # Transform y to a scala: [0,1], instead [2,3]
    # Check if unique values are [2, 3]
    agg_data = transform_y_to_0_1(agg_data, ylabel)

    # Split dataset into train and test set
    x_y_dict = split_train_test(agg_data, dec_tree_parameters,
                                K_featnames, K_featnames_wCompanyID,
                                ylabel,
                                person_id_label)

    x_train = x_y_dict["x_train"]
    y_train = x_y_dict["y_train"].astype(int)
    y_train_wCompanyID = x_y_dict["y_train_wCompanyID"]
    x_test = x_y_dict["x_test"]
    y_test = x_y_dict["y_test"].astype(int)
    y_test_wCompanyID = x_y_dict["y_test_wCompanyID"]

    # create df_y: Summary of all predicions of each Step.
    df_y = create_df_y(y_train_wCompanyID, ylabel, x_train, person_id_label)
    df_y_test = create_df_y(y_test_wCompanyID, ylabel, x_test, person_id_label)

    # STEP 1
    # HDTree
    hdtree_linear, df_y, df_y_test = \
        step1_hdtree(x_train, y_train, x_test, y_test,
                     dec_tree_parameters, df_y, df_y_test,
                     which_step="1")

    # Plot Data with Decision Boundary STEP 1
    rec_regions_x, rec_regions_y = \
        plot_decision_boundary(hdtree_linear, x_train, y_train,
                               save_to_folder,
                               datapoint_of_interest,
                               plot_color_area=True,
                               which_step="step1",
                               fs=12,
                               title="Datapoint: " + str(datapoint_of_interest),
                               cmap='Set3',
                               alpha=alpha)
    #
    if testdata:
        rec_regions_x, rec_regions_y = \
            plot_decision_boundary(hdtree_linear, x_test, y_test,
                                   save_to_folder,
                                   datapoint_of_interest_test,
                                   plot_color_area=True,
                                   which_step="step1",
                                   fs=12,
                                   title="Datapoint: " + str(datapoint_of_interest_test),
                                   cmap='Set3',
                                   alpha=alpha)

    #
    # datapoint_of_interest = [2.51, 2.25] # --> Company_ID = 433

    # # Get Person-ID / Company-ID from given datapoint_of_interest.
    df_y_sub = get_person_id_with_given_datapoint(df_y,
                                                  which_feature_set,
                                                  datapoint_of_interest)
    print(df_y_sub[["y_train", "Index", which_feature_set[0], which_feature_set[1]]])
    # df_y_test_sub = get_person_id_with_given_datapoint(df_y_test,
    #                                        which_feature_set,
    #                                        datapoint_of_interest_test)

    return df_y_sub