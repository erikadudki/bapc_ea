import lime
from lime import lime_tabular
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#
# For converting textual categories to integer labels
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# for creating train test split
from sklearn.model_selection import train_test_split

from base_model.Algorithms_Dez2023.error_amplification_hdtree import *

dir_main = "C:/Users/dudkin/Documents/Projects/inAlco_ExplainableAI/" \
           "bapcs_code/Speedinvest2/"

# In order to find the person_id_to_check:
# run the script: get_person_id.py.
person_id_to_check = 244  #286#94  # 4, 94
kernel_width = 2

# Read decision_tree_parameters
json_dec_tree_params = dir_main + "base_model/Algorithms_Dez2023/read_dec_tree_parameters.json"

simdata = True
show_figure = True
plot_lime_dec_bound = False

if simdata:
    # Read Simulation Data.
    json_data_dir = dir_main + 'base_model/Algorithms_Dez2023/read_simulated_data.json'
else:
    # Read Foundercheck Data.
    json_data_dir = dir_main + 'base_model/Algorithms_Dez2023/read_Foundercheck_data.json'




cost_function_for_lime = "Gini"

# Read data specifications from json-file.
data_specs = read_data_specs(json_data_dir)

#list_features = [["Activity Level", "Willingness to Compromise"]]
#list_features = [["Activity Level", "Intellectual Curiosity"]]
# for i_list_feat in range(0, len(list_features)):
i_list_feat = 0
if simdata:
    list_features = [["feat1", "feat2"]]
    data_specs["which_feature_set"] = list_features[i_list_feat]
else:
    list_features = data_specs["which_feature_set"]

# Saving Directory save_to_folder.../lime
data_specs = create_saving_dir(data_specs,
                               person_id_to_check)
save_to_folder = data_specs["save_to_folder"]
data_specs["save_to_folder"] = os.path.join(save_to_folder, "lime")

if not os.path.exists(data_specs["save_to_folder"]):
    # If it doesn't exist, create it
    os.makedirs(data_specs["save_to_folder"])
    print(f"Directory created: '{data_specs['save_to_folder']}' ")
#
if simdata:
    agg_data, data_specs = read_simulated_data(data_specs)
else:
    agg_data, data_specs = read_foundercheck_data(data_specs)

dec_tree_parameters = read_dec_tree_params(json_dec_tree_params)
dec_tree_parameters["which_cost_fct"] = cost_function_for_lime
# ------------------------
# Decision Tree Parameters
# ------------------------
circle_radius_factor = dec_tree_parameters["circle_radius_factor"]
which_cost_fct = dec_tree_parameters["which_cost_fct"] # "EntropyMeasure", "RelativeAccuracyMeasure"
max_depth = dec_tree_parameters["max_depth"]
min_samples_leaf = dec_tree_parameters["min_samples_leaf"]
min_samples_split = dec_tree_parameters["min_samples_split"]
randomstate = dec_tree_parameters["randomstate"]
trainingset = dec_tree_parameters["trainingset"]
alpha = 0.5 # for plotting the scatter data points
if not which_cost_fct in {"Gini", "RelativeAccuracyMeasure", "Entropy"}:
    raise ValueError("'" + which_cost_fct + "' as cost-function not available. "
                                            'Please define cost-function \n(in '
                                            'read_dec_tree_parameters.json) between: \n{"Gini", "RelativeAccuracyMeasure", "Entropy"}')

# Define variables
dir_to = data_specs["dir_to"]
dir_data = data_specs["dir_data"]
save_to_folder = data_specs["save_to_folder"]
name_to_save = data_specs["name_to_save"]
which_feature_set = data_specs["which_feature_set"]
ylabel = data_specs["ylabel"]
person_id_label = data_specs["person_id_label"]
dec_tree_parameters["person_id_label"] = person_id_label
features_str = "__".join(feature.replace(' ', '_') for feature in data_specs['which_feature_set'])
dec_tree_parameters["which_features"] = features_str

K_featnames = data_specs["K_featnames"]
K_featnames_wCompanyID = data_specs["K_featnames_wCompanyID"]

#
# Transform y to a scala: [0,1], instead [2,3]
# Check if unique values are [2, 3]
agg_data = transform_y_to_0_1(agg_data, ylabel)

# Get datapoint_of_interest, with given companyID
datapoint_of_interest = get_datapoint_oi_given_person_id(person_id_to_check,
                                                         agg_data,
                                                         person_id_label,
                                                         which_feature_set)
print("DATAPOINT OF INTEREST (TRAIN): " + str(datapoint_of_interest))
#
# ############### SAVING ##########################
# Specification for Saving -> Definition of Filenames
dec_tree_parameters = name_nametosave(dec_tree_parameters, circle_radius_factor)
dec_tree_parameters["save_to_folder"] = save_to_folder
#
# save original data matrix (agg_data)
agg_data.to_excel(os.path.join(save_to_folder, "original_data_" +
                               name_to_save + dec_tree_parameters["nametosave_short"] + ".xlsx"))
# save dec-tree-parameters
# Convert the dictionary to a DataFrame
df_tree_params = pd.DataFrame.from_dict(dec_tree_parameters, orient='index', columns=['Value'])
df_tree_params.loc["person_id_to_check"] = person_id_to_check
df_tree_params.loc["point_of_interest"] = str(datapoint_of_interest)
#
# Save to Excel
df_tree_params.to_excel(os.path.join(save_to_folder, "dec_tree_params_" +
                                     name_to_save + dec_tree_parameters["nametosave_short"] + ".xlsx"))
# ########################################

# Split dataset into train and test set
x_y_dict = split_train_test(agg_data, dec_tree_parameters,
                            K_featnames, K_featnames_wCompanyID,
                            ylabel,
                            person_id_label)
#
x_all = x_y_dict["x_all"]
y_all = x_y_dict["y_all"]
x_train = x_y_dict["x_train"]
y_train = x_y_dict["y_train"].astype(int)

y_train_wCompanyID = x_y_dict["y_train_wCompanyID"]
x_test = x_y_dict["x_test"]
y_test = x_y_dict["y_test"].astype(int)
y_test_wCompanyID = x_y_dict["y_test_wCompanyID"]
agg_data_orig_sample = x_y_dict["agg_data_orig_sample"]
x_train_wCompanyID = x_y_dict["x_train_wCompanyID"]
x_test_wCompanyID = x_y_dict["x_test_wCompanyID"]

# Create df_y: Summary of all predicions of each Step.
df_y = create_df_y(y_train_wCompanyID, ylabel, x_train, person_id_label)
df_y_test = create_df_y(y_test_wCompanyID, ylabel, x_test, person_id_label)
#
# Get Index in the dataframe, for the person of interest.
original_df_index_person_oi = df_y.loc[df_y[dec_tree_parameters["person_id_label"]] == person_id_to_check].index.values
if len(original_df_index_person_oi)==1:
    original_df_index_person_oi = original_df_index_person_oi[0]
else:
    raise ValueError("ACHTUNG! There are more than one entry "
                     "for df_y['Index'] == " + str(person_id_to_check))
#
#
# Plot simulation data without decision boundaries
clf_to_plot = None
#
# plot_simdata(clf_to_plot, x_train, y_train,
#              save_to_folder,
#              dec_tree_parameters,
#              datapoint_of_interest,
#              title="Trainingdata - Step1",
#              which_step="step1",
#              fs=14,
#              linecolor="rosa",
#              lw=3, alpha=alpha,
#              show_figure=show_figure)

# ####################################################
# ############### RANDOM FOREST CLASSIFIER ###########
# ####################################################
clf = RandomForestClassifier(random_state=42,
                             ccp_alpha=dec_tree_parameters["ccp_alpha"])
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)
print(classification_report(y_pred, y_train))

accuracy = np.round(accuracy_score(y_train, y_pred), 3)
#
y_pred_person_oi = y_pred[original_df_index_person_oi]
#
print('Random Forest Model accuracy score : {0:0.4f}'. format(accuracy))
#
# Plot Data with Decision Boundary STEP 1
rec_regions_x, rec_regions_y, ax, fig, \
    legend_labels, legend_keys = \
    plot_decision_boundary(clf, x_train, y_train,
                           save_to_folder,
                           dec_tree_parameters,
                           datapoint_of_interest,
                           plot_color_area=True,
                           which_step="step1",
                           fs=12,
                           title="LIME",
                           cmap='Set3',
                           alpha=alpha,
                           show_figure=show_figure)
#
# hdtree_linear = clf
#
# ###########################################
# ############# LIME ########################
# ###########################################

# Instantiating the explainer object by passing in the training set,
# and the extracted features
explainer_lime = lime_tabular.LimeTabularExplainer(x_train.values,
                                                   feature_names=K_featnames,
                                                   verbose=True,
                                                   training_labels=y_train,
                                                   mode='classification',
                                                   kernel_width=kernel_width,
                                                   random_state=42)



#
exp = explainer_lime.explain_instance(np.array(datapoint_of_interest),
                                      clf.predict_proba,
                                      #labels=(0,1),
                                      num_features=2)
#
pd_lime_summary = pd.DataFrame(columns=["DataIndex",
                                        "RF_Accuracy",
                                        "Intercerpt_Class0",
                                        "Intercerpt_Class1",
                                        "Prediction_local_Class0",
                                        "Prediction_local_Class1",
                                        "Right_Class0", "Right_Class1",
                                        "R2_Score"],
                               index=range(0, len(list_features)))
# Rename Indices with Features.
idx = K_featnames[0] + '_' + K_featnames[1]
pd_lime_summary.rename(index={i_list_feat:idx}, inplace=True)

pd_lime_summary.loc[idx, "DataIndex"] = person_id_to_check
pd_lime_summary.loc[idx, "RF_Accuracy"] = accuracy
#pd_lime_summary.loc[idx, "Intercerpt_Class0"] = exp.intercept[0]
pd_lime_summary.loc[idx, "Intercerpt_Class1"] = exp.intercept[1]
pd_lime_summary.loc[idx, "Prediction_local_Class0"] = 1 - exp.local_pred[0]# exp.local_pred[0] is the value for Class 1
pd_lime_summary.loc[idx, "Prediction_local_Class1"] = exp.local_pred[0] # Is giving the value for Class 1
pd_lime_summary.loc[idx, "Right_Class0"] = exp.predict_proba[0]
pd_lime_summary.loc[idx, "Right_Class1"] = exp.predict_proba[1]
pd_lime_summary.loc[idx, "R2_Score"] = exp.score



pd_lime_summary.to_csv(os.path.join(os.path.dirname(save_to_folder),
                                   "summary_LIME_Accuracy_Fidelity" +
                                   dec_tree_parameters["nametosave_short"] + '_personID_' +
                                   str(person_id_to_check) + "_nfeat" +
                                   str(len(list_features)) + ".csv"))

#
# Retrieve the coefficients of the local linear surrogate model
#lime_coefs = exp.local_exp[y_pred_person_oi]  # The linear model coefficients
#intercept = exp.intercept[y_pred_person_oi]   # The intercept
lime_coefs = exp.local_exp[1]
intercept = exp.intercept[1]

original_prediction = clf.predict([datapoint_of_interest])

#

# Extract the linear model coefficients for Feature 1 and Feature 2
coef_feature1 = lime_coefs[0][1]  # Coefficient for Feature 1
coef_feature2 = lime_coefs[1][1]  # Coefficient for Feature 2

X = x_train.values

# Generate points to plot the line
# Try shifting the intercept slightly
#intercept_adjustment = 4  # Modify this to shift the line horizontally
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(coef_feature1 / coef_feature2) * x_vals - (intercept / coef_feature2) #+ intercept_adjustment
y_vals = (0.5 - coef_feature1 * x_vals - intercept ) / coef_feature2 + intercept*10

# ############################
# PLOTTING LIME's DECISION BOUNDARY
# ############################
#ax.scatter(datapoint_of_interest[0], datapoint_of_interest[1], color='black',
#           marker='o', s=100, label='Datapoint of Interest')


#
# Plot LIME's local linear decision boundary
if plot_lime_dec_bound:
    legend_decbound = 'LIME Local Decision Boundary, kernel_width = ' + str(kernel_width)
    line3, = ax.plot(x_vals, y_vals, color='red',
                     label=legend_decbound)
    #ax.legend(loc='lower right')

    # Legend
    # Create a new legend entry for the new line
    new_handle = plt.Line2D([0], [0], linestyle='-', color='red')


    # Update the existing legend handles and labels
    legend_labels.append(new_handle)
    legend_keys = list(legend_keys)  # Convert to list to append
    legend_keys.append(legend_decbound)

    # Update the legend with the new entries
    ax.legend(legend_labels, legend_keys, loc="lower right")



ax.set_title("Explain Class " + str(y_pred_person_oi) +
             "\n Random Forest Accuracy = " + str(np.round(accuracy,2)) +
             "\n Intercerpt = " + str(np.round(pd_lime_summary.loc[idx, "Intercerpt_Class"+ str(y_pred_person_oi)],2)) +
             "\n Prediction_local = " + str(np.round(pd_lime_summary.loc[idx, "Prediction_local_Class"+ str(y_pred_person_oi)],2)) +
             "\n Random Forest Prediction Class = " + str(np.round(pd_lime_summary.loc[idx, "Right_Class"+ str(y_pred_person_oi)],2)) +
             "\n " + r"$R^2$ = " + str(np.round(pd_lime_summary.loc[idx, "R2_Score"],2)),
             loc='left')
fig.set_size_inches(6, 6)  # Width: 12 inches, Height: 6 inches
fig.tight_layout()
# Change the figure size after the plot is created



#
# plt.figure(figsize=[5,8])
# exp.as_pyplot_figure()
# plt.show()
# ##
# exp.show_in_notebook()

#
# PLOT FEATURE IMPORTANCE
# Get the explanation as a list of tuples
explanation = exp.as_list()

# Plot the explanation using Matplotlib
features, weights = zip(*explanation)
y_pos = np.arange(len(features))

plt.figure(figsize=(8, 2))
plt.barh(y_pos, weights, align='center')
plt.axvline(0, color="k")
plt.yticks(y_pos, features)
plt.xlabel('Feature Importance', fontsize=12)
plt.yticks(fontsize=12)
plt.title('LIME Feature Importance, Kernel Width = ' + str(kernel_width))
plt.xlim([-1,1])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
#plt.gca().invert_yaxis()
plt.show()


##
# ######## FIDELITY ##########
# ############################
# # Get the perturbed samples and the black-box model predictions
# # Generate perturbed samples manually
# num_samples = 100  # Number of perturbed samples to generate
# # Generate perturbed samples manually
# perturbed_samples = np.random.normal(loc=datapoint_of_interest,
#                                      scale=1, size=(num_samples, len(datapoint_of_interest)))
# ##
# # Replace values below 0 with 0 and above 7 with 7
# perturbed_samples = np.clip(perturbed_samples, np.min(x_train), np.max(x_train))
# #
#
# # Round the samples to the nearest integer
# if not simdata:
#     perturbed_samples = np.round(perturbed_samples).astype(int)
#
#
# black_box_predictions = clf.predict_proba(perturbed_samples)    # returns probability of class labels
# black_box_pred = clf.predict(perturbed_samples) # returns class labels
# y_pred_blackbox = clf.predict_proba([datapoint_of_interest])
# rounded_black_box_predictions = np.where(black_box_predictions < 0.5, 0, 1)
# #
# # Calculate local model's predictions
# intercept = exp.intercept[1]
# #
# explanation_map = dict(exp.local_exp[1])
# #
# coefficients = np.array([explanation_map.get(i, 0) for i in range(perturbed_samples.shape[1])])
# local_predictions = intercept + np.dot(perturbed_samples, coefficients)
# rounded_local_predictions = np.where(local_predictions < 0.5, 0, 1)
# #
# local_prediction_doi = intercept + np.dot([datapoint_of_interest], coefficients)
# #
# # Calculate fidelity using Mean Squared Error
# mse_fidelity = mean_squared_error(black_box_predictions[:, 1], local_predictions)
# mse_fidelity = mean_squared_error(black_box_pred, rounded_local_predictions)
#
# print("MSE_Fidelity = " + str(mse_fidelity))

##

# rec_regions_x, rec_regions_y = \
#     plot_decision_boundary(clf,
#                            pd.DataFrame(perturbed_samples),
#                            rounded_local_predictions,#black_box_pred,
#                            save_to_folder,
#                            dec_tree_parameters,
#                            datapoint_of_interest,
#                            plot_color_area=True,
#                            which_step="step1",
#                            fs=12,
#                            title="Step 1 - Trainingdata",
#                            cmap='Set3',
#                            alpha=alpha,
#                            show_figure=show_figure)
