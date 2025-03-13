from base_model.Algorithms_Dez2023.helper_functions import *


def run_error_amplification_routine(json_data_dir,
                                    json_dec_tree_params,
                                    person_id_to_check):
    """
    Routine for running all 3 steps of BAPC with error amplification.
    This function calls all necessary functions to run the analysis.
    :param json_data_dir: str - path to json-file with data-specifications
    :param json_dec_tree_params: str - path to json file with decision tree parameters
    :param person_id_to_check: int - index of the person of interest -
                    (run get_person_id.py, if it's not known)
    :return:
    """
    # ***************************************************************************
    # ******* Error Amplification - which epsilon value to amplify **************
    # ***************************************************************************
    testdata = False
    if testdata:
        person_id_to_check_test = 345

    agg_data, data_specs = read_foundercheck_data(json_data_dir)
    dec_tree_parameters = read_dec_tree_params(json_dec_tree_params)
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
#
    if not which_cost_fct in {"Gini", "RelativeAccuracyMeasure", "Entropy"}:
        raise ValueError( "'" + which_cost_fct + "' as cost-function not available. "
                         'Please define cost-function \n(in '
                         'read_dec_tree_parameters.json) between: \n{"Gini", "RelativeAccuracyMeasure", "Entropy"}')
#

    # Define variables
    dir_to = data_specs["dir_to"]
    dir_data = data_specs["dir_data"]
    save_to_folder = data_specs["save_to_folder"]
    name_to_save = data_specs["name_to_save"]
    which_feature_set = data_specs["which_feature_set"]
    ylabel = data_specs["ylabel"]
    person_id_label = data_specs["person_id_label"]
    dec_tree_parameters["person_id_label"] = person_id_label

    K_featnames = data_specs["K_featnames"]
    K_featnames_wCompanyID = data_specs["K_featnames_wCompanyID"]




    # Transform y to a scala: [0,1], instead [2,3]
    # Check if unique values are [2, 3]
    agg_data = transform_y_to_0_1(agg_data, ylabel)


    #
    # If for the same datapoint, multiple ambiguous ylabel exist:
    #     Rename the ylabels, if for in one category we have 2 or more datapoints,
    #     then in the other category.
    #     E.g. ylabel = 0 -> 5 datapoints
    #          ylabel = 1 -> 2 datapoints  -> change their ylabel to 0.
    # agg_data, counts_datapoints_ylabel, \
    #     counts_datapoints_ylabel2 = \
    #     change_ylabel_duplicated_datapoints_ambiguous_ylabel(agg_data,
    #                                                          which_feature_set,
    #                                                          ylabel)
    # Get datapoint_of_interest, with given companyID
    datapoint_of_interest = get_datapoint_oi_given_person_id(person_id_to_check,
                                                             agg_data,
                                                             person_id_label,
                                                             which_feature_set)
    print("DATAPOINT OF INTEREST (TRAIN): " + str(datapoint_of_interest))

    if testdata:
        datapoint_of_interest_test = get_datapoint_oi_given_person_id(person_id_to_check_test,
                                                                      agg_data,
                                                                      person_id_label,
                                                                      which_feature_set)
        print("DATAPOINT OF INTEREST (TEST): " + str(datapoint_of_interest_test))

    # Specification for Saving -> Definition of Filenames
    filenametosave = name_nametosave(dec_tree_parameters)
    save_to_folder = os.path.join(save_to_folder, "100Q_" + name_to_save + filenametosave + \
                 '__CIRCLE_RADIUS_' + str(float(circle_radius_factor)).replace(".","_") + '__' + \
                 'comp_oi_' + "_".join(map(str, datapoint_of_interest)) + "__")
    dec_tree_parameters["nametosave"] = filenametosave
    dec_tree_parameters["save_to_folder"] = save_to_folder


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


    # old step1-function (sklearn) to produce df_y
    # clf, df_y, df_y_test, _, _, _, _, _, _, _ = step1(dec_tree_parameters,
    #                                             x_train, y_train,
    #                                             x_test, y_test,
    #                                             y_train_wCompanyID=None,
    #                                             y_test_wCompanyID=None)
    #
    # Plot simulation data without decision boundaries
    clf_to_plot = None
    plot_simdata(clf_to_plot, x_train, y_train,
                 save_to_folder,
                 datapoint_of_interest,
                 title="Trainingdata - Step1",
                 which_step="step1",
                 fs=14,
                 linecolor="rosa",
                 lw=3, alpha=alpha)

    # ####################################################
    # ############### STEP 1 #############################
    # ####################################################
    if which_cost_fct == "Gini":
        # sklearn-Decision Tree
        hdtree_linear, df_y, df_y_test = step1(dec_tree_parameters,
                                               x_train, y_train,
                                               x_test, y_test,
                                               df_y, df_y_test,
                                               which_step="1")
    else:
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
                               title="Step 1 - Trainingdata",
                               cmap='Set3',
                               alpha=alpha)

    if testdata:
        rec_regions_x, rec_regions_y = \
            plot_decision_boundary(hdtree_linear, x_test, y_test,
                                   save_to_folder,
                                   datapoint_of_interest_test,
                                   plot_color_area=True,
                                   which_step="step1",
                                   fs=12,
                                   title="Step 1 - Testdata",
                                   cmap='Set3',
                                   alpha=alpha)


    # datapoint_of_interest = [2.51, 2.25] # --> Company_ID = 433
    # datapoint_of_interest = [3, 4]
    # # Get Person-ID / Company-ID from given datapoint_of_interest.
    # df_y_sub = get_person_id_with_given_datapoint(df_y,
    #                                        which_feature_set,
    #                                        datapoint_of_interest)
    # df_y_test_sub = get_person_id_with_given_datapoint(df_y_test,
    #                                        which_feature_set,
    #                                        datapoint_of_interest_test)
    #
    # Create df_accuracies-Summary
    df_accuracies = create_df_accuracies(y_train, y_test,
                                         df_y["y_predicted_hdtree1"],
                                         df_y_test["y_predicted_hdtree1"])
    #
    # Feature importances
    # Organizing feature names and importances in a DataFrame
    # if which_data == "simulated_data":
    #     features_df = pd.DataFrame({'features': K_featnames.get_level_values(0) +
    #                                             '(' + K_featnames.get_level_values(1) + ')',
    #                                 'importances1': clf.feature_importances_ })
    # elif which_data == 'original_data_sub':
    #     features_df = pd.DataFrame({'features': K_featnames,
    #                                 'importances1': clf.feature_importances_})
    # elif which_data == 'original_data':
    #     features_df = pd.DataFrame({'features': K_featnames,
    #                                 'importances1': clf.feature_importances_})
    #
    # dir_to = save_to_folder + "randomstate" + str(randomstate) + "_FI_step1" + ".png"
    #
    # features_df_sorted = plot_features(features_df, plot_feat="importances1",
    #                                    sort_by='importances1', save_to=dir_to,
    #                                    show=False, saving=False)
    #
    # Fix features of Step 1
    # x_train_s2, x_test_s2, df_new_feat_step1, add_name = \
    #     fix_features_of_step1(fix_step1, features_df_sorted,
    #                           x_train, x_test)
    #
    # ------------------
    # Generate Plot tree & visualize Decision Tree Structure with: http://webgraphviz.com/
    if which_cost_fct == "Gini":
        draw_graph(hdtree_linear, x_train, save_to_folder, addname=which_cost_fct, whichstep="1")
    # else:
    #     generate_dt_plot(hdtree_linear, save_to_folder, name_to_save=name_to_save)
    #

    # ####################################################
    # ############### STEP 2 #############################
    # ####################################################

    # Add column "epsilon" to df_y
    df_y["epsilon"] = df_y["y_train"] - df_y["y_predicted_hdtree1"]


    # --------------------------------------------
    # ------------- STEP 2 -----------------------
    # --------------------------------------------
    df_y, df_y_test, df_accuracies, \
        predicted_rf_tr, predicted_rf_test, \
        surr_predict, surr_predict_test, model = \
         step2(x_train, x_test, df_y["epsilon"],
               y_train, y_test,
               df_y["y_predicted_hdtree1"],
               df_y_test["y_predicted_hdtree1"],
               dec_tree_parameters,
               df_accuracies,
               df_y, df_y_test)

    #
    # -----------------------------------------------------------
    # ############## ERROR AMPLIFICATION #################
    # -----------------------------------------------------------
    # Get the corresponding epsilon value for the given person_id
    # (for the given datapoint of interest)
    check_for_eps_value = get_epsilon_value_for_person_id(
        df_y, person_id_label, person_id_to_check)

    df_y, circle_radius, \
        rec_oi_x, rec_oi_y = error_amplification(datapoint_of_interest,
                                                 x_train, hdtree_linear,
                                                 check_for_eps_value,
                                                 df_y,
                                                 circle_radius_factor,
                                                 circle_around_one_point=True)
    if testdata:
        df_y_test, circle_radius, \
            rec_oi_x, rec_oi_y = error_amplification(datapoint_of_interest_test,
                                                     x_test, hdtree_linear,
                                                     check_for_eps_value,
                                                     df_y_test,
                                                     circle_radius_factor,
                                                     circle_around_one_point=True)

    # plot
    # plot_decision_boundary(hdtree_linear, x_train, df_y.loc[:, "extended_eps"],
    #                        save_to_folder,
    #                        company_datapoint_to_check,
    #                        which_step="error_ampl",
    #                        plot_color_area=False,
    #                        fs=12,
    #                        title="extended_epsilon",
    #                        cmap='Set3',
    #                            alpha=alpha)
    #
    plot_decision_boundary(hdtree_linear, x_train, df_y.loc[:, "epsilon"],
                           save_to_folder,
                           datapoint_of_interest,
                           plot_color_area=False,
                           which_step="error_ampl",
                           fs=12,
                           title="Epsilon",
                           cmap='Set3',
                           alpha=alpha)
    plot_decision_boundary(hdtree_linear, x_train, df_y.loc[:, "epsilon_hat"],
                           save_to_folder,
                           datapoint_of_interest,
                           plot_color_area=False,
                           which_step="error_ampl",
                           fs=12,
                           title="Epsilon_hat",
                           cmap='Set3',
                           alpha=alpha)

    plot_decision_boundary(hdtree_linear, x_train, df_y.loc[:, "eps_EA"],
                           save_to_folder,
                           datapoint_of_interest,
                           plot_color_area=False,
                           which_step="error_ampl",
                           fs=12,
                           title="eps_EA",
                           cmap='Set3',
                           alpha=alpha)
    #
    # Testset
    if testdata:
        plot_decision_boundary(hdtree_linear, x_test, df_y_test.loc[:, "epsilon_hat"],
                               save_to_folder,
                               datapoint_of_interest_test,
                               plot_color_area=False,
                               which_step="error_ampl",
                               fs=12,
                               title="Epsilon_hat-test",
                               cmap='Set3',
                               alpha=alpha)

        plot_decision_boundary(hdtree_linear, x_test, df_y_test.loc[:, "eps_EA"],
                               save_to_folder,
                               datapoint_of_interest_test,
                               plot_color_area=False,
                               which_step="error_ampl",
                               fs=12,
                               title="eps_EA-test",
                               cmap='Set3',
                               alpha=alpha)
    #
    # plot_decision_boundary(hdtree_linear, x_train,
    #                        df_y.loc[:, "Y-eps"],
    #                        save_to_folder,
    #                        company_datapoint_to_check,
    #                        which_step="step1",
    #                        plot_color_area=False,
    #                        fs=12,
    #                        title="Y - epsilon_hat",
    #                        cmap='Set3',
    #                            alpha=alpha)
    #
    # plot_decision_boundary(hdtree_linear, x_train, df_y.loc[:, "epsilon"],
    #                        save_to_folder,
    #                        company_datapoint_to_check,
    #                        plot_color_area=False,
    #                        which_step="error_ampl",
    #                        fs=12,
    #                        title="Y - Y_tilde",
    #                        cmap='Set3',
    #                            alpha=alpha)
    #
    # plot_decision_boundary(hdtree_linear, x_train, df_y.loc[:, "eps_x"],
    #                        save_to_folder,
    #                        company_datapoint_to_check,
    #                        plot_color_area=False,
    #                        which_step="error_ampl",
    #                        fs=12,
    #                        title="Epsilon_hat",
    #                        cmap='Set3',
    #                            alpha=alpha)
    #
    # plot_decision_boundary(hdtree_linear, x_train, -df_y.loc[:, "eps_EA"],
    #                        save_to_folder,
    #                        company_datapoint_to_check,
    #                        plot_color_area=False,
    #                        which_step="error_ampl",
    #                        fs=12,
    #                        title="epsilon_EA",
    #                        cmap='Set3',
    #                            alpha=alpha)
    #

    #
    # Calculate y_epsilon_hat_ea -> Modified dataset with Error Amplification
    df_y["y_epsilon_hat_ea"] = df_y["y_train"] - df_y["eps_EA"]
    # Check that "y_epsilon_hat_ea" does not have values {-1} or {2}.
    df_y = correcting_wrong_prediction_min1_2(df_y)

    if testdata:
        df_y_test["y_epsilon_hat_ea"] = df_y_test["y_train"] - df_y_test["eps_EA"]
        df_y_test = correcting_wrong_prediction_min1_2(df_y_test)

    #
    # ........TESTING............
    # x_train_sorted = x_train.sort_values(by=("feat1","mean"))
    # # Target row values
    # target_values = [1.54, 3.34]
    # target_values = [5.21, 2.14]
    #
    # # Find the index of the row with target values
    # index = x_train.index[(x_train[('feat1','mean')] == target_values[0]) &
    #                    (x_train[('feat2','mean')] == target_values[1])].tolist()

    # .........For --Testing--- purposes:..............
    # i_data = 21
    # i_d_sub = 539
    # is_within_circle(x_train.iloc[i_d_sub, 0], x_train.iloc[i_d_sub, 1],
    #                             x_train.iloc[i_data, 0], x_train.iloc[i_data ,1],
    #                             radius=0.5)


    # ############################
    # local change of epsilon_hat
    local_epsilon_hat, df_y_big, \
        add_name_list = calc_local_epsilon_hat(df_y)
    #
    #
    # list_leafs = list(local_epsilon_hat.keys())
    # do analysis for all leaves ? or for a specific one ?
    # if leaf_to_search != "all":
    #     list_leafs = ["leaf" + str(leaf_to_search)]
    #for ii, i_leaves in enumerate(list_leafs): #range(range0, rangeend):

    # take a subset of df_y with the respective leaf
    fix_columns = [person_id_label, 'y_train',
                   'y_predicted_hdtree1',
                   'leaf_id_DT1',
                   'y_predicted_DT1+AI=y_pred_DT1+epsilon_hat',
                   'epsilon', 'epsilon_hat', #i_leaves,
                   #"epsilon_" + i_leaves,
                   "extended_eps", "eps_EA", "y_epsilon_hat_ea"]
    df_y = df_y_big[fix_columns]


    # --------------------------------------------
    # ------------- STEP 3 -----------------------
    # --------------------------------------------
    which_step = "3"

    if which_cost_fct == "Gini":
        hdtree_linear_s3, df_y, df_y_test = step1(dec_tree_parameters,
                                                  x_train, y_train,
                                                  x_test, y_test,
                                                  df_y, df_y_test,
                                                  which_step=which_step)
    else:
        hdtree_linear_s3, df_y, df_y_test = \
            step1_hdtree(x_train, np.array(df_y["y_epsilon_hat_ea"].values),
                         x_test, y_test,
                         dec_tree_parameters, df_y, df_y_test,
                         which_step=which_step)

    # DT1 - DT3
    df_y["DT1-DT3"] = df_y["y_predicted_hdtree1"] - \
                      df_y["y_predicted_hdtree3"]
    if testdata:
        df_y_test["DT1-DT3"] = df_y_test["y_predicted_hdtree1"] - \
                          df_y_test["y_predicted_hdtree3"]


    # Map values to colors
    df_y['edgecolor'] = df_y['eps_EA'].apply(map_epsilon_to_color)
    df_y.to_excel(save_to_folder + '_df_y.xlsx', index=True)

    if testdata:
        df_y_test['edgecolor'] = df_y_test['eps_EA'].apply(map_epsilon_to_color)
        df_y_test.to_excel(save_to_folder + '_df_y_test.xlsx', index=True)


    # generate plot tree
    if which_cost_fct == "Gini":
        draw_graph(hdtree_linear_s3, x_train, save_to_folder, addname=which_cost_fct, whichstep=which_step)
    # else:
    #     generate_dt_plot(hdtree_linear_s3, save_to_folder,
    #                      name_to_save=name_to_save + "_" + which_step)
    #
    #
    # Plot modified input data with one (original) decision boundary
    plot_decision_boundary(hdtree_linear, x_train, df_y.loc[:, "y_epsilon_hat_ea"],
                           save_to_folder,
                           datapoint_of_interest,
                           edgecolor=df_y['edgecolor'],
                           circle_radius=circle_radius,
                           plot_color_area=False,
                           which_step="step3",
                           fs=12,
                           title="Step3 - Modified Input Data",
                           cmap='Set3',
                           alpha=alpha)
    if testdata:
        plot_decision_boundary(hdtree_linear, x_test, df_y_test.loc[:, "y_epsilon_hat_ea"],
                               save_to_folder,
                               datapoint_of_interest_test,
                               edgecolor=df_y_test['edgecolor'],
                               circle_radius=None,
                               plot_color_area=False,
                               which_step="step3",
                               fs=12,
                               title="Step3 - Modified Input Data - Test",
                               cmap='Set3',
                               alpha=alpha)
    #
    # plot_decision_boundary(hdtree_linear,
    #                                   x_train,
    #                                   eps_hut_int,
    #                                   save_to_folder,
    #                                   company_datapoint_to_check,
    #                                   plot_color_area=False,
    #                                   which_step="step3",
    #                                   fs=12,
    #                                   title="Modified input data",
    #                                   cmap='Set3')
    #
    edgecolor = []
    plot_decision_boundary_two_dbound(hdtree_linear, hdtree_linear_s3,
                                      x_train,
                                      y_train,
                                      save_to_folder,
                                      edgecolor,
                                      datapoint_of_interest,
                                      circle_radius=None,
                                      plot_color_area=True,
                                      which_step="step3",
                                      fs=12,
                                      title="Step 3 - Original data",
                                      cmap='Set3')

    #
    # ---------------------------------------------------------
    # ------------ FIDELITY -----------------------------------
    # ---------------------------------------------------------
    # Fidelity within Rectangle of decision boundaries DT & DT3

    # Decision boundaries of DT3
    rec_regions_x_uni_dt3, rec_regions_y_uni_dt3,_,_,_ = get_dec_boundaries(
        x_train, hdtree_linear_s3)

    rec_oi_x_dt3, rec_oi_y_dt3 = rectangle_of_interest2(datapoint_of_interest,
                                                        rec_regions_x_uni_dt3,
                                                        rec_regions_y_uni_dt3,
                                                        x_train)
    #
    # Find the minimal rectangle around the datapoint_to_check:
    rec_around_x_oi, \
        rec_around_y_oi = min_rectangle_around_xy_2dec_bounds(datapoint_of_interest,
                                                              rec_oi_x, rec_oi_y,
                                                              rec_oi_x_dt3, rec_oi_y_dt3)


    # Vergebe ein Label an company_datapoint_to check (da das ein random Punkt
    # ist in einem Gebiet of interest)
    # if check_for_eps_value == 1:
    #     y_true_c_oi = 1
    # elif check_for_eps_value == -1:
    #     y_true_c_oi = 0

    # Get the corresponding y_train-label for the given
    # Person_id of interest.
    y_true_c_oi = get_y_train_label_for_person_id(person_id_to_check,
                                                  person_id_label,
                                                  df_y_big)


    #
    # ------------------------
    # Count all labels in minimal rectangle around datapoint_of_interest and
    # compare to true values
    # ------------------------
    # 1) get indices of x_train that are in this minimal rectangle around datapoint_oi:

    #
    # Use boolean indexing to filter the rows within the rectangle
    filtered_df = x_train[(x_train[K_featnames[0]] >= rec_around_x_oi[0]) &
                          (x_train[K_featnames[0]] <= rec_around_x_oi[1]) &
                          (x_train[K_featnames[1]] >= rec_around_y_oi[0]) &
                          (x_train[K_featnames[1]] <= rec_around_y_oi[1])]
    #
    # Get the indices of the data points within the rectangle and circle around datapoint
    indices_within_rectangle_circle = []
    for i_data_f in range(len(filtered_df)):
        if is_within_circle(datapoint_of_interest[0], datapoint_of_interest[1],
                            filtered_df.iloc[i_data_f, 0], filtered_df.iloc[i_data_f, 1],
                            radius=circle_radius):
            indices_within_rectangle_circle.append(filtered_df.index[i_data_f])


    #
    x_train_sub_rec = x_train.loc[indices_within_rectangle_circle,:]
    df_y_sub_rec = df_y.loc[indices_within_rectangle_circle,:]

    # for checking which datapoints are in fidelity region:
    df_y_big_sub_rec = df_y_big.loc[df_y_sub_rec.index,:]
    #
    # Modified input data
    # How many datapoints have the same label as the datapoint_of_interest ?
    #number_same_label = len(df_y_sub_rec[df_y_sub_rec["y_epsilon_hat_ea"] == y_true_c_oi]) # Modified input data
    number_same_label = len(df_y_sub_rec[df_y_sub_rec["y_train"] == y_true_c_oi])
    number_same_label = len(df_y_sub_rec[df_y_sub_rec["y_predicted_DT1+AI=y_pred_DT1+epsilon_hat"] == y_true_c_oi])

    # TODO
    fidelity_ratio = number_same_label / len(df_y_sub_rec)

    # If label of company_datapoint_to_check was not predicted correctly,
    # after Step 3 : Fidelity=0
    if y_true_c_oi != df_y.loc[indices_within_rectangle_circle[0], "y_predicted_hdtree3"]:
        fidelity_ratio = 0

    print("fidelity: " + str(np.round(fidelity_ratio, 3)))
    #
    plot_decision_boundary_two_dbound(hdtree_linear, hdtree_linear_s3,
                                      x_train,
                                      df_y["y_epsilon_hat_ea"],
                                      save_to_folder,
                                      df_y['edgecolor'],
                                      datapoint_of_interest,
                                      circle_radius,
                                      plot_color_area=True,
                                      which_step="step3",
                                      fs=12,
                                      title="Step 3 - New decision boundaries - \nfidelity_ratio = " +
                                            str(np.round(fidelity_ratio, 3)),
                                      cmap='Set3')

    #
    clfa = None
    plot_simdata(clfa, x_train, df_y["y_epsilon_hat_ea"],
                 save_to_folder,
                 datapoint_of_interest,
                 title="Modified input data",
                 which_step="step3",
                 fs=14,
                 linecolor="rosa",
                 lw=3, alpha=1)
    return

