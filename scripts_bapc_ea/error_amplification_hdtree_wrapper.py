from scripts_bapc_ea.helper_functions import *


def run_error_amplification_routine(json_data_dir,
                                    json_dec_tree_params,
                                    person_id_to_check,
                                    circle_radius_l=False,
                                    simdata = False,
                                    show_figure = True,
                                    all_features=False,
                                    plot_hyperbola=False):
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
    ##
    testdata = False
    if testdata:
        person_id_to_check_test = 345

    #
    # list_features = [["Activity Level", "Willingness to Compromise"]]

    # list_features = [["Activity Level", "Sociability"]]

    # Read data specifications from json-file.
    data_specs = read_data_specs(json_data_dir)
    # Read Decision Tree Parameters from JSON
    dec_tree_parameters = read_dec_tree_params(json_dec_tree_params)

    # Get list of features (either from data_specs JSON file, or all features)
    list_features = get_features(data_specs, all_features)

    alpha = 0.5  # for plotting the scatter data points

    ##
    for i_list_feat in range(0, len(list_features)):
        ##
        # Preprocess data, generate trainingset & testset (x_y_dict)
        data_specs, \
            x_y_dict, df_y, \
            df_y_test, \
            datapoint_of_interest,\
            circle_radius_factor = preprocess_data(data_specs,
                                                    list_features,
                                                    i_list_feat,
                                                    person_id_to_check,
                                                    simdata,
                                                    dec_tree_parameters,
                                                    circle_radius_l)

        # Plot simulation data without decision boundaries
        clf_to_plot = None
        #
        if show_figure:
            plot_simdata(clf_to_plot, x_y_dict["x_train"],
                         x_y_dict["y_train"],
                         data_specs["save_to_folder"],
                         dec_tree_parameters,
                         datapoint_of_interest,
                         title="Trainingdata - Step1",
                         which_step="step1",
                         fs=14,
                         linecolor="rosa",
                         lw=3, alpha=alpha,
                         show_figure=show_figure)
#
        # ####################################################
        # ############### STEP 1 #############################
        # ####################################################
        # Run STEP 1 of BAPC, Prediction with Decision Tree 1.
        hdtree_linear, df_y, \
            df_y_test, df_accuracies = bapc_step1(dec_tree_parameters,
                                                    x_y_dict,
                                                    df_y,
                                                    df_y_test)

        # Plot Data with Decision Boundary STEP 1
        rec_regions_x, rec_regions_y, ax, fig, _, _ = \
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   x_y_dict["y_train"],
                                   data_specs["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=True,
                                   which_step="step1",
                                   fs=12,
                                   title="Step 1 - Prediction DT1",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)

        if testdata:
            datapoint_of_interest_test = [0, 0]  # TODO
            rec_regions_x, rec_regions_y, ax, fig,_ ,_ = \
                plot_decision_boundary(hdtree_linear,
                                       x_y_dict["x_test"],
                                       x_y_dict["y_test"],
                                       data_specs["save_to_folder"],
                                       dec_tree_parameters,
                                       datapoint_of_interest_test,
                                       plot_color_area=True,
                                       which_step="step1",
                                       fs=12,
                                       title="Step 1 - Testdata",
                                       cmap='Set3',
                                       alpha=alpha,
                                       show_figure=show_figure)


        # ------------------
        # Generate Plot tree & visualize Decision Tree Structure with: http://webgraphviz.com/
        if dec_tree_parameters["which_cost_fct"] == "Gini":
            draw_graph(hdtree_linear, x_y_dict["x_train"],
                       data_specs["save_to_folder"],
                       addname=dec_tree_parameters["which_cost_fct"] + '_' +
                               dec_tree_parameters["nametosave_short"],
                       whichstep="1")

        # ####################################################
        # ############### STEP 2 #############################
        # ####################################################
    ##
        # --------------------------------------------
        # ------------- STEP 2 -----------------------
        # --------------------------------------------
        df_y, df_y_test, df_accuracies, \
            predicted_rf_tr, predicted_rf_test, \
            surr_predict, surr_predict_test, model_rf = \
            step2(x_y_dict,
                  dec_tree_parameters,
                  df_accuracies,
                  df_y, df_y_test)
##
        if show_figure:
            plot_decision_boundary(model_rf,
                                   x_y_dict["x_train"],
                                   df_y["epsilon"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="Epsilon - RandomForest",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
            plot_decision_boundary(model_rf,
                                   x_y_dict["x_train"],
                                   df_y["epsilon_hat"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="Epsilon_hat - RandomForest",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)

        ##
        # -----------------------------------------------------------
        # ############## ERROR AMPLIFICATION #################
        # -----------------------------------------------------------
        # Get the corresponding epsilon value for the given person_id
        # (for the given datapoint of interest)
        # check_for_eps_value = get_epsilon_value_for_person_id(
        #     df_y, data_specs["person_id_label"], person_id_to_check)

        df_y, df_y_big, circle_radius, \
            rec_oi_x, rec_oi_y = error_amplification(datapoint_of_interest,
                                                     x_y_dict["x_train"],
                                                     hdtree_linear,
                                                     data_specs,
                                                     df_y,
                                                     circle_radius_factor,
                                                     circle_around_one_point=True,
                                                     testdata = False)
        ##
        if testdata:
            df_y_test, df_y_test_big, circle_radius, \
                rec_oi_x, rec_oi_y = error_amplification(datapoint_of_interest_test,
                                                         x_y_dict["x_test"], hdtree_linear,
                                                         data_specs,
                                                         df_y_test,
                                                         circle_radius_factor,
                                                         circle_around_one_point=True,
                                                         testdata=True)

        if show_figure:
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   df_y.loc[:, "neighb_points"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="neighb_points",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
            #
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   df_y.loc[:, "circle_points"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="circle_points",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
            ##

        if show_figure:
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   df_y.loc[:, "extended_eps"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="extended_eps",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   df_y.loc[:, "epsilon"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="Epsilon",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   df_y.loc[:, "epsilon_hat"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="Epsilon_hat",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   df_y.loc[:, "eps_EA"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="eps_EA",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
        #
        # Testset
        if testdata:
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_test"],
                                   df_y_test.loc[:, "epsilon_hat"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest_test,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="Epsilon_hat-test",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)

            plot_decision_boundary(hdtree_linear, x_y_dict["x_test"], df_y_test.loc[:, "eps_EA"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest_test,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="eps_EA-test",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)

##

        # --------------------------------------------
        # ------------- STEP 3 -----------------------
        # --------------------------------------------

        hdtree_linear_s3, df_y, df_y_big, \
            df_y_test, df_accuracies = bapc_step3(dec_tree_parameters,
                                                  data_specs,
                                                  x_y_dict,
                                                  df_y,
                                                  df_y_big,
                                                  df_y_test,
                                                  df_accuracies)
        ##

        # generate plot tree
        if dec_tree_parameters["which_cost_fct"] == "Gini":
            draw_graph(hdtree_linear_s3, x_y_dict["x_train"],
                       dec_tree_parameters["save_to_folder"],
                       addname=dec_tree_parameters["which_cost_fct"] + '_' +
                               dec_tree_parameters["nametosave_short"], whichstep=which_step)

        if show_figure:
            # Plot modified input data with one (original) decision boundary
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_train"],
                                   df_y.loc[:, "y_epsilon_hat_ea"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   edgecolor=df_y['edgecolor'],
                                   circle_radius=circle_radius,
                                   plot_color_area=False,
                                   which_step="step3",
                                   fs=12,
                                   title="Step3 - Modified Input Data",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
        if testdata:
            plot_decision_boundary(hdtree_linear,
                                   x_y_dict["x_test"],
                                   df_y_test.loc[:, "y_epsilon_hat_ea"],
                                   dec_tree_parameters["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest_test,
                                   edgecolor=df_y_test['edgecolor'],
                                   circle_radius=None,
                                   plot_color_area=False,
                                   which_step="step3",
                                   fs=12,
                                   title="Step3 - Modified Input Data - Test",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)
##
        edgecolor = []
        if show_figure:
            plot_decision_boundary_two_dbound(hdtree_linear, hdtree_linear_s3,
                                              x_y_dict["x_train"],
                                              x_y_dict["y_train"],
                                              dec_tree_parameters["save_to_folder"],
                                              dec_tree_parameters,
                                              edgecolor,
                                              datapoint_of_interest,
                                              circle_radius=None,
                                              plot_color_area=True,
                                              plot_hyperbola=plot_hyperbola,
                                              plot_distance_to_hyperbola=False,
                                              which_step="step3",
                                              fs=12,
                                              title="Step 3 - Original data",
                                              cmap='Set3',
                                              figsize=(6.4, 4.8),
                                              show_figure=show_figure)


##      # -----------------------------
        # Change in Decision Boundaries.
        # -----------------------------
        # FEATURE IMPORTANCES
        # -----------------------------
        change_dec_boundary_l, \
            change_dec_boundary_l_old,\
            rec_around_x_oi, rec_around_y_oi = bapc_feature_importance(data_specs,
                                                        x_y_dict,
                                                        df_y_big,
                                                        hdtree_linear,
                                                        hdtree_linear_s3,
                                                        datapoint_of_interest,
                                                        rec_oi_x,
                                                        rec_oi_y)

        # -------------------------
        # PLOT FEATURE IMPORTANCES
        # -------------------------
        colors = ['mediumaquamarine', 'lightcoral']
        title = "Feature Importance"
        xlabel = 'Change of Decision Boundaries in Percent'
        if show_figure:
            fig_barplot, ax_barplot_change_db = plot_bar_change_of_decision_bound(
                change_dec_boundary_l, data_specs["K_featnames"],
                data_specs["save_to_folder"], dec_tree_parameters,
                title, xlabel,
                colors, show_figure, xlim=[-10, 10])

        # # OLD METHOD: Change of Decision boundaries
        # colors = ['teal', 'lightblue']
        # title = "Change of Decision Boundaries, crossing DOI\n(DT2 - DT1)"
        # xlabel = 'Change of Decision Boundaries in Percent'
        # if show_figure:
        #     fig_barplot, ax_barplot_change_db = plot_bar_change_of_decision_bound(
        #         change_dec_boundary_l_old, data_specs["K_featnames"],
        #         data_specs["save_to_folder"], dec_tree_parameters,
        #         title, xlabel,
        #         colors, show_figure)
##
        # --------------------------------------
        # ############# FIDELITY ###############
        # --------------------------------------
        df_y_big, df_y_in_rectangle_circle, \
            df_y_in_rectangle, df_y_full_circle, \
            df_summary_fidelity = bapc_fidelity(data_specs,
                                                df_y_big,
                                                datapoint_of_interest,
                                                rec_around_x_oi,
                                                rec_around_y_oi)
        ##

        if show_figure:
            plot_decision_boundary(hdtree_linear, x_y_dict["x_train"],
                                   df_y_big.loc[:, "within_rectangle_and_circle"],
                                   data_specs["save_to_folder"],
                                   dec_tree_parameters,
                                   datapoint_of_interest,
                                   plot_color_area=False,
                                   which_step="error_ampl",
                                   fs=12,
                                   title="within_rectangle_and_circle",
                                   cmap='Set3',
                                   alpha=alpha,
                                   show_figure=show_figure)

        if show_figure:
            edgecolor = []
            plot_decision_boundary_two_dbound(hdtree_linear, hdtree_linear_s3,
                                              df_y_in_rectangle_circle[[data_specs["K_featnames"][0],
                                                                        data_specs["K_featnames"][1]]],
                                              df_y_in_rectangle_circle["y_train"],
                                              data_specs["save_to_folder"],
                                              dec_tree_parameters,
                                              edgecolor,
                                              datapoint_of_interest,
                                              circle_radius=None,
                                              plot_color_area=True,
                                              plot_hyperbola=plot_hyperbola,
                                              plot_distance_to_hyperbola=False,
                                              which_step="step3",
                                              fs=12,
                                              title="Region - Fidelity-Calculation",
                                              cmap='Set3',
                                              figsize=(6.4, 4.8))


##
        #if show_figure:
        plot_decision_boundary_two_dbound(hdtree_linear, hdtree_linear_s3,
                                              x_y_dict["x_train"],
                                              df_y["y_epsilon_hat_ea"],
                                              data_specs["save_to_folder"],
                                              dec_tree_parameters,
                                              df_y['edgecolor'],
                                              datapoint_of_interest,
                                              circle_radius,
                                              plot_color_area=True,
                                              plot_hyperbola=plot_hyperbola,
                                              plot_distance_to_hyperbola=plot_hyperbola,
                                              which_step="step3",
                                              fs=12,
                                              title="radius = " + str(circle_radius_factor) + " , Accuracy(DT1 + AI) =" +
                                                    str(df_accuracies.loc["acc_train", "DT1 + AI"]) +
                                                    " \nfidelity_rectangle = " +
                                                    str(round(df_summary_fidelity.at[0, 'Fidelity_rectangle'], 3)) +
                                                    "\nfidelity_circle_rec = " +
                                                    str(round(df_summary_fidelity.at[0, 'Fidelity_circle_rec'], 3)) +
                                                    "\nfidelity_circle = " +
                                                    str(round(df_summary_fidelity.at[0, 'Fidelity_circle'], 3)) +
                                                    "\n fidelity_all = " +
                                                    str(round(df_summary_fidelity.at[0, 'Fidelity_all'], 3)) +
                                                    "\n" +
                                                    r"$R^2$_circle = " + str(round(df_summary_fidelity.at[0, 'R2_circle'], 2)) +
                                                    "\n" +
                                                    r" $R^2$_all = " + str(round(df_summary_fidelity.at[0, 'R2_all'], 3)),
                                              cmap='Set3',
                                          figsize=(6.4, 7.3),
                                              show_figure=True)

        ##
        clfa = None
        if show_figure:
            plot_simdata(clfa, x_y_dict["x_train"], df_y["y_epsilon_hat_ea"],
                         data_specs["save_to_folder"], dec_tree_parameters,
                         datapoint_of_interest,
                         title="Modified input data",
                         which_step="step3",
                         fs=14,
                         linecolor="rosa",
                         lw=3, alpha=1, show_figure=show_figure)

##
        # ------------------------------------------------------
        # ----------- PLOT YLABEL-PREDICTIONS ------------------
        # ------------------------------------------------------
        # Plot Ylabel-Predictions for: True Label, DT1, DT3, DT1+AI
        df_prediction = plot_y_predictions(df_y_big, data_specs, dec_tree_parameters)
    ##
        # --------------------------------------
        # ----------- SUMMARY ------------------
        # --------------------------------------
        # Summary of Results: Fidelity, Accuracy, Change of Dec Boundaries, for each Feature Set
        if i_list_feat==0:
            df_summary = pd.DataFrame(columns=["DataIndex", "radius", "num_points_circle", "Y_true", "Y_predicted_DT1",
                                               "Y_predicted_DT1+AI", "Y_predicted_DT3",
                                               "Y_true=Y_pred_DT1+AI", "Y_true=Y_predicted_DT3",
                                               "Y_true=Y_AI=Y_DT3",
                                               "Y_pred_DT1=Y_pred_DT1_AI",
                                               "Change of Dec Boundary Feat1",
                                               "Change of Dec Boundary Feat2",
                                               "Fidelity_rectangle",
                                               "Fidelity_circle_rec",
                                               "Fidelity_circle",
                                               "R2_rectangle",
                                               "R2_circle_rectangle",
                                               "R2_circle",
                                               "R2_all",
                                               "Accuracy_DT1_Traindata",
                                               "Accuracy_DT3_Traindata",
                                               "Accuracy_DT1+AI_Traindata",
                                               "Accuracy_DT3+AI_Testdata",
                                               "Accuracy_DT1_Testdata",
                                               "Accuracy_DT3_Testdata"],
                                      index=range(0, len(list_features)))

    ##
        # Rename Indices with Features.
        idx = data_specs["K_featnames"][0] + '_' + data_specs["K_featnames"][1]
        df_summary.rename(index={i_list_feat:idx}, inplace=True)
        df_summary.loc[idx, "DataIndex"] = person_id_to_check
        df_summary.loc[idx, "radius"] = circle_radius_factor
        df_summary.loc[idx, "num_points_circle"] = len(df_y_full_circle)
        df_summary.loc[idx, "Y_true"] = df_prediction.loc["Y_true","prediction"]
        df_summary.loc[idx, "Y_predicted_DT1"] = df_prediction.loc["Y_pred_DT1","prediction"]
        df_summary.loc[idx, "Y_predicted_DT1+AI"] = df_prediction.loc["Y_pred_DT1+AI","prediction"]
        df_summary.loc[idx, "Y_predicted_DT3"] = df_prediction.loc["Y_pred_DT3","prediction"]
        df_summary.loc[idx, "Y_true=Y_pred_DT1+AI"] = df_prediction.loc["Y_true","prediction"] - \
                                                      df_prediction.loc["Y_pred_DT1+AI","prediction"]
        df_summary.loc[idx, "Y_true=Y_predicted_DT3"] = df_prediction.loc["Y_true","prediction"] - \
                                                        df_prediction.loc["Y_pred_DT3","prediction"]
        df_summary.loc[idx, "Y_true=Y_AI=Y_DT3"] = df_summary.loc[idx, "Y_true=Y_pred_DT1+AI"] - \
                                                   df_summary.loc[idx, "Y_true=Y_predicted_DT3"]
        df_summary.loc[idx, "Y_pred_DT1=Y_pred_DT1_AI"] = df_prediction.loc["Y_pred_DT1","prediction"] - \
                                                          df_prediction.loc["Y_pred_DT1+AI","prediction"]
        df_summary.loc[idx, "Change of Dec Boundary Feat1"] = change_dec_boundary_l[0]
        df_summary.loc[idx, "Change of Dec Boundary Feat2"] = change_dec_boundary_l[1]
        df_summary.loc[idx, "Fidelity_rectangle"] = df_summary_fidelity["Fidelity_rectangle"]
        df_summary.loc[idx, "Fidelity_circle_rec"] = df_summary_fidelity["Fidelity_circle_rec"]
        df_summary.loc[idx, "Fidelity_circle"] = df_summary_fidelity["Fidelity_circle"]

        df_summary.loc[idx, "R2_rectangle"] = df_summary_fidelity["R2_rectangle"]
        df_summary.loc[idx, "R2_circle_rectangle"] = df_summary_fidelity["R2_circle_rectangle"]
        df_summary.loc[idx, "R2_circle"] = df_summary_fidelity["R2_circle"]
        df_summary.loc[idx, "R2_all"] = df_summary_fidelity["R2_all"]

        df_summary.loc[idx, "Accuracy_DT1_Traindata"] = df_accuracies.loc["acc_train", "DT1"]
        df_summary.loc[idx, "Accuracy_DT3_Traindata"] = df_accuracies.loc["acc_train", "DT3"]
        df_summary.loc[idx, "Accuracy_DT1+AI_Traindata"] = df_accuracies.loc["acc_train", "DT1 + AI"]
        df_summary.loc[idx, "Accuracy_DT3+AI_Testdata"] = df_accuracies.loc["acc_test", "DT1 + AI"]
        df_summary.loc[idx, "Accuracy_DT1_Testdata"] = df_accuracies.loc["acc_test", "DT1"]
        df_summary.loc[idx, "Accuracy_DT3_Testdata"] = df_accuracies.loc["acc_test", "DT3"]

        print("df_summary.csv saved in loop. " + str(i_list_feat))

    print("save csv.")
    print(os.path.dirname(data_specs["save_to_folder"]))
    df_summary.to_csv(os.path.join(os.path.dirname(data_specs["save_to_folder"]),
                                   "summary_Accuracy_changeDecBound_Fidelity" +
                                   dec_tree_parameters["nametosave_short"] + '_personID_' +
                                   str(person_id_to_check) + "_nfeat" +
                                   str(len(list_features)) + ".csv"))

    return df_summary

