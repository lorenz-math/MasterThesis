"""
Nothing was changed here; this file is only here to show the difference between this and the new file
CollectEnsembleData_SemiLinWave.py
"""
from CollectUtils import *

np.random.seed(42)

base_path_list = ["Test\\5_10_0", "Test\\10_20_0"]
folder = ["Test"]

convergenceAnalysis_df = pd.DataFrame(columns=["batch_size",
                                        "regularization_parameter",
                                        "kernel_regularizer",
                                        "neurons",
                                        "hidden_layers",
                                        "residual_parameter",
                                        "L2_norm_test",
                                        "error_train",
                                        "error_val",
                                        "error_test"])

meanConvergenceAnalysis_df = pd.DataFrame(columns=["batch_size",
                                        "regularization_parameter",
                                        "kernel_regularizer",
                                        "neurons",
                                        "hidden_layers",
                                        "residual_parameter",
                                        "L2_norm_test",
                                        "error_train",
                                        "error_val",
                                        "error_test"])

for base_path in base_path_list:
    print("#################################################")
    print(base_path)
    print(os.listdir(base_path))

    b = False
    compute_std = True
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame(columns=["batch_size",
                                           "regularization_parameter",
                                           "kernel_regularizer",
                                           "neurons",
                                           "hidden_layers",
                                           "residual_parameter",
                                           "L2_norm_test",
                                           "error_train",
                                           "error_val",
                                           "error_test"])
    # print(sensitivity_df)
    selection_criterion = "error_train"

    Nu_list = []
    Nf_list = []
    t_0 = 0
    t_f = 1
    x_0 = -1
    x_f = 1

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        model_path = base_path

        sample_path = model_path + "\\" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            if os.path.isfile(sample_path + "\\" + ret + "\\TrainedModel\\Information.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file is not None:
            info_model = pd.read_csv(sample_path + "\\" + retr_to_check_file + "\\TrainedModel\\Information.csv", header=0, sep=",")
            best_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode="mean", compute_std=compute_std, compute_val=True, rs_val=0)
            info_model["error_train"] = best_retrain["error_train"]
            info_model["train_time"] = best_retrain["train_time"]
            info_model["error_val"] = 0
            info_model["error_test"] = 0
            info_model["L2_norm_test"] = best_retrain["L2_norm_test"]
            info_model["rel_L2_norm"] = best_retrain["rel_L2_norm"]
            if os.path.isfile(sample_path + "\\" + retr_to_check_file + "\\Images\\errors.txt"):
                info_model["l2_glob"] = best_retrain["l2_glob"]
                info_model["l2_glob_rel"] = best_retrain["l2_glob_rel"]
                info_model["l2_om_big"] = best_retrain["l2_om_big"]
                info_model["l2_om_big_rel"] = best_retrain["l2_om_big_rel"]
                info_model["h1_glob"] = best_retrain["h1_glob"]
                info_model["h1_glob_rel"] = best_retrain["h1_glob_rel"]
                try:
                    info_model["l2_p_rel"] = best_retrain["l2_p_rel"]
                except:
                    print("l2_p_rel not found")
                try:
                    info_model["h1_p_rel"] = best_retrain["h1_p_rel"]
                except:
                    print("h1_p_rel not found")
            info_model["setup"] = setup_num
            info_model["retraining"] = best_retrain["retraining"]
            info_model["loss_u0_train"]=best_retrain["loss_u0_train"]
            info_model["loss_ub0_train"]=best_retrain["loss_ub0_train"]
            info_model["loss_ub1_train"]=best_retrain["loss_ub1_train"]
            info_model["res_loss"]=best_retrain["res_loss"]
            info_model["val_gap_u0"]=best_retrain["loss_u0_train"]-best_retrain["loss_u0_val"]
            info_model["val_gap_ub0"]=best_retrain["loss_ub0_train"]-best_retrain["loss_ub0_val"]
            info_model["val_gap_ub1"]=best_retrain["loss_ub1_train"]-best_retrain["loss_ub1_val"]
            info_model["val_gap_int"]=best_retrain["res_loss"]-best_retrain["res_loss_val"]
            info_model["sigma_Ru0"]=best_retrain["sigma_Ru0"]
            info_model["sigma_Rint"]=best_retrain["sigma_Rint"]
            info_model["sigma_Rub0"]=best_retrain["sigma_Rub0"]
            info_model["sigma_Rub1"]=best_retrain["sigma_Rub1"]
            info_model["c2_0"]=best_retrain["c2_0"]
            info_model["c2_1"]=best_retrain["c2_1"]
            info_model["Nu_train"]=best_retrain["Nu_train"]
            info_model["Nf_train"]=best_retrain["Nf_train"]

            if info_model["batch_size"].values[0] == "full":
                info_model["batch_size"] = best_retrain["Nu_train"] + best_retrain["Nf_train"]
            sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
            convergenceAnalysis_df = convergenceAnalysis_df.append(info_model, ignore_index=True)
        else:
            print(sample_path + "\\TrainedModel\\Information.csv not found")


    sensitivity_df = sensitivity_df.sort_values(selection_criterion)
    best_setup = sensitivity_df.iloc[0]
    mean_setup = sensitivity_df.mean(axis=0)
    meanConvergenceAnalysis_df = meanConvergenceAnalysis_df.append(mean_setup, ignore_index=True)

    best_setup.to_csv(base_path + "\\best.csv", header = 0)
    mean_setup.to_csv(base_path + "\\mean.csv", header = 0)
    sensitivity_df.to_csv(base_path + "\\ConvergenceAnalysis.csv", index = False)


    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(sensitivity_df[selection_criterion], sensitivity_df["L2_norm_test"])
    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon_G$')
    plt.savefig(base_path + "\\et_vs_eg.png", dpi=400)

convergenceAnalysis_df.to_csv("Test\\ConvergenceAnalysis.csv", index = False)
meanConvergenceAnalysis_df.to_csv("Test\\meanConvergenceAnalysis.csv", index = False)

quit()
