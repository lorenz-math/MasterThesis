from CollectUtils_SemiLinWave import *

#this file collects all the the norms and constants computed in CollectUtils_SemiLinWave and puts them
#into one collective dataframe convergenceAnalysis_df and computes the mean of convergenceAnalysis_df

np.random.seed(42)

base_path_list = ["Run05_Fin/80_64_0", "Run05_Fin/245_343_0", "Run05_Fin/500_1000_0", "Run05_Fin/845_2197_0", "Run05_Fin/1280_4096_0", "Run05_Fin/2000_8000_0", "Run05_Fin/3125_15625_0"]

convergenceAnalysis_df = pd.DataFrame()

meanConvergenceAnalysis_df = pd.DataFrame()

BestConvergenceAnalysis_df = pd.DataFrame()

for base_path in base_path_list:
    print("#################################################")
    print(base_path)
    print(os.listdir(base_path))

    b = False
    compute_std = True
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame()
    selection_criterion = "error_train"

    Nu_list = []
    Nf_list = []

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        model_path = base_path

        sample_path = os.path.join(model_path, subdirec)
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        info_model = select_over_retrainings2(sample_path, selection=selection_criterion, mode="mean", compute_std=compute_std, compute_val=True, rs_val=0)
        sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
        convergenceAnalysis_df = convergenceAnalysis_df.append(info_model, ignore_index=True)

    sensitivity_df = sensitivity_df.sort_values(selection_criterion)
    best_setup = sensitivity_df.iloc[0]
    mean_setup = sensitivity_df.mean(axis=0)
    meanConvergenceAnalysis_df = meanConvergenceAnalysis_df.append(mean_setup, ignore_index=True)
    BestConvergenceAnalysis_df = BestConvergenceAnalysis_df.append(best_setup, ignore_index=True)

    best_setup.to_csv(os.path.join(base_path, "best.csv"), header = 0)
    mean_setup.to_csv(os.path.join(base_path, "mean.csv"), header = 0)
    sensitivity_df.to_csv(os.path.join(base_path, "ConvergenceAnalysis.csv"), index = False)


    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(sensitivity_df[selection_criterion], sensitivity_df["L2_norm_test"])
    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon_G$')
    # plt.show()
    # quit()
    plt.savefig(os.path.join(base_path, "et_vs_eg.png"), dpi=400)

convergenceAnalysis_df.to_csv(os.path.join("ReLu1", "ConvergenceAnalysis.csv"), index = False)
meanConvergenceAnalysis_df.to_csv(os.path.join("ReLu1", "meanConvergenceAnalysis.csv"), index = False)

quit()
