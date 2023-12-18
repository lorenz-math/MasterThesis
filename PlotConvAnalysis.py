"""
Nothing was changed here; this file is only here to show the difference between this and the new file
PlotConvAnalysis_SemiLinWave.py
"""
from CollectUtils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.random.seed(42)

sensitivity_df = pd.read_csv("Test\\ConvergenceAnalysis.csv")
mean_df = pd.read_csv("Test\\meanConvergenceAnalysis.csv")
Nf_list = [10,20]
Nu_list = [5,10]

c2 = 0
c1 = np.sqrt(1 + np.exp(1))
fig = plt.figure()
plt.grid(True, which="both", ls=":")
sensitivity_df['Nu']=sensitivity_df['Nu_train']+sensitivity_df['Nf_train']
mean_df['Nu']=mean_df['Nu_train']+mean_df['Nf_train']

sensitivity_df['N0_train'] = sensitivity_df['Nu_train'] / 2
sensitivity_df['Nb0_train'] = sensitivity_df['Nu_train'] / 4
sensitivity_df['Nb1_train'] = sensitivity_df['Nu_train'] / 4

mean_df['N0_train'] = mean_df['Nu_train'] / 2
mean_df['Nb0_train'] = mean_df['Nu_train'] / 4
mean_df['Nb1_train'] = mean_df['Nu_train'] / 4

sensitivity_df['bound'] = np.sqrt(c1 ** 2 * (sensitivity_df['loss_u0_train'] ** 2 +
                               sensitivity_df['c2_0'] ** 2 * sensitivity_df['loss_ub0_train'] ** 2 +
                               sensitivity_df['c2_1'] ** 2 * sensitivity_df['loss_ub1_train'] ** 2 +
                               sensitivity_df['res_loss'] ** 2 +
                               sensitivity_df['val_gap_u0'] ** 2 +
                               sensitivity_df['c2_0'] ** 2 * sensitivity_df['val_gap_ub0'] ** 2 +
                               sensitivity_df['c2_1'] ** 2 * sensitivity_df['val_gap_ub1'] ** 2 +
                               sensitivity_df['val_gap_int'] ** 2 +
                               c1 ** 2 * (sensitivity_df['sigma_Ru0'] / sensitivity_df['N0_train'] ** 0.5 +
                                          sensitivity_df['sigma_Rint'] / sensitivity_df['Nf_train'] ** 0.5 +
                                          sensitivity_df['c2_0'] ** 2 * np.sqrt(sensitivity_df['sigma_Rub0'] / sensitivity_df['Nb0_train'] ** 0.5) +
                                          sensitivity_df['c2_1'] ** 2 * np.sqrt(sensitivity_df['sigma_Rub1'] / sensitivity_df['Nb1_train'] ** 0.5))).values)


mean_df['bound'] = np.sqrt(c1 ** 2 * (mean_df['loss_u0_train'] ** 2 +
                               mean_df['c2_0'] ** 2 * mean_df['loss_ub0_train'] ** 2 +
                               mean_df['c2_1'] ** 2 * mean_df['loss_ub1_train'] ** 2 +
                               mean_df['res_loss'] ** 2 +
                               mean_df['val_gap_u0'] ** 2 +
                               mean_df['c2_0'] ** 2 * mean_df['val_gap_ub0'] ** 2 +
                               mean_df['c2_1'] ** 2 * mean_df['val_gap_ub1'] ** 2 +
                               mean_df['val_gap_int'] ** 2 +
                               c1 ** 2 * (mean_df['sigma_Ru0'] / mean_df['N0_train'] ** 0.5 +
                                          mean_df['sigma_Rint'] / mean_df['Nf_train'] ** 0.5 +
                                          mean_df['c2_0'] ** 2 * np.sqrt(mean_df['sigma_Rub0'] / mean_df['Nb0_train'] ** 0.5) +
                                          mean_df['c2_1'] ** 2 * np.sqrt(mean_df['sigma_Rub1'] / mean_df['Nb1_train'] ** 0.5))).values)


plt.scatter(sensitivity_df.Nu.values, sensitivity_df.L2_norm_test.values, color='pink', zorder = 0)   
plt.scatter(sensitivity_df.Nu.values,
            np.sqrt(sensitivity_df.loss_ub0_train.values ** 2 + sensitivity_df.loss_ub1_train.values ** 2 + sensitivity_df.loss_u0_train.values ** 2 + sensitivity_df.res_loss.values ** 2), 
            color='lightgray', zorder=0)
plt.scatter(sensitivity_df.Nu.values, sensitivity_df.bound, color='lightsteelblue', zorder=0)

plt.scatter(mean_df.Nu.values, mean_df.L2_norm_test.values, label=r'$\varepsilon_G$', color='red')   
plt.scatter(mean_df.Nu.values,
            np.sqrt(mean_df.loss_ub0_train.values ** 2 + mean_df.loss_ub1_train.values ** 2 + mean_df.loss_u0_train.values ** 2 + mean_df.res_loss.values ** 2),
            label=r'$\varepsilon_T$', color='gray')
plt.scatter(mean_df.Nu.values, mean_df.bound, label=r'$Bound$', color='blue')

for i, Nf in enumerate(Nf_list):
    print(Nf)
    scale = scale_vec[i]
    index_list_i = sensitivity_df.index[sensitivity_df.Nf_train == Nf]
    new_df = sensitivity_df.loc[index_list_i]
    new_df['N0_train'] = new_df['Nu_train'] / 2
    new_df['Nb0_train'] = new_df['Nu_train'] / 4
    new_df['Nb1_train'] = new_df['Nu_train'] / 4
    bound = (new_df['error_train'] + 2 * new_df['sigma_res'] / new_df['Nf_train'] ** (1 / 2) + 2 * (new_df['sigma_u'] + new_df['sigma_u_star']) / new_df['Nu_train'] ** (
          1 / 2)).values

    bound = np.sqrt(c1 ** 2 * (new_df['loss_u0_train'] ** 2 +
                               new_df['c2_0'] ** 2 * new_df['loss_ub0_train'] ** 2 +
                               new_df['c2_1'] ** 2 * new_df['loss_ub1_train'] ** 2 +
                               new_df['res_loss'] ** 2 +
                               new_df['val_gap_u0'] ** 2 +
                               new_df['c2_0'] ** 2 * new_df['val_gap_ub0'] ** 2 +
                               new_df['c2_1'] ** 2 * new_df['val_gap_ub1'] ** 2 +
                               new_df['val_gap_int'] ** 2 +
                               c1 ** 2 * (new_df['sigma_Ru0'] / new_df['N0_train'] ** 0.5 +
                                          new_df['sigma_Rint'] / new_df['Nf_train'] ** 0.5 +
                                          new_df['c2_0'] ** 2 * np.sqrt(new_df['sigma_Rub0'] / new_df['Nb0_train'] ** 0.5) +
                                          new_df['c2_1'] ** 2 * np.sqrt(new_df['sigma_Rub1'] / new_df['Nb1_train'] ** 0.5))).values)

    
    if Nf == 20:
        plt.scatter(new_df.Nu_train.values, new_df.L2_norm_test.values, label=r'$\varepsilon_G$', color=lighten_color('red', scale), zorder=0)    
        plt.scatter(new_df.Nu_train.values,
                    np.sqrt(new_df.loss_ub0_train.values ** 2 + new_df.loss_ub1_train.values ** 2 + new_df.loss_u0_train.values ** 2 + new_df.res_loss.values ** 2),
                    label=r'$\varepsilon_T$', color=lighten_color('gray', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values, bound, label=r'$Bound$', color=lighten_color('blue', scale), zorder=0)
    else:
        plt.scatter(new_df.Nu_train.values, new_df.L2_norm_test.values, color=lighten_color('red', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values,
                    np.sqrt(new_df.loss_ub0_train.values ** 2 + new_df.loss_ub1_train.values ** 2 + new_df.loss_u0_train.values ** 2 + new_df.res_loss.values ** 2),
                    color=lighten_color('gray', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values, new_df.val_gap_ub0.values + new_df.val_gap_ub1.values + new_df.val_gap_int.values + new_df.res_loss.values,
                    color=lighten_color('DarkOrange', scale), zorder=0)
        plt.scatter(new_df.Nu_train.values, bound, color=lighten_color('blue', scale), zorder=0)
plt.legend()
plt.xscale("log")
plt.xlabel(r'$N_u$')
plt.yscale("log")
plt.savefig("Conv.png", dpi=400)
plt.show()


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

quit()
