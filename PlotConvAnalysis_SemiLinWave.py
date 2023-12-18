from CollectUtils_SemiLinWave import *
from scipy.interpolate import CubicSpline

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.random.seed(42)

#this file plots total error, total error + total error of time derivative, training error and bound

pi = math.pi

#all the constants that are needed for Theorem 3.4
C_pw= 1 / math.sqrt(2)
C_Omega = 1 / 12
C_OmegaT = 1 / 8
C_dOmega = 1 / 3
d=2
T=0.5


#sensitivity_df holds the the constants and error norms for all models, mean_df is the mean over sensitivity_df
sensitivity_df = pd.read_csv("Run05_Fin\\ConvergenceAnalysis.csv")
mean_df = pd.read_csv("Run05_Fin\meanConvergenceAnalysis.csv")

#compute the training error
sensitivity_df["E_T"]=np.sqrt(sensitivity_df["res_loss"] ** 2 + sensitivity_df["loss_su"] ** 2 
                              + sensitivity_df["loss_sut"] ** 2 + sensitivity_df["loss_u0"] ** 2 
                              + sensitivity_df["loss_u1"] ** 2 + sensitivity_df["loss_nablau"] ** 2)

mean_df["E_T"]=np.sqrt(mean_df["res_loss"] ** 2 + mean_df["loss_su"] ** 2 
                              + mean_df["loss_sut"] ** 2 + mean_df["loss_u0"] ** 2 
                              + mean_df["loss_u1"] ** 2 + mean_df["loss_nablau"] ** 2)


fig = plt.figure()
plt.grid(True, which="both", ls=":")
#computing total number of trainin points (Nu_train), collocation points (Nf_train),
#training points on boundary (Nb_train), training points for initial value (Ni_train)
sensitivity_df['Nu']=sensitivity_df['Nu_train']+sensitivity_df['Nf_train']
mean_df['Nu']=mean_df['Nu_train']+mean_df['Nf_train']

sensitivity_df['Nb_train'] = (sensitivity_df['Nu_train'] / 5) * 4
sensitivity_df['Ni_train'] = sensitivity_df['Nu_train'] / 5

mean_df['Nb_train'] = (mean_df['Nu_train'] / 5) * 4
mean_df['Ni_train'] = mean_df['Nu_train'] / 5

#The constants C_1 to C_4 which include the C^2 norm of the square residuals
c1_s = C_Omega * sensitivity_df['R_u1']
c2_s = C_OmegaT * sensitivity_df['R_PDE']
c3_s = C_Omega * sensitivity_df['R_nablau']
c4_s = np.sqrt(C_dOmega * sensitivity_df['R_sut'])
c5_s = 2 * C_Omega / (C_pw ** 2) * sensitivity_df['R_u0']

c1_m = C_Omega * mean_df['R_u1']
c2_m = C_OmegaT * mean_df['R_PDE']
c3_m = C_Omega * mean_df['R_nablau']
c4_m = np.sqrt(C_dOmega * mean_df['R_sut'])
c5_m = 2 * C_Omega / (C_pw ** 2) * mean_df['R_u0']

#the bound as in equation 4.4 for the all runs (sensitivity_df) and the mean of all runs
sensitivity_df['bound'] = T * (c1_s * sensitivity_df['Nb_train'] ** (-2 / d) + sensitivity_df['loss_u1'] ** 2 + c2_s * sensitivity_df['Nf_train'] ** (-2 / (d+1))
                           + sensitivity_df['res_loss'] ** 2 + c3_s * sensitivity_df['Nb_train'] ** (- 2 / d) + sensitivity_df['loss_nablau'] ** 2
                           + 4 * math.sqrt(T) * sensitivity_df['u_C1'] *
                           (c4_s * sensitivity_df['Ni_train'] ** (-1 / d) + sensitivity_df['loss_sut']) + c5_s * sensitivity_df['Nb_train'] ** (- 2 / d)
                           + 2 / (C_pw ** 2) * sensitivity_df['loss_u0'] ** 2) * math.exp(T *  (1 + (2 * math.sqrt(T)) / (C_pw ** 2)))




mean_df['bound'] = T * (c1_m * mean_df['Nb_train'] ** (-2 / d) + mean_df['loss_u1'] ** 2 + c2_m * mean_df['Nf_train'] ** (-2 / (d+1))
                           + mean_df['res_loss'] ** 2 + c3_m * mean_df['Nb_train'] ** (-2 / d) + mean_df['loss_nablau'] ** 2
                           + 4 * math.sqrt(T) * mean_df['u_C1'] * (c4_m * mean_df['Ni_train'] ** (-1 / d) + mean_df['loss_sut'])  + c5_m * mean_df['Nb_train'] ** (- 2 / d)
                           + 2 / (C_pw ** 2) * mean_df['loss_u0'] ** 2) * math.exp(T * (1 + (2 * math.sqrt(T))/ (C_pw ** 2)))


#plot total error, total error + total error of time derivative, training error and bound

sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==80,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==80,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==245,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==245,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==500,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==500,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==845,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==845,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==1280,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==1280,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==2000,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==2000,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==3125,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==3125,'L2_norm_test'].values

yerrL2 = np.zeros((2,7))
yerrL2[0][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff'].min()
yerrL2[1][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff'].max()
yerrL2[0][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff'].min()
yerrL2[1][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff'].max()
yerrL2[0][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff'].min()
yerrL2[1][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff'].max()
yerrL2[0][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff'].min()
yerrL2[1][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff'].max()
yerrL2[0][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff'].min()
yerrL2[1][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff'].max()
yerrL2[0][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff'].min()
yerrL2[1][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff'].max()
yerrL2[0][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff'].min()
yerrL2[1][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff'].max()
yerrL2 = np.abs(yerrL2)

sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==80,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==80,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==245,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==245,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==500,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==500,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==845,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==845,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==1280,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==1280,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==2000,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==2000,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==3125,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==3125,'L2_norm_test'].values

yerrL2 = np.zeros((2,7))
yerrL2[0][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff'].min()
yerrL2[1][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff'].max()
yerrL2[0][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff'].min()
yerrL2[1][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff'].max()
yerrL2[0][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff'].min()
yerrL2[1][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff'].max()
yerrL2[0][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff'].min()
yerrL2[1][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff'].max()
yerrL2[0][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff'].min()
yerrL2[1][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff'].max()
yerrL2[0][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff'].min()
yerrL2[1][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff'].max()
yerrL2[0][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff'].min()
yerrL2[1][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff'].max()
yerrL2 = np.abs(yerrL2)

sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==80,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==80,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==245,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==245,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==500,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==500,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==845,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==845,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==1280,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==1280,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==2000,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==2000,'L2_norm_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==3125,'L2_norm_test']-mean_df.loc[mean_df['Nu_train']==3125,'L2_norm_test'].values

yerrL2 = np.zeros((2,7))
yerrL2[0][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff'].min()
yerrL2[1][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_norm_test_diff'].max()
yerrL2[0][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff'].min()
yerrL2[1][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_norm_test_diff'].max()
yerrL2[0][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff'].min()
yerrL2[1][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_norm_test_diff'].max()
yerrL2[0][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff'].min()
yerrL2[1][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_norm_test_diff'].max()
yerrL2[0][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff'].min()
yerrL2[1][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_norm_test_diff'].max()
yerrL2[0][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff'].min()
yerrL2[1][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_norm_test_diff'].max()
yerrL2[0][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff'].min()
yerrL2[1][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_norm_test_diff'].max()
yerrL2 = np.abs(yerrL2)

sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_t_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==80,'L2_t_test']-mean_df.loc[mean_df['Nu_train']==80,'L2_t_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_t_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==245,'L2_t_test']-mean_df.loc[mean_df['Nu_train']==245,'L2_t_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_t_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==500,'L2_t_test']-mean_df.loc[mean_df['Nu_train']==500,'L2_t_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_t_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==845,'L2_t_test']-mean_df.loc[mean_df['Nu_train']==845,'L2_t_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_t_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==1280,'L2_t_test']-mean_df.loc[mean_df['Nu_train']==1280,'L2_t_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_t_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==2000,'L2_t_test']-mean_df.loc[mean_df['Nu_train']==2000,'L2_t_test'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_t_test_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==3125,'L2_t_test']-mean_df.loc[mean_df['Nu_train']==3125,'L2_t_test'].values

yerrL2t = np.zeros((2,7))
yerrL2t[0][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_t_test_diff'].min()
yerrL2t[1][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'L2_t_test_diff'].max()
yerrL2t[0][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_t_test_diff'].min()
yerrL2t[1][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'L2_t_test_diff'].max()
yerrL2t[0][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_t_test_diff'].min()
yerrL2t[1][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'L2_t_test_diff'].max()
yerrL2t[0][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_t_test_diff'].min()
yerrL2t[1][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'L2_t_test_diff'].max()
yerrL2t[0][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_t_test_diff'].min()
yerrL2t[1][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'L2_t_test_diff'].max()
yerrL2t[0][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_t_test_diff'].min()
yerrL2t[1][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'L2_t_test_diff'].max()
yerrL2t[0][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_t_test_diff'].min()
yerrL2t[1][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'L2_t_test_diff'].max()
yerrL2t = np.abs(yerrL2t)

sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'E_T_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==80,'E_T']-mean_df.loc[mean_df['Nu_train']==80,'E_T'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'E_T_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==245,'E_T']-mean_df.loc[mean_df['Nu_train']==245,'E_T'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'E_T_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==500,'E_T']-mean_df.loc[mean_df['Nu_train']==500,'E_T'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'E_T_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==845,'E_T']-mean_df.loc[mean_df['Nu_train']==845,'E_T'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'E_T_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==1280,'E_T']-mean_df.loc[mean_df['Nu_train']==1280,'E_T'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'E_T_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==2000,'E_T']-mean_df.loc[mean_df['Nu_train']==2000,'E_T'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'E_T_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==3125,'E_T']-mean_df.loc[mean_df['Nu_train']==3125,'E_T'].values

yerrET = np.zeros((2,7))
yerrET[0][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'E_T_diff'].min()
yerrET[1][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'E_T_diff'].max()
yerrET[0][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'E_T_diff'].min()
yerrET[1][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'E_T_diff'].max()
yerrET[0][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'E_T_diff'].min()
yerrET[1][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'E_T_diff'].max()
yerrET[0][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'E_T_diff'].min()
yerrET[1][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'E_T_diff'].max()
yerrET[0][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'E_T_diff'].min()
yerrET[1][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'E_T_diff'].max()
yerrET[0][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'E_T_diff'].min()
yerrET[1][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'E_T_diff'].max()
yerrET[0][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'E_T_diff'].min()
yerrET[1][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'E_T_diff'].max()
yerrET = np.abs(yerrET)

sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'bound_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==80,'bound']-mean_df.loc[mean_df['Nu_train']==80,'bound'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'bound_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==245,'bound']-mean_df.loc[mean_df['Nu_train']==245,'bound'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'bound_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==500,'bound']-mean_df.loc[mean_df['Nu_train']==500,'bound'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'bound_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==845,'bound']-mean_df.loc[mean_df['Nu_train']==845,'bound'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'bound_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==1280,'bound']-mean_df.loc[mean_df['Nu_train']==1280,'bound'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'bound_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==2000,'bound']-mean_df.loc[mean_df['Nu_train']==2000,'bound'].values
sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'bound_diff']=sensitivity_df.loc[sensitivity_df['Nu_train']==3125,'bound']-mean_df.loc[mean_df['Nu_train']==3125,'bound'].values

yerrb = np.zeros((2,7))
yerrb[0][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'bound_diff'].min()
yerrb[1][0]=sensitivity_df.loc[sensitivity_df['Nu_train']==80, 'bound_diff'].max()
yerrb[0][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'bound_diff'].min()
yerrb[1][1]=sensitivity_df.loc[sensitivity_df['Nu_train']==245, 'bound_diff'].max()
yerrb[0][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'bound_diff'].min()
yerrb[1][2]=sensitivity_df.loc[sensitivity_df['Nu_train']==500, 'bound_diff'].max()
yerrb[0][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'bound_diff'].min()
yerrb[1][3]=sensitivity_df.loc[sensitivity_df['Nu_train']==845, 'bound_diff'].max()
yerrb[0][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'bound_diff'].min()
yerrb[1][4]=sensitivity_df.loc[sensitivity_df['Nu_train']==1280, 'bound_diff'].max()
yerrb[0][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'bound_diff'].min()
yerrb[1][5]=sensitivity_df.loc[sensitivity_df['Nu_train']==2000, 'bound_diff'].max()
yerrb[0][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'bound_diff'].min()
yerrb[1][6]=sensitivity_df.loc[sensitivity_df['Nu_train']==3125, 'bound_diff'].max()
yerrb = np.abs(yerrb)

#plt.scatter(sensitivity_df.Nu.values, sensitivity_df.L2_norm_test.values, color='pink', zorder = 0) 
#plt.scatter(sensitivity_df.Nu.values, sensitivity_df.L2_t_test.values, color='lightgreen', zorder = 0)   
#plt.scatter(sensitivity_df.Nu.values, sensitivity_df.E_T.values, color='lightgray', zorder=0)
#plt.scatter(sensitivity_df.Nu.values, sensitivity_df.bound, color='lightsteelblue', zorder=0)

cs1 = CubicSpline(mean_df.Nu.values, mean_df.L2_norm_test.values)
cs2 = CubicSpline(mean_df.Nu.values, mean_df.L2_t_test.values)
cs3 = CubicSpline(mean_df.Nu.values, mean_df.E_T.values)
cs4 = CubicSpline(mean_df.Nu.values, mean_df.bound)
plt.errorbar(mean_df.Nu.values, cs1(mean_df.Nu.values), yerr = yerrL2, capsize = 2, label=r'$\varepsilon_{Total}$', color='red')
plt.errorbar(mean_df.Nu.values, cs2(mean_df.Nu.values), yerr = yerrL2t, capsize = 2, label=r'$\varepsilon_{Total}+\varepsilon_{Total_t}$', color='forestgreen')
plt.errorbar(mean_df.Nu.values, cs3(mean_df.Nu.values), yerr = yerrET, capsize = 2, label=r'$\varepsilon_{Train}$', color='gray')
plt.errorbar(mean_df.Nu.values, cs4(mean_df.Nu.values), yerr = yerrb, capsize = 2, label=r'$Bound$', color='blue')
#plt.scatter(mean_df.Nu.values, mean_df.L2_norm_test.values, label=r'$\varepsilon_{Total}$', color='red')  
#plt.scatter(mean_df.Nu.values, mean_df.L2_t_test.values, label=r'$\varepsilon_{Total}+\varepsilon_{Total_t}$', color='forestgreen') 
#plt.scatter(mean_df.Nu.values, mean_df.E_T.values, label=r'$\varepsilon_{Train}$', color='gray')
#plt.scatter(mean_df.Nu.values, mean_df.bound, label=r'$Bound$', color='blue')

plt.legend()
plt.xscale("log")
plt.xlabel(r'$M$')
plt.yscale("log")
#plt.savefig("Run05_Fin\\Conv2.png", dpi=400)
plt.show()

quit()
