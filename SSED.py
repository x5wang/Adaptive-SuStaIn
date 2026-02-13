# !pip install -q git+https://github.com/ucl-pond/pySuStaIn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pySuStaIn
import os
import pickle
import shutil
from pathlib import Path
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score

import AbstractSustain as abstract
import ZscoreSustain_sigma as zscore

#data = np.loadtxt("ADNIA4_sym", delimiter=",")
data = np.loadtxt("A4_5.csv", delimiter=",") - 1

adni = data[:159]
a4_data = data[159:]

#plt.plot(adni[:,0],adni[:,1],'x')

#plt.plot(a4_data[:,0],a4_data[:,1],'x')

a4_sc = a4_data * np.mean(adni, 0)/np.mean(a4_data, 0)

def Create_Input(input_data, N_S_max, of):

  M = input_data.shape[0]
  N = input_data.shape[1]

  Z_max = np.max(input_data,0)
  Z_vals = np.linspace(np.zeros(N), Z_max, 4).T[:,1:3]
  #Z_vals = np.linspace(np.zeros(N), Z_max, 3).T[:,1:2]
  SuStaInLabels = ['temporal','frontal','parietal','occipital','mtl']
  N_startpoints = 10
  N_iterations_MCMC = 1
  output_folder = os.path.join(os.getcwd(), of)
  sustain_input = zscore.ZscoreSustain(input_data,
                              Z_vals,
                              Z_max,
                              SuStaInLabels,
                              N_startpoints,
                              N_S_max,
                              N_iterations_MCMC,
                              output_folder,
                              of,
                              False,
                              .1)
  return sustain_input

def Run_SuStaIn(sustain_input,output_folder):

  if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
  if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

  samples_sequence,   \
  samples_f,          \
  ml_subtype,         \
  prob_ml_subtype,    \
  ml_stage,           \
  prob_ml_stage,      \
  prob_subtype_stage  = sustain_input.run_sustain_algorithm()

  return samples_sequence, samples_f, ml_subtype

def Run_Separate(new_sub, which_data, data_name, iter):

  ls_d = []
  ls_in = []
  ls_s_d = []

  num_sub = len(np.unique(new_sub))
  for i in range(num_sub):
      d = which_data[new_sub == i]
      ls_d.append(d)
      in_temp = Create_Input(d, 1, data_name + str(i) + str(iter))
      ls_in.append(in_temp)
      s_d_temp,_,_ = Run_SuStaIn(in_temp, data_name + str(i) + str(iter))
      ls_s_d.append(s_d_temp)

  return ls_in, ls_s_d

def Eval(ls_in, ls_s_d, data_eval, f_d_norm):

  ls_d_lh = []
  num_sub = len(ls_in)

  for i in range(num_sub):
      p_perm = ls_in[i]._calculate_likelihood_stage(pySuStaIn.ZScoreSustainData(data_eval, 15), ls_s_d[i].squeeze())
      d_lh_t = np.sum(p_perm,1)
      ls_d_lh.append(d_lh_t)

  d_lh = ls_d_lh[0]
  for i in range(num_sub - 1):
      d_lh = np.vstack((d_lh, ls_d_lh[i+1]))

  d_lh = d_lh.T
  d_lh_temp = d_lh * f_d_norm
  d_lh_subj = np.sum(d_lh_temp,1)
  total_lh = np.sum(np.log(d_lh_subj + 1e-250))

  d_lh_final = d_lh_temp / np.tile(np.sum(d_lh_temp, 1), (num_sub,1)).T
  f_d_norm = np.mean(d_lh_final, 0)
  sub_d_pred = np.argmax(d_lh_final,1)

  return sub_d_pred, f_d_norm, total_lh

def Run_SE_Recurse_Stage(sub, data, data_name, iter, score, f):

  in_d, s_d = Run_Separate(sub, data, data_name, iter)
  sub1new, f, score_new = Eval(in_d, s_d, data, f)

  print('proportions: ', f)
  print('likelihood: ', score_new)

  if abs(score_new - score) <= 1:
      return in_d, s_d, sub1new

  return Run_SE_Recurse_Stage(sub1new, data, data_name, iter+1, score_new, f)

def Run_SE_Recurse(sub1, sub2, data1, data2, data1_name, data2_name, iter, score):

  in_d, s_d = Run_Separate(sub1, data1, data1_name, iter)
  # Note: The Eval call below seems to use variables (in1, sd1, etc.) that are not defined in this scope 
  # in the original notebook. I have kept it as is, but you may need to fix the variable names.
  sub1new = Eval(in1, sd1, in2, sd2, in3, sd3, data1) 
  sub2pred = Eval(in1, sd1, in2, sd2, in3, sd3, data2)
  sub2new = Run_Sep_Eval(sub2, data2, data2_name, iter, data2)

  score_new = adjusted_rand_score(sub2pred, sub2new)
  print(score_new)

  if abs(score_new - score) <= 0.01:
      return

  return Run_SE_Recurse(sub1new, sub2new, data1, data2, data1_name, data2_name, iter+1, score_new)

def Run_K_Proj(data1, data2, data1_name, data2_name, k):

  in_d1 = Create_Input(data1, k, data1_name)
  s_d1, f_d1, sub_d1 = Run_SuStaIn(in_d1, data1_name)
  pred = in_d1.subtype_and_stage_individuals_newData(data2,s_d1,f_d1,data2.shape[0])[0]

  in_d2 = Create_Input(data2, k, data2_name)
  _, _, sub_d2 = Run_SuStaIn(in_d2, data2_name)

  sub_d1 = sub_d1.squeeze()
  sub_d2 = sub_d2.squeeze()

  score = adjusted_rand_score(pred.squeeze(), sub_d2)
  print(score)

  Run_SE_Recurse(sub_d1, sub_d2, data1, data2, data1_name, data2_name, 0, score)

in_adni = Create_Input(adni, 4, 'output_adni')
seq_adni, sam_f_adni, sub_adni = Run_SuStaIn(in_adni,'output_adni')
