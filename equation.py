# Important packages
import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import brentq
import sys, os

# Variable notation
# x g c C D
# 0 1 2 3 4
# d m G
# 0 1 2

# Optimized function to impliment single step update of 
# ternary onsite equation update

@nb.jit(nopython=True, nogil=True)
def free_d(d, m, ki, qi, gi, Dit, kj, qj, gj, Djt, dt):

    return (d + d*m*ki*gi + d*kj*gj + d*(
            m*ki*qj*gi*Dit/(1+qi*ki*gi*d*m) + 
            kj*qj*gj*Djt/(1+qj*kj*gj*d)) - dt)

@nb.jit(nopython=True, nogil=True)
def free_m(m, d, ki, qi, gi, Dit, kj, qj, gj, Djt, mt):

    return (m + d*m*ki*gi + d*m*(
            ki*qi*gi*Dit/(1+qi*ki*gi*d*m)) - mt)


@nb.jit(nopython=True, nogil=True)
def free_m2(m, d, ki, qi, gi, Dit, kj, qj, gj, Djt, mt):

    return (m + d*m*(ki*gi+kj*gj) + d*m*(
            ki*qi*gi*Dit/(1+qi*ki*gi*d*m) + 
            kj*qj*gj*Djt/(1+qj*kj*gj*d*m)) - mt)

def ter_eqns(i_arr, j_arr, sys_arr, p, dt):

    # temporary variables
    x_i = i_arr[0]
    g_i = i_arr[1]
    c_i = i_arr[2]
    C_i = i_arr[3]
    D_i = i_arr[4]
    x_j = j_arr[0]
    g_j = j_arr[1]
    c_j = j_arr[2]
    C_j = j_arr[3]
    D_j = j_arr[4]
    d = sys_arr[0]
    m = sys_arr[1]
    G = sys_arr[2]

    # non linear solver to find free dCas9 and free MCP-SoxS concerntration
    sys_arr[0] = brentq(f=free_d, a=0, b=p["d_t"], args=(sys_arr[1], p["k_i"], 
                                    p["q_i"], g_i, p["Di_t"], 
                                    p["k_j"], p["q_j"], 
                                    g_j, p["Dj_t"], p["d_t"]))                                
    sys_arr[1] = brentq(f=free_m, a=0, b=p["m_t"], args=(sys_arr[0], p["k_i"], 
                                    p["q_i"], g_i, p["Di_t"], 
                                    p["k_j"], p["q_j"], 
                                    g_j, p["Dj_t"], p["m_t"]))                                                            

    # Update concentrations
    i_arr[0] = x_i
    j_arr[0] = x_j
    i_arr[1] += (p["ui"] + p["gamma_i"]*x_i - p["dg_i"]*g_i)*dt
    j_arr[1] += (p["uj"] + p["gamma_j"]*x_j - p["dg_j"]*g_j)*dt
    i_arr[2] = p["k_i"]*g_i*d*m
    j_arr[2] = p["k_j"]*g_j*d
    i_arr[3] = p["q_i"]*p["k_i"]*g_i*d*m*D_i
    j_arr[3] = p["q_j"]*p["k_j"]*g_j*d*D_j 
    i_arr[4] = p["Di_t"]/(1+p["q_i"]*p["k_i"]*g_i*d*m)
    j_arr[4] = p["Dj_t"]/(1+p["q_j"]*p["k_j"]*g_j*d)
    sys_arr[2] += (p["K_i"]*C_i*D_j/p["Dj_t"] + p["K_j"]*D_j*D_i/p["Dj_t"] - p["dG"]*G)*dt
    #sys_arr[2] += (p["K_i"]*C_i + p["K_j"]*D_j - p["dG"]*G)*dt
    return i_arr, j_arr, sys_arr




def quat_eqns(i_arr, j_arr, sys_arr, p, dt):

    # temporary variables
    x_i = i_arr[0]
    g_i = i_arr[1]
    c_i = i_arr[2]
    C_i = i_arr[3]
    D_i = i_arr[4]
    x_j = j_arr[0]
    g_j = j_arr[1]
    c_j = j_arr[2]
    C_j = j_arr[3]
    D_j = j_arr[4]
    d = sys_arr[0]
    m = sys_arr[1]
    G = sys_arr[2]

    # non linear solver to find free dCas9 and free MCP-SoxS concerntration
    sys_arr[0] = brentq(f=free_d, a=0, b=p["d_t"], args=(sys_arr[1], p["k_i"], 
                                    p["q_i"], g_i, p["Di_t"], 
                                    p["k_j"], p["q_j"], 
                                    g_j, p["Dj_t"], p["d_t"]))                                
    sys_arr[1] = brentq(f=free_m2, a=0, b=p["m_t"], args=(sys_arr[0], p["k_i"], 
                                    p["q_i"], g_i, p["Di_t"], 
                                    p["k_j"], p["q_j"], 
                                    g_j, p["Dj_t"], p["m_t"]))                                                            

    # Update concentrations
    i_arr[0] = x_i
    j_arr[0] = x_j
    i_arr[1] += (p["ui"] + p["gamma_i"]*x_i - p["dg_i"]*g_i)*dt
    j_arr[1] += (p["uj"] + p["gamma_j"]*x_j - p["dg_j"]*g_j)*dt
    i_arr[2] = p["k_i"]*g_i*d*m
    j_arr[2] = p["k_j"]*g_j*d*m
    i_arr[3] = p["q_i"]*p["k_i"]*g_i*d*m*D_i
    j_arr[3] = p["q_j"]*p["k_j"]*g_j*d*m*D_j 
    i_arr[4] = p["Di_t"]/(1+p["q_i"]*p["k_i"]*g_i*d*m)
    j_arr[4] = p["Dj_t"]/(1+p["q_j"]*p["k_j"]*g_j*d*m)
    sys_arr[2] += (p["K_i"]*C_i*D_j/p["Dj_t"] + p["K_j"]*D_j*D_i/p["Dj_t"] - p["dG"]*G)*dt
    #sys_arr[2] += (p["K_i"]*C_i + p["K_j"]*D_j - p["dG"]*G)*dt
    return i_arr, j_arr, sys_arr