import xarray as xr
import numpy as np
import pandas as pd
 
from energy_onshore import power_output

class_i = {
        "turbine_model": "Enercon E70",
        "rotor_diameter": 71,
        "rated_power": 2300,
        "hub_height": 85,
        "cut_in_speed": 2.0,
        "rated_speed": 15.5,
        "cut_out_speed": 25.0,
    }
class_ii = {
        "turbine_model": "Gamesa G87",
        "rotor_diameter": 87,
        "rated_power": 2000,
        "hub_height": 83.5,
        "cut_in_speed": 3.0,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }

class_iii = {
       "turbine_model": "Vestas V110",
       "rotor_diameter": 110,
       "rated_power": 2000,
       "hub_height": 100,
       "cut_in_speed": 4.0,
       "rated_speed": 12.0,
       "cut_out_speed": 20.0,
    }
class_s = {
        "turbine_model": "Vestas V164",
        "rotor_diameter": 164,
        "rated_power": 9500,
        "hub_height": 105,
        "cut_in_speed": 3.5,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }


#iec_class:"I"
power_output(np.linspace(0,50,30), iec_class="I",rated_power=class_i['rated_power'],cut_in_speed=class_i["cut_in_speed"],rated_speed=class_i["rated_speed"],cut_out_speed=class_i["cut_out_speed"])

#iec_class:"II
power_output(np.linspace(0,50,30), iec_class="II",rated_power=class_ii['rated_power'],cut_in_speed=class_ii["cut_in_speed"],rated_speed=class_ii["rated_speed"],cut_out_speed=class_ii["cut_out_speed"])

#iec_class:"III"
power_output(np.linspace(0,50,30), iec_class="III",rated_power=class_iii['rated_power'],cut_in_speed=class_iii["cut_in_speed"],rated_speed=class_iii["rated_speed"],cut_out_speed=class_iii["cut_out_speed"])

#iec_class:"S"
power_output(np.linspace(0,50,30), iec_class="S",rated_power=class_s['rated_power'],cut_in_speed=class_s["cut_in_speed"],rated_speed=class_s["rated_speed"],cut_out_speed=class_s["cut_out_speed"])

power_output(np.linspace(0,50,30), iec_class="I",rated_power=class_i['rated_power'],cut_in_speed=class_i["cut_in_speed"],rated_speed=class_i["rated_speed"],cut_out_speed=class_i["cut_out_speed"])

#alfa_opt: 0.09448412217119548; k_opt: 4.114120346123451; iec_class: I
#alfa_opt: 0.10679470725755313; k_opt: 4.358850779176722; iec_class: II
#alfa_opt: 0.12383571301799268; k_opt: 5.3922433841254716; iec_class: III
#alfa_opt: 0.10035196564753286; k_opt: 4.182870066171126; iec_class: S
#alfa_opt: 0.09448412217119548; k_opt: 4.114120346123451; iec_class: I

    # Define a parametrized Weibull Cumulative Distribution Function to fit the power curve.
def weibull_distribution(x, alfa, k):
        f = rated_power - rated_power * np.exp(-((x * alfa) ** k))
        return f

#I
alfa_delta=0.005
delta_k=0.1

rated_power=class_i['rated_power']
cut_in_speed=class_i["cut_in_speed"]
rated_speed=class_i["rated_speed"]
cut_out_speed=class_i["cut_out_speed"]

#I_delta_plus=weibull_distribution(np.linspace(0,50,30), alfa=0.09448412217119548+alfa_delta, k=4.114120346123451)

#I_delta_minus=weibull_distribution(np.linspace(0,50,30), alfa=0.09448412217119548-alfa_delta, k=4.114120346123451)

#I_k_plus=weibull_distribution(np.linspace(0,50,30), alfa=0.09448412217119548, k=4.114120346123451+delta_k)

#I_k_minus=weibull_distribution(np.linspace(0,50,30), alfa=0.09448412217119548, k=4.114120346123451-delta_k)

ws=np.linspace(0,30,50)

conditions = [
    (ws < cut_in_speed),
    (ws >= cut_in_speed) & (ws <= rated_speed),
    (ws > rated_speed) & (ws <= cut_out_speed),
    (ws > cut_out_speed),
]
functions_I_d_p = [
    0,
    lambda x: weibull_distribution(x, alfa=0.09448412217119548+alfa_delta, k=4.114120346123451),
    rated_power,
    0,
]

functions_I_d_m = [
    0,
    lambda x: weibull_distribution(x, alfa=0.09448412217119548-alfa_delta, k=4.114120346123451),
    rated_power,
    0,
]

functions_I_k_p = [
    0,
    lambda x: weibull_distribution(x, alfa=0.09448412217119548, k=4.114120346123451+delta_k),
    rated_power,
    0,
]

functions_I_k_m = [
    0,
    lambda x: weibull_distribution(x, alfa=0.09448412217119548, k=4.114120346123451-delta_k),
    rated_power,
    0,
]

# A piecewise function is used to obtain the power curve.
I_delta_plus = np.piecewise(ws, conditions, functions_I_d_p)
I_delta_minus = np.piecewise(ws, conditions, functions_I_d_m)
I_k_plus = np.piecewise(ws, conditions, functions_I_k_p)
I_k_minus = np.piecewise(ws, conditions, functions_I_k_m)


import matplotlib.pyplot as plt

plt.plot(ws,I_delta_minus, label='I delta minus')
plt.plot(ws,I_delta_plus, label='I delta plus')
plt.plot(ws,I_k_minus, label='I k minus')
plt.plot(ws,I_k_plus, label='I k plus')
plt.legend()
plt.title('I Class Wind Turbine Power Curve Perturbation')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power Output (Kw)')
plt.grid()
plt.show()

# change cut_out_speed
delta_cut_out_speed = 1.

rated_power=class_i['rated_power']
cut_in_speed=class_i["cut_in_speed"]
rated_speed=class_i["rated_speed"]
cut_out_speed=class_i["cut_out_speed"]

condition1 = [
    (ws < cut_in_speed),
    (ws >= cut_in_speed) & (ws <= rated_speed),
    (ws > rated_speed) & (ws <= cut_out_speed-delta_cut_out_speed),
    (ws > (cut_out_speed-delta_cut_out_speed)),
]

condition2 = [
    (ws < cut_in_speed),
    (ws >= cut_in_speed) & (ws <= rated_speed),
    (ws > rated_speed) & (ws <= cut_out_speed-delta_cut_out_speed*2),
    (ws > (cut_out_speed-delta_cut_out_speed*2)),
]

functions_I_cut = [
    0,
    lambda x: weibull_distribution(x, alfa=0.09448412217119548+alfa_delta, k=4.114120346123451),
    rated_power,
    0,
]

I_cut_out_m = np.piecewise(ws, condition1, functions_I_cut)
I_cut_out_mm = np.piecewise(ws, condition2, functions_I_cut)

plt.plot(ws,I_cut_out_m, label='I cut out - delta')
plt.plot(ws,I_cut_out_mm, label='I cut out - 2*delta')
plt.legend()
plt.title('I Class Wind Turbine Power Curve Perturbation - cut out speed')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power Output (Kw)')
plt.grid()
plt.show()
