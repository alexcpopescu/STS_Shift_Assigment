import pandas as pd
import numpy as np
import scipy.optimize as opt

""" ####################################################
    # Adapted from Data 100 Section Assignment Program #
    ####################################################
    
    Author: Alex Popescu
"""

def best_assignment(pref, sec_sizes):
    (n, d) = pref.shape
    w = -pref.flatten()

    # Single assignment constraints
    Aeq = np.zeros((n, n * d))
    for i in range(n):
        Aeq[i, (i * d):((i + 1) * d)] = 1

    # Classroom size constraints
    Aub = np.zeros((d, n * d))
    for i in range(d):
        tmp = np.zeros((n, d))
        tmp[:, i] = 1
        Aub[i, :] = tmp.flatten()

    res = opt.linprog(w, Aub, sec_sizes, Aeq, np.ones(n), bounds=(0, 1), options=dict(maxiter=100000, disp=True))
    return res

def compute_assignments(prefs, cap, opt):
    (n, d) = prefs.shape
    print(np.unique(opt.x))
    w = prefs.flatten()
    unhappy_students = np.sum(w[opt.x == 1.] == 1.0)
    print("Unhappy:", unhappy_students)
    rooms = n * list(range(1, d + 1))
    asgs = np.array(rooms)[opt.x == 1.]
    return asgs

def make_prefs(df):
    return (df.apply(lambda x: x.str.replace("Worst", "0"))
            .apply(lambda x: x.str.replace("Best", "7"))
            .apply(lambda x: x.astype("int")))

def init_asg(df):
    asg = {}
    for i in df['Name']:
        asg[i] = []
    return asg

def create_assignments(moffitt_shifts=2, helpdesk_shifts=1):
    print("#############################\n"+
          "# Starting Shift Assignment #\n"+
          "# Moffitt Shifts: "+str(moffitt_shifts) +"         #\n"+
          "# Helpdesk Shifts: "+str(helpdesk_shifts) +"        #\n"+
          "#############################\n")
    roster = pd.read_csv("Resources/stc_roster.csv")
    emails = roster[roster['Role'] == "STC"]['Email']

    raw = pd.read_csv("Resources/shift_pref.csv",
                      names=["Time", "Email", "Name", "Email2",
                             "Helpdesk Monday 9:00AM", "Helpdesk Monday 10:30AM", "Helpdesk Monday 12:00PM",
                             "Helpdesk Monday 1:30PM", "Helpdesk Monday 3:00PM", "Helpdesk Monday 4:30PM",
                             "Helpdesk Tuesday 9:00AM", "Helpdesk Tuesday 10:30AM", "Helpdesk Tuesday 12:00PM",
                             "Helpdesk Tuesday 1:30PM", "Helpdesk Tuesday 3:00PM", "Helpdesk Tuesday 4:30PM",
                             "Helpdesk Wednesday 9:00AM", "Helpdesk Wednesday 10:30AM", "Helpdesk Wednesday 12:00PM",
                             "Helpdesk Wednesday 1:30PM", "Helpdesk Wednesday 3:00PM", "Helpdesk Wednesday 4:30PM",
                             "Helpdesk Thursday 9:00AM", "Helpdesk Thursday 10:30AM", "Helpdesk Thursday 12:00PM",
                             "Helpdesk Thursday 1:30PM", "Helpdesk Thursday 3:00PM", "Helpdesk Thursday 4:30PM",
                             "Helpdesk Friday 9:00AM", "Helpdesk Friday 10:30AM", "Helpdesk Friday 12:00PM",
                             "Helpdesk Friday 1:30PM", "Helpdesk Friday 3:00PM", "Helpdesk Friday 4:30PM",
                             "Moffitt Monday 9:00AM", "Moffitt Monday 10:00AM", "Moffitt Monday 11:00AM",
                             "Moffitt Monday 12:00PM", "Moffitt Monday 1:00PM", "Moffitt Monday 2:00PM",
                             "Moffitt Monday 3:00PM", "Moffitt Monday 4:00PM",
                             "Moffitt Tuesday 9:00AM", "Moffitt Tuesday 10:00AM", "Moffitt Tuesday 11:00AM",
                             "Moffitt Tuesday 12:00PM", "Moffitt Tuesday 1:00PM", "Moffitt Tuesday 2:00PM",
                             "Moffitt Tuesday 3:00PM", "Moffitt Tuesday 4:00PM",
                             "Moffitt Wednesday 9:00AM", "Moffitt Wednesday 10:00AM", "Moffitt Wednesday 11:00AM",
                             "Moffitt Wednesday 12:00PM", "Moffitt Wednesday 1:00PM", "Moffitt Wednesday 2:00PM",
                             "Moffitt Wednesday 3:00PM", "Moffitt Wednesday 4:00PM",
                             "Moffitt Thursday 9:00AM", "Moffitt Thursday 10:00AM", "Moffitt Thursday 11:00AM",
                             "Moffitt Thursday 12:00PM", "Moffitt Thursday 1:00PM", "Moffitt Thursday 2:00PM",
                             "Moffitt Thursday 3:00PM", "Moffitt Thursday 4:00PM",
                             "Moffitt Friday 9:00AM", "Moffitt Friday 10:00AM", "Moffitt Friday 11:00AM",
                             "Moffitt Friday 12:00PM", "Moffitt Friday 1:00PM", "Moffitt Friday 2:00PM",
                             "Moffitt Friday 3:00PM", "Moffitt Friday 4:00PM"],
                      skiprows=1).groupby('Email').last()

    helpdesk = ["Helpdesk Monday 9:00AM", "Helpdesk Monday 10:30AM", "Helpdesk Monday 12:00PM",
                "Helpdesk Monday 1:30PM", "Helpdesk Monday 3:00PM", "Helpdesk Monday 4:30PM",
                "Helpdesk Tuesday 9:00AM", "Helpdesk Tuesday 10:30AM", "Helpdesk Tuesday 12:00PM",
                "Helpdesk Tuesday 1:30PM", "Helpdesk Tuesday 3:00PM", "Helpdesk Tuesday 4:30PM",
                "Helpdesk Wednesday 9:00AM", "Helpdesk Wednesday 10:30AM", "Helpdesk Wednesday 12:00PM",
                "Helpdesk Wednesday 1:30PM", "Helpdesk Wednesday 3:00PM", "Helpdesk Wednesday 4:30PM",
                "Helpdesk Thursday 9:00AM", "Helpdesk Thursday 10:30AM", "Helpdesk Thursday 12:00PM",
                "Helpdesk Thursday 1:30PM", "Helpdesk Thursday 3:00PM", "Helpdesk Thursday 4:30PM",
                "Helpdesk Friday 9:00AM", "Helpdesk Friday 10:30AM", "Helpdesk Friday 12:00PM",
                "Helpdesk Friday 1:30PM", "Helpdesk Friday 3:00PM", "Helpdesk Friday 4:30PM"]
    moffitt = ["Moffitt Monday 9:00AM", "Moffitt Monday 10:00AM", "Moffitt Monday 11:00AM",
               "Moffitt Monday 12:00PM", "Moffitt Monday 1:00PM", "Moffitt Monday 2:00PM",
               "Moffitt Monday 3:00PM", "Moffitt Monday 4:00PM",
               "Moffitt Tuesday 9:00AM", "Moffitt Tuesday 10:00AM", "Moffitt Tuesday 11:00AM",
               "Moffitt Tuesday 12:00PM", "Moffitt Tuesday 1:00PM", "Moffitt Tuesday 2:00PM",
               "Moffitt Tuesday 3:00PM", "Moffitt Tuesday 4:00PM",
               "Moffitt Wednesday 9:00AM", "Moffitt Wednesday 10:00AM", "Moffitt Wednesday 11:00AM",
               "Moffitt Wednesday 12:00PM", "Moffitt Wednesday 1:00PM", "Moffitt Wednesday 2:00PM",
               "Moffitt Wednesday 3:00PM", "Moffitt Wednesday 4:00PM",
               "Moffitt Thursday 9:00AM", "Moffitt Thursday 10:00AM", "Moffitt Thursday 11:00AM",
               "Moffitt Thursday 12:00PM", "Moffitt Thursday 1:00PM", "Moffitt Thursday 2:00PM",
               "Moffitt Thursday 3:00PM", "Moffitt Thursday 4:00PM",
               "Moffitt Friday 9:00AM", "Moffitt Friday 10:00AM", "Moffitt Friday 11:00AM",
               "Moffitt Friday 12:00PM", "Moffitt Friday 1:00PM", "Moffitt Friday 2:00PM",
               "Moffitt Friday 3:00PM", "Moffitt Friday 4:00PM"]

    helpdesk_pref = make_prefs(raw[helpdesk].astype(str).replace('nan', '-101').replace(
        {'1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4', '5.0': '5', '6.0': '6'})).values + 1
    moffitt_pref = make_prefs(raw[moffitt].astype(str).replace('nan', '-101').replace(
        {'1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4', '5.0': '5', '6.0': '6'})).values + 1

    helpdesk_cap = np.ones(len(helpdesk))
    moffitt_cap = np.repeat(2, len(moffitt))
    combined = np.hstack([helpdesk_cap, moffitt_cap])
    capacity = pd.DataFrame(
        {'Type': np.append(np.array(['Helpdesk'] * len(helpdesk_cap)), np.array(['Moffitt'] * len(moffitt_cap))),
         'Time': np.append(helpdesk, moffitt),
         'Size': combined})

    moffitt_asg_total = []
    helpdesk_asg_total = []
    asg_dict = init_asg(raw)
    name_arr = list(asg_dict.keys())

    print("###############################\n" +
          "# Starting Moffitt Assignment #\n" +
          "###############################\n")

    for i in range(moffitt_shifts):
        moffitt_opt = best_assignment(moffitt_pref, moffitt_cap)
        moffitt_asg = compute_assignments(moffitt_pref, moffitt_cap, moffitt_opt)
        moffitt_asg_total.append(moffitt_asg.tolist())
        print("\n")

        for j in range(len(moffitt_asg)):
            print(name_arr[j] + ": Number " + str(9 - moffitt_pref[j][moffitt_asg[j] - 1]) + " Preference")
            moffitt_pref[j][moffitt_asg[j] - 1] = -100
            capacity.at[29 + moffitt_asg[j], 'Size'] = capacity.at[29 + moffitt_asg[j], 'Size'] - 1
        print("\n")

    moffitt_asg_total = np.array(moffitt_asg_total)

    print("################################\n" +
          "# Starting Helpdesk Assignment #\n" +
          "################################\n")

    for i in range(helpdesk_shifts):
        helpdesk_opt = best_assignment(helpdesk_pref, helpdesk_cap)
        helpdesk_asg = compute_assignments(helpdesk_pref, helpdesk_cap, helpdesk_opt)
        helpdesk_asg_total.append(helpdesk_asg.tolist())
        print("\n")

        for j in range(len(helpdesk_asg)):
            print(name_arr[j] + ": Number " + str(9 - helpdesk_pref[j][helpdesk_asg[j] - 1]) + " Preference")
            helpdesk_pref[j][helpdesk_asg[j] - 1] = -100
            capacity.at[helpdesk_asg[j] - 1, 'Size'] = capacity.at[helpdesk_asg[j] - 1, 'Size'] - 1
        print("\n")

    helpdesk_asg_total = np.array(helpdesk_asg_total)

    for i in range(len(moffitt_asg_total)):
        for j in range(len(moffitt_asg_total[i])):
            asg_dict[name_arr[j]].append(raw.columns[32 + moffitt_asg_total[i][j]])

    for i in range(len(helpdesk_asg_total)):
        for j in range(len(helpdesk_asg_total[i])):
            asg_dict[name_arr[j]].append(raw.columns[2 + helpdesk_asg_total[i][j]])

    print(asg_dict)
    shift_asg = pd.DataFrame(asg_dict)
    shift_asg.to_csv("Resources/Shift_Assignment.csv")
    return asg_dict

create_assignments()
