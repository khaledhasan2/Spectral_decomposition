import numpy as np
import scipy.linalg as la
from sympy.physics.secondquant import Fd, FKet
from itertools import combinations


N = 9
L_1ELECTRON = 3
S_1ELECTRON = 1/2

LIST_OF_STATES = []
for comb in combinations(range(1, 15), N):
    LIST_OF_STATES.append(list(comb))

N_VSPACE = len(LIST_OF_STATES)

INDEX_MAPPING = {}

CHARGING_APPROX = True


for i, state in enumerate(LIST_OF_STATES):
    INDEX_MAPPING[tuple(state)] = i

print("DP lists and mappings are defined")

def matrix_el(state1, state2, i, j):
    s1 = state1.copy()
    s2 = state2.copy()
    res = 1
    # calculate <s1| f†_i f_j |s2>
    if not j in s2: return 0
    if not i in s1: return 0
    ind_i = s1.index(i)
    ind_j = s2.index(j)
    s1.__delitem__(ind_i)
    s2.__delitem__(ind_j)
    if  (ind_j+ind_i)%2 == 1: res*= -1
    for ind_1 in range(len(s1)):
        if s1[ind_1] != s2[ind_1]:
            return 0
    return res


def find_all_eigenstates():

    L_p, L_m, Lz = np.zeros((N_VSPACE, N_VSPACE)), np.zeros((N_VSPACE, N_VSPACE)), np.zeros((N_VSPACE, N_VSPACE))
    S_p, S_m, Sz = np.zeros((N_VSPACE, N_VSPACE)), np.zeros((N_VSPACE, N_VSPACE)), np.zeros((N_VSPACE, N_VSPACE))
    J_p, J_m, J_z = np.zeros((N_VSPACE, N_VSPACE)), np.zeros((N_VSPACE, N_VSPACE)), np.zeros((N_VSPACE, N_VSPACE))
    index_to_ml = lambda index: (index -1)//round(2*S_1ELECTRON+1) - L_1ELECTRON
    index_to_ms = lambda index: (index -1)%round(2*S_1ELECTRON+1) - S_1ELECTRON
    for i, state_1 in enumerate(LIST_OF_STATES):
        for j, state_2 in enumerate(LIST_OF_STATES):
            # https://physics.stackexchange.com/questions/792036/angular-momentum-operator-in-second-quantization
            for k in range(1, 15):
                L_p[i, j] += np.sqrt(L_1ELECTRON*(L_1ELECTRON+1) - index_to_ml(k)*(index_to_ml(k)+1))*matrix_el(state_1, state_2, k+round(2*S_1ELECTRON+1), k)
                L_m[i, j] += np.sqrt(L_1ELECTRON*(L_1ELECTRON+1) - index_to_ml(k)*(index_to_ml(k)-1))*matrix_el(state_1, state_2, k-round(2*S_1ELECTRON+1), k)
                Lz [i, j] += index_to_ml(k)*matrix_el(state_1, state_2, k, k)

                S_p[i, j] += np.sqrt(1.5*0.5 - index_to_ms(k)*(index_to_ms(k)+1))*matrix_el(state_1, state_2, k+1, k)
                S_m[i, j] += np.sqrt(1.5*0.5 - index_to_ms(k)*(index_to_ms(k)-1))*matrix_el(state_1, state_2, k-1, k)
                Sz [i, j] += index_to_ms(k)*matrix_el(state_1, state_2, k, k)
                
    J_p = L_p + S_p
    J_m = L_m + S_m
    J_z = Lz + Sz
    
    L_sq = (L_m@L_p + L_p@L_m)/2 + Lz@Lz
    S_sq = (S_m@S_p + S_p@S_m)/2 + Sz@Sz
    # this a habbatiya way of diagonalizing multiple matrices simultanuously
    eig_vals, eig_vecs = la.eigh(100*L_sq + 0.01*S_sq + (J_m@J_p + J_p@J_m)/2 + J_z@J_z + J_z)
    eig_vecs = np.real(eig_vecs).T
    return eig_vals, eig_vecs


def get_J_state(j, mj, l, s, eig_vecs: np.ndarray =None, eig_vals: np.ndarray=None):
    
    if eig_vals is None:
        print("ED")
        eig_vals, eig_vecs= find_all_eigenstates()
        print("Finished ED")
    exp_eig_value = 100*l*(l+1)+0.01*s*(s+1)+j*(j+1)+mj
    states = eig_vecs[np.isclose(exp_eig_value, eig_vals, rtol=1e-6)]

    if len(states)>1:
        print(eig_vals[np.isclose(exp_eig_value, eig_vals)])
        print(f"{len(states)} J, Mj state were found")
        raise Warning("more than on J, Mj state was found")
    return states[0]

def represent_state_2nd_Q(state, real_inidces=False, ket=True):
        
    q2_rep = 0
    if ket:
        for i, s in enumerate(state):
            if np.isclose(s, 0): continue
            if real_inidces is False: q2_rep += np.real(s)*FKet(LIST_OF_STATES[i])
            else: 
                real_inidces_list = []
                for alp in LIST_OF_STATES[i]:
                    real_inidces_list.append(((alp-1)//round(2*S_1ELECTRON+1) - L_1ELECTRON, (alp-1)%round(2*S_1ELECTRON+1) - S_1ELECTRON))
                q2_rep = q2_rep + s.real*FKet(real_inidces_list)
    else:
        for i, s in enumerate(state):
            if np.isclose(s, 0): continue
            real_inidces_list = []
            prod = 1
            for alp in LIST_OF_STATES[i]:
                if real_inidces: prod = Fd(((alp-1)//round(2*S_1ELECTRON+1) - L_1ELECTRON, (alp-1)%round(2*S_1ELECTRON+1) - S_1ELECTRON))*prod
                else: prod = Fd(alp)*prod
            q2_rep += s.real*prod
    return q2_rep