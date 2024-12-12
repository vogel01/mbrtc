from mbrtc import *
from math import sqrt
from icecream import ic
import scipy.signal
import matplotlib.pyplot as plt

w_r2 = 0.803
w_g2 = 55.85
Q = 20

teller = np.array([w_r2, 0])
noemer = np.array([1, (sqrt(w_g2)/Q), w_g2])

# opdracht 1
def opdracht1():
    A, B, C, D = ic(tf2ss(teller, noemer))
    return A, B, C, D

# opdracht 2
def opdracht2():
    A, B, C, D = opdracht1()
    step = ic(scipy.signal.step((A, B, C, D)))
    impulse = ic(scipy.signal.impulse((A, B, C, D)))
    plt.figure("step")
    plt.plot(step[0], step[1])
    plt.figure("impulse")
    plt.plot(impulse[0], impulse[1])
    plt.show()



# opdracht 3
def opdracht3(A):
    polen = ic(np.linalg.eigvals(A))
    for p in polen:
        if np.iscomplex(p):
            reeleDeel = np.real(p)
            imaginaireDeel = np.imag(p)
            freq = imaginaireDeel/(2*np.pi)
            print(f"Complexe pool: {p:.3f}, frequentie = {freq:.3f}, demping = {reeleDeel}")
        else:
            tau = -1/p if p != 0  else np.inf
            print(f"reele pool: {p:.3f}, demping = {tau}")
            
def opdracht5():
    ic.disable()
    A, B, C, D = opdracht1()
    ic.enable()
    h = 0.42
    Ad, Bd, Cd, Dd = ic(c2d_zoh(A, B, C, D, h))
    return Ad, Bd, Cd, Dd

def opdracht6():
    NS = 100
    u_step = step_signal(NS)
    u_impulse = impulse_signal(NS)
    Ad, Bd, Cd, Dd = opdracht5()
    y_stepSim = sim(Ad, Bd, Cd, Dd, u_step)
    y_impulseSim = sim(Ad, Bd, Cd, Dd, u_impulse)
    plt.plot(y_stepSim.T, label="Stap")
    plt.plot(y_impulseSim.T, label = "impulse")
    plt.legend()
    plt.show()

def opdracht7():
    ic.disable()
    A, B, C, D = opdracht1()
    ic.enable()
    opdracht3(A)
    T = np.array([[1, 1],[1, -1]])
    At, Bt, Ct = similarity_trans(A, B, C, T)
    opdracht3(At)
    
def opdracht8():
    A, B, C, D = opdracht1()
    h = 0.42
    u_step = step_signal()
    time = np.linspace(0, 5, 100)
    simresx, simresy = sim_intersample(A, B, C, D, h, 10, u_step, time)
    plt.plot(simresx, simresy)
    plt.show()

if __name__ == "__main__":
    opdracht8()