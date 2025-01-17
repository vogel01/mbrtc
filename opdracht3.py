import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sgnl
from scipy.optimize import minimize
from mbrtc import *
import os
from icecream import ic

## Stap 1: Systeem identificatie: stepper motor naar encoder signaal
#
# Het systeem wordt eerst geschat door een blokvorming signaal op de stappen
# motor te zetten en de encoder te meten. Zowel het ingangs- als uitgangssignaal
# worden gebruikt voor het schatten van het systeem, dit heet systeem identificatie.
# - Zet evt regelaars van de slingeropstelling uit
# - Zet control_add aan
# - Stel het aantal batches voor logging op 10 en activeer "Log Data". De plotter
#   zal even stoppen, zolang er data gelogged wordt.
# - Kijk in de folder waarin je werkt, en merk op dat er een bestand is aangemaakt
#   log_data_YYYYMMDD-HHMMSS.pickle, waarbij YYYYMMDD de datum en HHMMSS de tijd.
# - Pas de onderstaande code aan, om het juiste data log bestand uit te lezen.
# Opmerking: Deze code kun je runnen in de Python prompt (linker prompt) van het dashboard
# maar voor het ontwikkelen en testen van python code is een python IDE, zoals vscode of
# spyder handiger. Advies: Run deze code dus eerst in vscode of spyder.

Path = r"C:\Users\arend\OneDrive - De Haagse Hogeschool\school\Jaar 4\Blok 2\Model based real-time control\Practicum\Edukit_Micorpython\edukit-micropython"
fname = 'log_data_20241219-145150.pickle'
FilePath = os.path.join(Path, fname)
with open(FilePath,'rb') as f:
    data = pickle.load(f)

# selecteer de juiste signalen uit het logbestand
# u = ingang = stepper frequency, y = uitgang = encoder
y = data[:,1]
u = data[:,2]


N = len(u)           # number of samples
Nhalf = N//2         # half the number of samples
h = 0.01             # sampling time 
td = h*np.arange(N)  # discrete time instants

# deze functie genereerd een discrete-tijd state space model op basis van
# een aantal tuning parameters waarmee de fysische constanten van het systeem
# worden aangepast zodat ze beter passen bij de gemeten data. De tuning parameters
# worden opgeslagen in de vector x.
def param2ss(x,h):
    # helper function to get the state-space matrices out of the
    # optimization vector x (h is the sampling time)
    a,b,c,d,e=x # model and initial state parameters to optimize
    A = np.array([[0.,1.],[a*(-53), b*(-0.118)]])
    B = np.array([[0.],[c*(-3.0)]])
    C = np.array([[0.,1.]])
    D = np.array([[0.]])
    Ad,Bd,Cd,Dd = c2d_zoh(A,B,C,D,h)
    x0 = np.array([d,e])
    return Ad,Bd,Cd,Dd,x0

# Deze functie voert eerst param2ss uit op basis van een vector x, en
# berekend dan de gesimuleerde uitgang en vergelijkt die met de gemeten uitgang.
# Van het verschil tussen gemeten en berekende uitgang wordt een getal berekend
# dat 0 is als beide signalen hetzelfde zijn en steeds groter naarmate het vershil
# groter is. Deze functie wordt aangeroepen door een optimalisatie routine minimize
# hieronder.
def func(x,u,y,h):
    # helper function for the optimization, given the
    # parameter vector x, the input-signal u, output-signal y
    # and the sampling time h, the error between the measured output
    # and the model-based simulated output is to be minimized
    Ad,Bd,Cd,Dd,x0 = param2ss(x,h)
    ye = sim(Ad,Bd,Cd,Dd,u,x0)  # simulated output
    return np.linalg.norm(y-ye) # cost-value to be minimized

# Hier wordt het echter rekenwerk gedaan
x_init = np.array([1.,1.,1.,0.,0.])
other_func_args = (u[0:Nhalf],y[0:Nhalf],h)
result = minimize(func,x_init,other_func_args)
Ad,Bd,Cd,Dd,x0 = param2ss(result['x'],h)

# Je hebt nu een discreet model (Ad,Bd,Cd,Dd),
# en een schatting van de begin toestand x0.

# Opdracht: Simuleer het systeem met u als ingang, en
#           noem het signaal ye, en plot zowel y als ye.

ye = sim(Ad, Bd, Cd, Dd, u, x0)

plt.plot(y)
plt.plot(ye)
plt.legend('Gemeten encoder','Gesimuleerde encoder')
# plt.show()

# Vraag: Hoe goed komen y en ye met elkaar overeen?
# redelijk, niet perfect want de wereld is niet perfect..

## Stap 2: Bepaal de statefeedback gain L waarmee de gesloten lus polen op 0.8 komen
#         te liggen. Gebruik hiervoor de functie place in mbrtc.py.

L = place(Ad, Bd, [0.8, 0.8])

## Stap 3: Controlleer of de gesloten lus polen daadwerkelijk op 0.8 liggen.
#         (We kiezen 0.8 voor de polen en niet kleiner, zodat de regelaar
#          niet te agressief is, wat leidt tot prestatieverlies of instabiliteit.
#          Als je tijd hebt, kun je evt later kleinere waarden uitproberen, maar
#          doe dat voorzichting, bijv. eerst 0.7 etc.)
Ad_closed = Ad - Bd @ L
poles = np.linalg.eigvals(Ad_closed)
print('De gesloten-lus polen van de state-feedback regeleaar liggen op: ',poles)

## Stap 4: Bepaal de state-observer gain waarvoor observer polen op 0.8 liggen met place
#         en controlleer de observer polen, 
K = place(Ad.T,Cd.T,[0.8,0.8]).T
Ad_observer = Ad - K @ Cd
poles_observer = np.linalg.eigvals(Ad_observer)
print('De gesloten-lus polen van de state observer liggen op: ',poles_observer)


## Stap 5: Maak de state-space matrices van de regelaar met
#         (theorie, niet voor practicum: waarom zo?):
Actrl = Ad-K@Cd-Bd@L
Bctrl = K
Cctrl = -L
Dctrl = np.zeros((1,1))

## Stap 6: Check of je edukit de juist firmware heeft, door
#         dir(ss)
#         uit te voeren (ss is het object voor de state-space)
#         controller. Krijg je een foutmelding dat ss niet bestaat,
#         meldt je dan bij de docent, om de juiste code te laten flashen.
#         Zorg er ook voor dat je de meest up to date versie van de file
#         textual_mpy_edukit.py hebt. Doe evt een git pull, of zie
#         https://github.com/prfraanje/edukit-micropython/blob/main/textual_mpy_edukit.py

## Stap 7: Kopieer de waarden van Actrl, Bctrl, Cctrl naar het edukit ss object.
#         De structuur van A, B, C in ss is alsvolgt (dit zijn dus geen numpy arrays
#         maar lijsten):
#         ss.A = [ [rij 1], [rij 2]]
#         ss.B = [ waarde 1, waarde 2]
#         ss.C = [ waarde 1, waarde 2]
#         Om de waarden handig te kopieren, kun je de volgende code gebruiken in python:

def print_ss(A,B,C):
    print(f'ss.A = [[{A[0,0]},{A[0,1]}],[{A[1,0]},{A[1,1]}]]')
    print(f'ss.B = [{B[0,0]},{B[1,0]}]')
    print(f'ss.C = [{C[0,0]},{C[0,1]}]')

print_ss(Actrl,Bctrl,Cctrl)
    
# Kopieer dit resultaat in de Micropython prompt.

# Kies voor de State-space regelaar en zet de run.ss flag aan.

# Als het goed is, zal de regelaar nog niet veel doen. Dit komt
# omdat de uitgang van de regelaar wordt vermenigvuldigd met een gain
# (dit is een veiligheidsfactor). Deze gain kun je geleidelijk verhogen
# van 0 tot 1, bijv. in stappen van 0.25 met
# ss.gain += 0.25

## Stap 8: Test of de slinger dempt, door een licht tikje te geven aan de slinger,
# met en zonder regelaar.

## Stap 9: Reflectievragen:
#   - Is de demping van deze regelaar beter dan die je met PID hebt ontworpen?
#   - Is het ontwerp van deze regelaar beter dan het PID ontwerp?
#   - Welke methode kies jij bij je volgende regelaar ontwerp taak?

## Stap 10: Demonstreer je regeling aan de docent, ter beoordeling.

# *** The END! ***

# Toegift: Er zit nog een pid regelaar verstopt in het state-space regelaar object
# waarmee je de rotor stand kunt regelen, zie hiervoor de StateSpace class in ucontrol.py.
# Bestudeer de code, en stel de pid parameters in, gebruik hiervoor ss.set_pid(),
# geef een referentiewaarde op met ss.r1 en zet de pid regelaar aan met ss.run_pid = True
# Natuurlijk zou je ook hier weer een state-space regelaar kunnen ontwerpen, maar
# dan moet je wel eerst een goed state-space model maken van de overbrenging van
# de stappen motor naar de rotor, waar dan de dempende regelaar in zit. Dit kan,
# maar voert voor nu te ver in dit practicum.
