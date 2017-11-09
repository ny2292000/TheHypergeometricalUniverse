import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import parameters
import pandas as pd
import seaborn as sns
from IPython.display import Image
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*'))

#########################################################
#########################################################
#########################################################



def rxn(C, t, knidecay,tc):
    # Chandrasekhar radius
    # Reaction rates
    tc3=(tc/10)**3
    tc6 = (tc / 10) ** 6
    kmg = 0.05*tc6
    kca = 2*tc3
    kni = 2*tc3
    knidecay=knidecay*tc3
    ##############
    carbon0 = C[0]
    mg0 = C[1]
    ca0 = C[2]
    ox0 = C[3]
    ni0 = C[4]
    # Light = kLight*[Ni] = kLight*dnidt*deltaT
    # assuming constant dnidt and contanst deltaT - this is the coasting velocity approximation
    # dcarbondt=-kmg*[C][C]
    dcarbdt = -kmg * carbon0 ** 2*(t<tc)

    # dMgdt=kmg*[C][C]-kca*[Mg][O]
    dmgdt = kmg * carbon0 ** 2*(t<tc) - kca * mg0 * ox0*(t<tc)

    # dCadt=kca*[Mg]*[O]-kni*[Ca][0]
    dcadt = kca * mg0 * ox0*(t<tc) - kni * ca0 * ox0*(t<tc)

    # doxdt=-kca*mg0*ox0-kni*ca0*ox0
    doxdt = -kca*mg0*ox0*(t<tc) - kni*ca0*ox0*(t<tc)

    # dNidt=kni*[Ca][O]-klight*[Ni]
    # We consider Calcium + Oxigen fusion to take place only until the shockwave reaches surface
    # knidecay refers to the decay Ni->Co->Fe
    # dnidt = kni * ca0 * ox0*(t<=tc) - knidecay * ni0
    dnidt = kni * ca0 * ox0*(t<tc)  - knidecay * ni0
    # Light is created as the shockwave progresses. The accumulated light is supposed
    # to travel with the shockwave. It corresponds to the integral of the light along the radial line
    # is integrate(0,t,dnidt*dt)= dnidt*t
    # For t>tc, the Ni density reaches a limit of dnidt*tc
    return [dcarbdt, dmgdt, dcadt, doxdt, dnidt]


def getLight(Ni, tt, knidecay = 0.8, tc=10 ):
    # Diffusion process with two rates 0.3 for radiation created before the shockwave
    # reaches surface and 0.03 for radiation diffusion across ejecta
    tc=tc/10
    photons = Ni
    return photons

if(__name__=='__main__'):
    logy = False
    xmax = 110
    t = np.linspace(0, xmax, 2000)
    C0 = [0.5, 0.0, 0.0, 0.5, 0.0]
    goodnessOfApproximation = {}
    knidecay=0.3
    tc=10
    argwords = (tc, knidecay)
    cc = pd.DataFrame(odeint(rxn, C0, t,args=(knidecay,tc)), index=t, columns=['C', 'Mg', 'Ca', 'O', 'Ni'])
    cc['Light']=getLight(cc.Ni,tc,knidecay=knidecay, tc=tc)
    cc['Light']=cc['Light']/cc['Light'].max()
    # cc[[0, 1, 2, 3, 4,5]].plot(ax=axes, xlim=[0, xmax],  legend=True, logy=logy)
    # plt.xlabel('Time (days)', fontsize=15, )
    # plt.ylabel('Normalized Absolute Luminosity', fontsize=15, )
    # plt.title('Light/G**(-3)', fontsize=18, )
    # plt.show()
    # # plt.axhline(y=0.94, xmin=0, xmax=1, hold=None)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    fig.subplots_adjust(hspace=.4)
    for i in range(50, 100, 1):
        tc=i/10
        print(tc)
        tc = t[np.argmin(np.abs(t - tc))]
        ccc = pd.DataFrame(odeint(rxn, C0, t, args=(knidecay,tc)), index=t, columns=['C', 'Mg', 'Ca', 'O', 'Ni'])
        ccc['Light'] = ccc.Ni
        ccc['Light'] = ccc['Light'] #/ ccc['Light'].max()
        ccc[[5]].plot(ax=axes,xlim=[0, xmax], legend=False, logy=logy)
        plt.xlabel('Time (days)', fontsize=15, )
        plt.ylabel('Normalized Absolute Luminosity', fontsize=15, )
        # plt.title('Light/G**(-3)', fontsize=18, )
        goodnessOfApproximation[tc] = ccc.Light.sum()
        print(tc, '   ',goodnessOfApproximation[tc])
        a=1
    # plt.show()
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    goodnessOfApproximation=pd.DataFrame.from_dict(goodnessOfApproximation,orient='index')
    goodnessOfApproximation.columns=['goodness']
    goodnessOfApproximation.goodness=goodnessOfApproximation.goodness/goodnessOfApproximation.goodness.max()
    goodnessOfApproximation.plot(logy=logy)
    plt.show()



    a=1
