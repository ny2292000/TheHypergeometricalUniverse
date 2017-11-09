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

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
fig.subplots_adjust(hspace=.4)


def rxn(C, t, knidecay,tc):
    # Chandrasekhar radius
    # Reaction rates
    kmg = 0.05
    kca = 1
    kni = 5
    ##############
    carbon0 = C[0]
    mg0 = C[1]
    ca0 = C[2]
    ox0 = C[3]
    ni0 = C[4]
    # Light = kLight*[Ni] = kLight*dnidt*deltaT
    # assuming constant dnidt and contanst deltaT - this is the coasting velocity approximation
    # dcarbondt=-kmg*[C][C]
    dcarbdt = -kmg * carbon0 ** 2

    # dMgdt=kmg*[C][C]-kca*[Mg][O]
    dmgdt = kmg * carbon0 ** 2 - kca * mg0 * ox0

    # dCadt=kca*[Mg]*[O]-kni*[Ca][0]
    dcadt = kca * mg0 * ox0 - kni * ca0 * ox0

    # doxdt=-kca*mg0*ox0-kni*ca0*ox0
    doxdt = -kca * mg0 * ox0 - kni * ca0 * ox0

    # dNidt=kni*[Ca][O]-klight*[Ni]
    # We consider Calcium + Oxigen fusion to take place only until the shockwave reaches surface
    # knidecay refers to the decay Ni->Co->Fe
    dnidt = kni * ca0 * ox0 - knidecay * ni0
    # Light is created as the shockwave progresses. The accumulated light is supposed
    # to travel with the shockwave. It corresponds to the integral of the light along the radial line
    # is integrate(0,t,dnidt*dt)= dnidt*t
    # For t>tc, the Ni density reaches a limit of dnidt*tc
    return [dcarbdt, dmgdt, dcadt, doxdt, dnidt]


def getLight(Ni, tc, knidecay = 0.8, kdiff = 0.1 ):
    # Diffusion process with two rates 0.3 for radiation created before the shockwave
    # reaches surface and 0.03 for radiation diffusion across ejecta
    t = Ni.index
    deepPhotons = {}
    tc = t[np.argmin(np.abs(t - tc))]
    tlow=t[t <= tc]
    thigh = t[t > tc]
    for i in range(len(tlow)):
        # trev plays the role of time but also distance propagagate by the shockwave. Only shells smaller that the max trev have been detonated
        trev = tlow[:i]
        # diffusion plays the role of diffusion to the surface. Photons created at the core take time to reach the surface
        # larger numbers should appear for outermost shells
        diffusion = np.exp(-kdiff * (tc - trev))
        diffusion=diffusion[::-1]
        #reverse the quantity of Nickel produced
        # inner shells are ahead of Nickel production due to more time cooking
        nic=Ni.values[:i]
        nic=nic[::-1]
        # The number of produced photons goes with the volume of the shell (Area times time step * velocity).
        # We make time step and velocity equal to 1 and disregard the 4Pi, keeping just the radius squared
        # kNidecay is not necessary but it is placed here just to remind us that we are using Luminosity=kNidecay*Ni (integrated over the volume and subjected to diffusion).
        deepPhotons[i] = sum(knidecay*nic * trev ** 2*diffusion)

    j=1
    for tt in thigh:
        i=np.where(t==tt)
        i=i[0]
        i=i[0]
        #reverse the quantity of Nickel produced
        # inner shells are ahead of Nickel production due to more time cooking
        nic=Ni.values[j:i]
        j=j+1
        nic=nic[::-1]
        # The number of produced photons goes with the volume of the shell (Area times time step * velocity).
        # We make time step and velocity equal to 1 and disregard the 4Pi, keeping just the radius squared
        # kNidecay is not necessary but it is placed here just to remind us that we are using Luminosity=kNidecay*Ni (integrated over the volume and subjected to diffusion).
        deepPhotons[i] = sum(knidecay*nic * trev ** 2*diffusion)
    photons = pd.DataFrame.from_dict(deepPhotons, orient='index')
    photons.index=t


    return photons

if(__name__=='__main__'):
    logy = False
    xmax = 100
    t = np.linspace(0, xmax, 200)
    C0 = [0.5, 0.0, 0.0, 0.5, 0.0]
    goodnessOfApproximation = {}
    knidecay=0.01
    kdiff=0.03
    tc=10
    argwords = (tc, knidecay)
    cc = pd.DataFrame(odeint(rxn, C0, t,args=(knidecay,tc)), index=t, columns=['C', 'Mg', 'Ca', 'O', 'Ni'])
    cc['Light']=getLight(cc.Ni,tc,knidecay=knidecay,kdiff=kdiff)
    cc['Light']=cc['Light']/cc['Light'].max()
    cc[[0, 1, 2, 3, 4,5]].plot(ax=axes, xlim=[0, xmax], ylim=[0, 1.1], legend=True, logy=logy)
    plt.xlabel('Time (days)', fontsize=15, )
    plt.ylabel('Normalized Absolute Luminosity', fontsize=15, )
    plt.title('Light/G**(-3)', fontsize=18, )
    # plt.axhline(y=0.94, xmin=0, xmax=1, hold=None)

    for tc in range(10, 5, -1):
        tc = t[np.argmin(np.abs(t - tc))]
        ccc = pd.DataFrame(odeint(rxn, C0, t, args=(knidecay,tc)), index=t, columns=['C', 'Mg', 'Ca', 'O', 'Ni'])
        goodnessOfApproximation[tc] = getLight(ccc.Ni, tc, knidecay=knidecay, kdiff=kdiff).loc[tc] / (tc / 10.0) ** (5.0)
        # goodnessOfApproximation[tc] = getLight(ccc.Ni,tc,knidecay=knidecay,kdiff=kdiff).sum() / (tc / 10.0) ** (6.0)

    goodnessOfApproximation=pd.DataFrame.from_dict(goodnessOfApproximation,orient='index')
    goodnessOfApproximation.columns=['goodness']
    goodnessOfApproximation.goodness=goodnessOfApproximation.goodness/goodnessOfApproximation.goodness.max()
    goodnessOfApproximation.plot(ax=axes, logy=logy)
    plt.show()


    a=1
