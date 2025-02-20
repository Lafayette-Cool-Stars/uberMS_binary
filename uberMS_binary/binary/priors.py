from .advancedpriors import IMF_Prior,Gal_Prior

import numpyro
import numpyro.distributions as distfn
import jax.numpy as jnp

def defaultprior(parname):
    # define defaults for sampled parameters
    if "EEP" in parname:
        return numpyro.sample(parname, distfn.Uniform(300,800))
    if "initial_Mass" in parname:
        return numpyro.sample(parname, IMF_Prior())
    if "initial_[Fe/H]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-3.5,0.49))
    if "initial_[a/Fe]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-0.19,0.59))
    if "vmic" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.5, 3.0))
    if "vstar" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.0, 25.0))
    if "Teff" in parname:
        return numpyro.sample(parname, distfn.Uniform(2500.0, 10000.0))
    if "log(g)" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.0, 5.5))
    if "[Fe/H]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-3.5,0.49))
    if "[a/Fe]" in parname:
        return numpyro.sample(parname, distfn.Uniform(-0.19,0.59))        
    if ("vrad" in parname) and (parname != "vrad_b"):
        return numpyro.sample(parname, distfn.Uniform(-500.0, 500.0))
    if "pc0" in parname:
        return numpyro.sample(parname, distfn.Uniform(0.5, 2.0))
    if "pc1" in parname:
        return numpyro.sample(parname, distfn.Normal(0.0, 0.25))
    if "pc2" in parname:
        return numpyro.sample(parname, distfn.Normal(0.0, 0.25))
    if "pc3" in parname:
        return numpyro.sample(parname, distfn.Normal(0.0, 0.25))
    if "lsf" in parname:
        return numpyro.sample(parname, distfn.Normal(32000.0,1000.0))
    if "specjitter" in parname:
        return numpyro.sample(parname, distfn.HalfNormal(0.001))

    if "log(R)" in parname:
        return numpyro.sample(parname, distfn.Uniform(-2,3.0))  
    
    if parname == "mass_ratio":
        return numpyro.sample("mass_ratio", distfn.Uniform(1e-5, 1.0))

    if parname == "Av":
        return numpyro.sample("Av", distfn.Uniform(1E-6,5.0))
    if parname == 'dist':
        return numpyro.sample("dist", distfn.Uniform(1,200000.0))  
    if parname == "photjitter":
        return numpyro.sample("photjitter", distfn.HalfNormal(0.001))


def determineprior(parname, priorinfo, *args):
    # advanced priors
    if (priorinfo[0] is 'IMF'):
        mass_le,mass_ue = priorinfo[1]['mass_le'],priorinfo[1]['mass_ue']
        return numpyro.sample("initial_Mass",IMF_Prior(low=mass_le,high=mass_ue))

    if (priorinfo[0] is 'GAL'):
        dist_le,dist_ue = priorinfo[1]['dist_ll'],priorinfo[1]['dist_ul']
        GP = Gal_Prior(l=priorinfo[1]['l'],b=priorinfo[1]['b'],low=dist_le,high=dist_ue)
        return numpyro.sample("dist",GP)

    if (priorinfo[0] is 'GALAGE'):
        GP = Gal_Prior(l=priorinfo[1]['l'],b=priorinfo[1]['b'])
        return numpyro.sample("dist",GP)

    if (priorinfo[0] is 'binchem'):
        # last option is both stars have same FeH and aFe (this is the default)
        if priorinfo[1][0] == 'normal':
            feh_a = numpyro.sample('[Fe/H]_a', distfn.Uniform(-4.0,0.5))
            feh_b = numpyro.sample('[Fe/H]_b', distfn.Normal(feh_a,priorinfo[1][1]))
            afe_a = numpyro.sample('[a/Fe]_a', distfn.Uniform(-0.2,0.6))
            afe_b = numpyro.sample('[a/Fe]_b', distfn.Normal(afe_a,priorinfo[1][1]))
            return (feh_a,feh_b,afe_a,afe_b)

        elif priorinfo[1][0] == 'uniform':
            feh_a = numpyro.sample('[Fe/H]_a', distfn.Uniform(-4.0,0.5))
            feh_b = numpyro.sample('[Fe/H]_b', distfn.Uniform(
                feh_a-priorinfo[1][1],feh_a+priorinfo[1][1]))
            afe_a = numpyro.sample('[a/Fe]_a', distfn.Uniform(-0.2,0.6))
            afe_b = numpyro.sample('[a/Fe]_b', distfn.Uniform(
                afe_a-priorinfo[1][1],afe_a+priorinfo[1][1]))
            return (feh_a,feh_b,afe_a,afe_b)

        else:
            feh_a = numpyro.sample('[Fe/H]_a', distfn.Uniform(-4.0,0.5))
            feh_b = numpyro.deterministic('[Fe/H]_b',feh_a)
            afe_a = numpyro.sample('[a/Fe]_a', distfn.Uniform(-0.2,0.6))
            afe_b = numpyro.deterministic('[a/Fe]_b',afe_a)
            return (feh_a,feh_b,afe_a,afe_b)

    # handle lsf properly
    if "lsf_array" in parname:
        specindex = parname.split('_')[-1]
        return jnp.asarray(priorinfo[0]) * numpyro.sample(
            "lsf_scaling_{}".format(specindex), distfn.Uniform(*priorinfo[1]))
    
    if 'vmic' in parname:
        # check to see if user wants to use relationship for vmic
        if (priorinfo[0] is 'Bruntt2012'):
            teff = args[0]
            logg = args[1]
            vmic_pred = 1.095 + (5.44E-4) * (teff-5700.0) + (2.56E-7) * (teff-5700.0)**2.0 - 0.378 * (logg - 4.0)
            if priorinfo[1] == 'fixed':
                return numpyro.deterministic(parname,vmic_pred)
            if priorinfo[1] == 'normal':
                return numpyro.sample(parname,distfn.TruncatedDistribution(
                    distfn.Normal(loc=vmic_pred,scale=0.1),
                    low=0.5,high=3.0))


    # handle vrads and mass ratio
    # Based on the equation q = (v_a - vrad_sys) / (vrad_sys - v_b),
    # which can be solved v_b to give: v_b = vrad_sys - (v_a - vrad_sys)/q

    # Keeping vrad_sys as just a uniform prior between +/- 500 km/s for now
    # TODO: Figure out where to put in a user-defined flag to make vrad_sys
    # normal instead of uniform
    if "vrad" in parname:
        mass_ratio = numpyro.sample("mass_ratio", distfn.Uniform(1e-5, 1.0))
        vradsys = numpyro.sample("vrad_sys", distfn.Uniform(-500.0, 500.0))
        vrada = numpyro.sample("vrad_a", distfn.Uniform(-500.0, 500.0))

        if priorinfo[0] == 'Wilson1941':
            vradb = numpyro.deterministic("vrad_b",
                                    vradsys - (vrada - vradsys)/(mass_ratio))
        else:
            vradb = numpyro.sample("vrad_b", distfn.Uniform(-500.0, 500.0))
        
        return (mass_ratio, vradsys, vrada, vradb)
    
    # define user defined priors

    # standard prior distributions
    if priorinfo[0] == 'uniform':
        return numpyro.sample(parname, distfn.Uniform(*priorinfo[1]))
    if priorinfo[0] == 'normal':
        return numpyro.sample(parname, distfn.Normal(*priorinfo[1]))
    if priorinfo[0] == 'halfnormal':
        return numpyro.sample(parname, distfn.HalfNormal(priorinfo[1]))
    if priorinfo[0] == 'tnormal':
        return numpyro.sample(parname, distfn.TruncatedDistribution(
            distfn.Normal(loc=priorinfo[1][0],scale=priorinfo[1][1]),
            low=priorinfo[1][2],high=priorinfo[1][3]))
    if priorinfo[0] == 'fixed':
        return numpyro.deterministic(parname, priorinfo[1])