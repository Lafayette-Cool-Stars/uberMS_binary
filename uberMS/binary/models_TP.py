import numpyro
import numpyro.distributions as distfn
from numpyro.distributions import constraints
from numpyro.contrib.control_flow import cond

import jax.numpy as jnp

from .priors import determineprior, defaultprior

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

def planck(wav, T):
    wave_i = wav*(1E-10)
    a = 2.0*h*c**2
    b = h*c/(wave_i*k*T)
    intensity = a/ ( (wave_i**5) * (jnp.exp(b) - 1.0) )
    return intensity

# define the model
def model_specphot(
    indata={},
    fitfunc={},
    priors={},
    additionalinfo={},):

    # pull out the observed data
    specwave   = indata['specwave']
    specobs    = indata['specobs']
    specobserr = indata['specobserr']
    photobs    = indata['photobs']
    photobserr = indata['photobserr']
    filtarray  = indata['filterarray']

    # pull out fitting functions
    genphotfn = fitfunc['genphotfn']
    genspecfn = fitfunc['genspecfn']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])
    vmicbool = additionalinfo['vmicbool']

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "specjitter",
        "photjitter",
        # "Teff_a",
        # "Teff_b",
        # "log(g)_a",
        # "log(g)_b",
        "vrad_a",
        "vrad_b",
        'vrad_sys',
        "vstar_a",
        "vstar_b",
        'log(R)_a',
        'log(R)_b',
        'q',
        "dist",
        "Av",
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # define the primary as the hotter of the two stars
    sample_i['Teff_a'] = numpyro.sample("Teff_a",distfn.Uniform(2500.0, 10000.0))
    sample_i['Teff_b'] = numpyro.sample("Teff_b",distfn.Uniform(2500.0, sample_i['Teff_a']))
    # sample_i['Teff_b'] = numpyro.sample("Teff_b",distfn.Uniform(2500.0, 10000.0))

    sample_i['log(g)_a'] = numpyro.sample("log(g)_a",distfn.Uniform(0.0, 5.5))
    # sample_i['log(g)_b'] = numpyro.sample("log(g)_b",distfn.Uniform(sample_i['log(g)_a'],5.5))
    sample_i['log(g)_b'] = numpyro.sample("log(g)_b",distfn.Uniform(0.0,5.5))

    # Mass ratio and velocity priors
    # Based on the equation q = (v_a - vrad_sys) / (vrad_sys - v_b),
    # which can be solved v_b to give: v_b = vrad_sys - (v_a - vrad_sys)/q
    sample_i['q'] = numpyro.sample("q", distfn.Uniform(1e-5, 1.0))

    # Keeping this as just a uniform prior between +/- 500
    # TODO: Need to figure out how Phill did the vrad priors in smes
    # TODO: Figure out where to put in a user-defined flag to make vrad_sys
    # normal instead of uniform
    sample_i['vrad_sys'] = numpyro.sample("vrad_sys", 
                                          distfn.Uniform(-500.0, 500.0))
    
    sample_i['v_a'] = numpyro.sample("v_a", 
                                          distfn.Uniform(-500.0, 500.0))
    
    sample_i['v_b'] = numpyro.deterministic("v_b", 
                                            sample_i['vrad_sys'] - (sample_i['v_a'] - sample_i['vrad_sys'])/(sample_i['q']))

    # require that |vrad_a - vrad_b| > 1.0
    # mixing_dist = distfn.Categorical(probs=jnp.ones(2) / 2.)
    # component_dists = ([
    #     distfn.Uniform(sample_i['vrad_a']-100.0,sample_i['vrad_a']-1.0,),
    #     distfn.Uniform(sample_i['vrad_a']+1.0,sample_i['vrad_a']+100.0,),
    #     ])
    # sample_i['vrad_b'] = numpyro.sample('vrad_b',distfn.MixtureGeneral(mixing_dist, component_dists))
    
    # figure out if user defines extra pc terms in priors
    # or should use the default pc0-pc3 terms
    pcln = len([kk for kk in priors.keys() if 'pc' in kk])
    if pcln == 0:
        pcterms = ['pc0','pc1','pc2','pc3']
    else:
        pcterms = ['pc{0}'.format(x) for x in range(pcln)]

    # now sample from priors for pc terms
    for pp in pcterms:
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic_a' in priors.keys():
            sample_i['vmic_a'] = determineprior('vmic_a',priors['vmic_a'])
        else:
            sample_i['vmic_a'] = defaultprior('vmic_a')
        if 'vmic_b' in priors.keys():
            sample_i['vmic_b'] = determineprior('vmic_b',priors['vmic_b'])
        else:
            sample_i['vmic_b'] = defaultprior('vmic_b')
    else:
        sample_i['vmic_a'] = 1.0
        sample_i['vmic_b'] = 1.0

    # handle various lsf cases
    if 'lsf_array' in priors.keys():
        # user has defined an lsf array, so set as free parameter 
        # a scaling on the lsf
        sample_i['lsf'] = determineprior('lsf_array',priors['lsf_array'])
    else:
        # user hasn't set a lsf array, treat lsf as R
        if 'lsf' in priors.keys():
            sample_i['lsf'] = determineprior('lsf',priors['lsf'])
        else:
            sample_i['lsf'] = defaultprior('lsf')

    # handle different cases for the treatment of [Fe/H] and [a/Fe]
    if 'binchem' in priors.keys():
        (sample_i["[Fe/H]_a"],sample_i["[Fe/H]_b"],sample_i["[a/Fe]_a"],sample_i["[a/Fe]_b"]) = determineprior(None,priors['binchem'])
    else:
        for pp in ["[Fe/H]_a","[Fe/H]_b","[a/Fe]_a","[a/Fe]_b"]:  
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            else:
                sample_i[pp] = defaultprior(pp)

    # sample in a jitter term for error in spectrum
    specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )

    # make the spectral prediciton
    specpars_a = ([
        sample_i['Teff_a'],sample_i['log(g)_a'],sample_i['[Fe/H]_a'],sample_i['[a/Fe]_a'],
        sample_i['vrad_a'],sample_i['vstar_a'],sample_i['vmic_a'],sample_i['lsf']])
    specpars_a += [sample_i['pc{0}'.format(x)] for x in range(len(pcterms))]
    specmod_a = genspecfn(specpars_a,outwave=specwave,modpoly=True)
    specmod_a = jnp.asarray(specmod_a[1])

    specpars_b = ([
        sample_i['Teff_b'],sample_i['log(g)_b'],sample_i['[Fe/H]_b'],sample_i['[a/Fe]_b'],
        sample_i['vrad_b'],sample_i['vstar_b'],sample_i['vmic_b'],sample_i['lsf']])
    specpars_b += [1.0,0.0]
    specmod_b = genspecfn(specpars_b,outwave=specwave,modpoly=True)
    specmod_b = jnp.asarray(specmod_b[1])

    radius_a = 10.0**sample_i['log(R)_a']
    radius_b = 10.0**sample_i['log(R)_b']

    R = (
        (planck(specwave,sample_i['Teff_a']) * radius_a**2.0) / 
        (planck(specwave,sample_i['Teff_b']) * radius_b**2.0)
         )
    specmod_est = (specmod_a + R * specmod_b) / (1.0 + R)

    # calculate likelihood for spectrum
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars_a = ([
        sample_i['Teff_a'],sample_i['log(g)_a'],sample_i['[Fe/H]_a'],sample_i['[a/Fe]_a'],
        sample_i['log(R)_a'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_a = genphotfn(photpars_a)
    photmod_a = [photmod_a[xx] for xx in filtarray]

    photpars_b = ([
        sample_i['Teff_b'],sample_i['log(g)_b'],sample_i['[Fe/H]_b'],sample_i['[a/Fe]_b'],
        sample_i['log(R)_b'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_b = genphotfn(photpars_b)
    photmod_b = [photmod_b[xx] for xx in filtarray]

    photmod_est = (
        [-2.5 * jnp.log10( 10.0**(-0.4 * m_a) + 10.0**(-0.4 * m_b) )
         for m_a,m_b in zip(photmod_a,photmod_b)
         ] 
    )
    photmod_est = jnp.asarray(photmod_est)

    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])


# define the model
def model_spec(
    indata={},
    fitfunc={},
    priors={},
    additionalinfo={},):

    # pull out the observed data
    specwave   = indata['specwave']
    specobs    = indata['specobs']
    specobserr = indata['specobserr']

    # pull out fitting functions
    genspecfn = fitfunc['genspecfn']

    # pull out additional info
    vmicbool = additionalinfo['vmicbool']

    # define sampled parameters apply the user defined priors

    sampledpars = ([
        "specjitter",
        # "Teff_a",
        # "Teff_b",
        # "log(g)_a",
        # "log(g)_b",
        "vrad_a",
        # "vrad_b",
        "vstar_a",
        "vstar_b",
        "log(R)_a",
        "log(R)_b",        
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # define the primary as the hotter of the two stars
    sample_i['Teff_a'] = numpyro.sample("Teff_a",distfn.Uniform(2500.0, 10000.0))
    sample_i['Teff_b'] = numpyro.sample("Teff_b",distfn.Uniform(2500.0, sample_i['Teff_a']))

    sample_i['log(g)_a'] = numpyro.sample("log(g)_a",distfn.Uniform(0.0, 5.5))
    # sample_i['log(g)_b'] = numpyro.sample("log(g)_b",distfn.Uniform(sample_i['log(g)_a'],5.5))
    sample_i['log(g)_b'] = numpyro.sample("log(g)_b",distfn.Uniform(0.0,5.5))

    # require that |vrad_a - vrad_b| > 1.0
    mixing_dist = distfn.Categorical(probs=jnp.ones(2) / 2.)
    component_dists = ([
        distfn.Uniform(sample_i['vrad_a']-100.0,sample_i['vrad_a']-1.0,),
        distfn.Uniform(sample_i['vrad_a']+1.0,sample_i['vrad_a']+100.0,),
        ])
    sample_i['vrad_b'] = numpyro.sample('vrad_b',distfn.MixtureGeneral(mixing_dist, component_dists))
    
    # figure out if user defines extra pc terms in priors
    # or should use the default pc0-pc3 terms
    pcln = len([kk for kk in priors.keys() if 'pc' in kk])
    if pcln == 0:
        pcterms = ['pc0','pc1','pc2','pc3']
    else:
        pcterms = ['pc{0}'.format(x) for x in range(pcln)]

    # now sample from priors for pc terms
    for pp in pcterms:
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    # set vmic only if included in NNs
    if vmicbool:
        if 'vmic_a' in priors.keys():
            sample_i['vmic_a'] = determineprior('vmic_a',priors['vmic_a'])
        else:
            sample_i['vmic_a'] = defaultprior('vmic_a')
        if 'vmic_b' in priors.keys():
            sample_i['vmic_b'] = determineprior('vmic_b',priors['vmic_b'])
        else:
            sample_i['vmic_b'] = defaultprior('vmic_b')
    else:
        sample_i['vmic_a'] = 1.0
        sample_i['vmic_b'] = 1.0

    # handle various lsf cases
    if 'lsf_array' in priors.keys():
        # user has defined an lsf array, so set as free parameter 
        # a scaling on the lsf
        sample_i['lsf'] = determineprior('lsf_array',priors['lsf_array'])
    else:
        # user hasn't set a lsf array, treat lsf as R
        if 'lsf' in priors.keys():
            sample_i['lsf'] = determineprior('lsf',priors['lsf'])
        else:
            sample_i['lsf'] = defaultprior('lsf')

    # handle different cases for the treatment of [Fe/H] and [a/Fe]
    if 'binchem' in priors.keys():
        (sample_i["[Fe/H]_a"],sample_i["[Fe/H]_b"],sample_i["[a/Fe]_a"],sample_i["[a/Fe]_b"]) = determineprior(None,priors['binchem'])
    else:
        for pp in ["[Fe/H]_a","[Fe/H]_b","[a/Fe]_a","[a/Fe]_b"]:  
            if pp in priors.keys():
                sample_i[pp] = determineprior(pp,priors[pp])
            else:
                sample_i[pp] = defaultprior(pp)

    # sample in a jitter term for error in spectrum
    specsig = jnp.sqrt( (specobserr**2.0) + (sample_i['specjitter']**2.0) )

    # make the spectral prediciton
    specpars_a = ([
        sample_i['Teff_a'],sample_i['log(g)_a'],sample_i['[Fe/H]_a'],sample_i['[a/Fe]_a'],
        sample_i['vrad_a'],sample_i['vstar_a'],sample_i['vmic_a'],sample_i['lsf']])
    specpars_a += [sample_i['pc{0}'.format(x)] for x in range(len(pcterms))]
    specmod_a = genspecfn(specpars_a,outwave=specwave,modpoly=True)
    specmod_a = jnp.asarray(specmod_a[1])

    specpars_b = ([
        sample_i['Teff_b'],sample_i['log(g)_b'],sample_i['[Fe/H]_b'],sample_i['[a/Fe]_b'],
        sample_i['vrad_b'],sample_i['vstar_b'],sample_i['vmic_b'],sample_i['lsf']])
    specpars_b += [1.0,0.0]
    specmod_b = genspecfn(specpars_b,outwave=specwave,modpoly=True)
    specmod_b = jnp.asarray(specmod_b[1])

    radius_a = 10.0**sample_i['log(R)_a']
    radius_b = 10.0**sample_i['log(R)_b']

    R = (
        (planck(specwave,sample_i['Teff_a']) * radius_a**2.0) / 
        (planck(specwave,sample_i['Teff_b']) * radius_b**2.0)
         )
    specmod_est = (specmod_a + R * specmod_b) / (1.0 + R)

    # calculate likelihood for spectrum
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

# define the model
def model_phot(
    indata={},
    fitfunc={},
    priors={},
    additionalinfo={},):

    # pull out the observed data
    photobs    = indata['photobs']
    photobserr = indata['photobserr']
    filtarray  = indata['filterarray']

    # pull out fitting functions
    genphotfn = fitfunc['genphotfn']

    # pull out additional info
    parallax = additionalinfo.get('parallax',[None,None])

    # define sampled parameters apply the user defined priors
    sampledpars = ([
        "photjitter",
        "Teffp",
        "log(g)",
        "[Fe/H]",
        "[a/Fe]",
        'log(R)',
        "dist",
        "Av",
        "ff"
        ])

    sample_i = {}
    for pp in sampledpars:  
        if pp in priors.keys():
            sample_i[pp] = determineprior(pp,priors[pp])
        else:
            sample_i[pp] = defaultprior(pp)

    teffs_a = -3.58 * (10**-5) * (sample_i['Teffp']**2.0) + 0.751 * sample_i['Teffp'] + 808.0
    sample_i['Teffs'] = numpyro.sample(
        "Teffs",
        distfn.TruncatedDistribution(distfn.Normal(loc=teffs_a,scale=250.0),
                                     low=teffs_a-500.0,high=teffs_a+500.0))

    # sample in jitter term for error in photometry
    photsig = jnp.sqrt( (photobserr**2.0) + (sample_i['photjitter']**2.0) )

    # make photometry prediction
    photpars_a = ([
        sample_i['Teffp'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_a = genphotfn(photpars_a)
    photmod_a = [photmod_a[xx] for xx in filtarray]

    photpars_b = ([
        sample_i['Teffs'],sample_i['log(g)'],sample_i['[Fe/H]'],sample_i['[a/Fe]'],
        sample_i['log(R)'],sample_i['dist'],sample_i['Av'],3.1])
    photmod_b = genphotfn(photpars_b)
    photmod_b = [photmod_b[xx] for xx in filtarray]

    photmod_est = (
        [-2.5 * jnp.log10( (1.0-sample_i['ff']) * 10.0**(-0.4 * m_a) + sample_i['ff'] * 10.0**(-0.4 * m_b) )
         for m_a,m_b in zip(photmod_a,photmod_b)
         ] 
    )
    photmod_est = jnp.asarray(photmod_est)

    # calculate likelihood of photometry
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    
    # calcluate likelihood of parallax
    numpyro.sample("para", distfn.Normal(1000.0/sample_i['dist'],parallax[1]), obs=parallax[0])
