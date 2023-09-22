#!/usr/bin/env python3

import os
import h5py as h5
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit
import numpy as np
import pandas as pd
from os import path
import n_parameters as parameters


class Fit:
    def __init__(self):
        self.params = parameters.params()
        self.plot = Plot(self.params)
        self.func = Functions(self.params)
        self.data = self.gevp_correlators()
        self.ratio_denom = self.get_ratio_combinations()

    def nucleon_data(self):
        """ Reads nucleon data from h5.
        Constructs same momentum lists.
        Average correlators of same momentum.
        Construct dictionary of momentum^2 with data of shape [ncfg, tsep].
        """
        fpath = self.params["fpath"]["nucleon"]
        file = h5.File(fpath, "r")
        correlators = file.keys()
        avglist = dict()
        dcorr = dict()
        for correlator in correlators:
            dcorr[correlator] = file[f"{correlator}/data"][()].real
            irrep = correlator.split("_")[1]
            mom = correlator.split("_")[0][1:].replace("m", "")
            mom2 = f"{sum([int(i) ** 2 for i in mom])}"
            if mom2 == "5":
                mom2 = f"{mom2}{irrep}"
            if mom2 in avglist:
                avglist[mom2].append(correlator)
            else:
                avglist[mom2] = [correlator]
        if self.params["debug"]:
            """Plot effective mass for each correlator"""
            for d in avglist:
                for c in avglist[d]:
                    self.plot.simple_plot(dcorr, c)
                    break
                break
        data = dict()
        for mom2 in avglist:
            data[mom2] = np.average([dcorr[corr] for corr in avglist[mom2]], axis=0)
        if self.params["debug"]:
            """Plot effective mass for averaged correlator"""
            for mom2 in data:
                self.plot.simple_plot(data, mom2)
        return data

    def singlet_data(self):
        fpath = self.params["fpath"]["singlet"]
        if fpath in ["./data/singlet_S0.hdf5"]:
            file = h5.File(fpath, "r")
            dcorr, avglist = self.get_equivalent_momenta(file)

            if self.params["debug"]:
                phase_check = dict()
                for tag in list(avglist.keys())[4:5]:
                    print(tag)
                    gvdat = gv.dataset.avg_data({key: dcorr[key] for key in avglist[tag]})
                    for c in avglist[tag]:
                        cshape = np.shape(gvdat[c])
                        if len(cshape) == 3:
                            dat = gvdat[c][:, :, 2]
                            mdat = dat
                            pdat = np.array(
                                [
                                    [abs(dat[j, i].mean) for i in range(len(dat))]
                                    for j in range(len(dat))
                                ]
                            )
                            phase = mdat / pdat
                        else:
                            dat = gvdat[c][2]
                            phase = int(dat / abs(dat.mean))
                        if tag in phase_check:
                            phase_check[tag] += phase
                        else:
                            phase_check[tag] = phase
                print("phase_check")
                for tag in phase_check:
                    print(tag)
                    print(phase_check[tag])
            if self.params["debug"]:
                # 1 6
                for tag in list(avglist.keys())[4:5]:
                    for c in avglist[tag]:
                        print(c)
                        sdat = {c: dcorr[c][:, 0, 7, :]}
                        self.plot.simple_plot(sdat, c, "corr")

            data = dict()
            for tag in avglist:
                data[tag] = np.average([dcorr[corr] for corr in avglist[tag]], axis=0)
        elif fpath in ["./data/singlet_S0_avg_mom.hdf5"]:
            data = dict()
            file = h5.File(fpath, "r")
            correlators = file.keys()
            for correlator in correlators:
                irrep = correlator.split("_")[0]
                mom2 = correlator.split("Psq")[1]
                tag = (mom2, irrep)
                data[tag] = file[f"{correlator}/data"][()]
        return data

    def get_equivalent_momenta(self, file):
        correlators = file.keys()
        avglist = dict()
        dcorr = dict()
        for correlator in correlators:
            dcorr[correlator] = file[f"{correlator}/data"][()]
            irrep = correlator.split("_")[1]
            mom = correlator.split("_")[0][1:].replace("m", "")
            mom2 = sum([int(i) ** 2 for i in mom])
            tag = (mom2, irrep)
            if tag in avglist:
                avglist[tag].append(correlator)
            else:
                avglist[tag] = [correlator]
        return dcorr, avglist

    def get_nn_operators(self):
        import re

        fpath = self.params["fpath"]["singlet"]

        if fpath in ["./data/singlet_S0.hdf5"]:
            file = h5.File(fpath, "r")
            _, avglist = self.get_equivalent_momenta(file)
            n_irrep = dict()
            for tag in avglist:
                oplist = list(file[f"/{avglist[tag][0]}"].attrs["opList"])
                momlist = []
                for idx, op in enumerate(oplist):
                    mom = re.findall(r"P=\((.*?)\)", op)
                    mom2 = [sum([int(i) ** 2 for i in m.split(",")]) for m in mom]
                    irrep = [i.split(" ")[1] for i in re.findall(r"\[(.*?)\]", op)]
                    if mom2[0] == 5:
                        mom2[0] = f"5{irrep[0]}"
                    if mom2[1] == 5:
                        mom2[1] = f"5{irrep[1]}"
                    momlist.append(mom2)
                n_irrep[tag] = momlist
        elif fpath in ["./data/singlet_S0_avg_mom.hdf5"]:
            file = h5.File(fpath, "r")
            n_irrep = dict()
            correlators = file.keys()
            for correlator in correlators:
                irrep = correlator.split("_")[0]
                mom2 = correlator.split("Psq")[1]
                tag = (mom2, irrep)
                n_irrep[tag] = [[i.decode('utf-8') for i in c] for c in file[f"{correlator}/irreps"][()]]
        return n_irrep

    def get_ratio_combinations(self):
        autotime = self.params["autotime"]
        data = self.data
        irrep = self.get_nn_operators()
        nonint_lvls = dict()
        for tag in irrep:
            meff_list = []
            for element in irrep[tag]:
                x, meff1 = self.func.meff(data[element[0]])
                meff1 = meff1[x.index(autotime)].mean
                x, meff2 = self.func.meff(data[element[1]])
                meff2 = meff2[x.index(autotime)].mean
                meff_list.append(meff1 + meff2)
            nonint_lvls[tag] = {"meff": np.array(meff_list), "irrep": irrep[tag]}
        ratio_denom = dict()
        for tag in data:
            if tag in ["0", "1", "2", "3", "4", "5F1", "5F2"]:
                continue
            x, meff = self.func.meff(data[tag])
            meff = meff[x.index(autotime)].mean
            idx = np.abs(nonint_lvls[(tag[0], tag[1])]["meff"] - meff).argmin()
            ratio_denom[tag] = nonint_lvls[(tag[0], tag[1])]["irrep"][idx]
        return ratio_denom

    def get_bootstrap_draws(self):
        """
        Read or generate bootstrap list.
        Return as pandas dataframe.
        """
        try:
            ncfgs = len(self.nucleon_data()[0])
            bslist = pd.read_csv(f"./data/bslist_{ncfgs}.csv", sep=";", header=0)
        except:
            from random import randint

            nbs = 5000
            ncfgs = len(self.nucleon_data()[0])
            draws = []
            for idx in range(nbs):
                idraw = []
                for draw in range(ncfgs):
                    idraw.append(randint(0, ncfgs - 1))
                draws.append(idraw)
            output = "nbs;draws\n"
            output += f"0;{str(list(range(ncfgs))).replace(' ', '')}\n"
            output += "\n".join(
                [
                    f"{idx + 1};{str(draw).replace(' ', '')}"
                    for idx, draw in enumerate(draws)
                ]
            )
            with open(f"./data/bslist_{ncfgs}.csv", "w") as file:
                file.write(output)
            bslist = pd.read_csv(f"./data/bslist_{ncfgs}.csv", sep=";", header=0)
        return bslist

    def gevp_correlators(self):

        def get_gevp_rotation(data):
            from scipy.linalg import eigh
            from numpy.linalg import cond

            t0 = self.params["t0"]
            td = self.params["td"]
            drot = dict()
            for key in data:
                if len(np.shape(data[key])) == 4:
                    try:
                        mean = np.average(data[key], axis=0)
                        val, vec = eigh(a=mean[:, :, td], b=mean[:, :, t0])
                        drot[key] = vec
                        print(f"{key} Success, condition numbers:")
                        print(cond(mean[:, :, td]), cond(mean[:, :, t0]))
                    except:
                        print(f"{key} Fail, condition numbers:")
                        print(cond(mean[:, :, td]), cond(mean[:, :, t0]))
                        val1, vec1 = eigh(mean[:, :, td])
                        val2, vec2 = eigh(mean[:, :, t0])
                        print(f"{key} {t0} Eigenvalue spectrum and vector:")
                        print(val1)
                        print(vec1[0])
                        print(f"{key} {td} Eigenvalue spectrum and vector:")
                        print(val2)
                        print(vec2[0])
            return drot

        t0 = self.params["t0"]
        td = self.params["td"]
        datapath = f"./data/gevp_{t0}_{td}.pickle"
        if path.exists(datapath):
            print("Read data from gvar dump")
            gvdata = gv.load(datapath)
        else:
            print("Constructing data from HDF5")
            nucleon = self.nucleon_data()
            singlet = self.singlet_data()
            allsing = {
                key: singlet[key] for key in singlet if len(np.shape(singlet[key])) == 2
            }
            drot = get_gevp_rotation(singlet)
            for key in drot:
                eigVecs = np.fliplr(drot[key])
                rotated_singlet = np.einsum('cijt,in,jm->cnmt', singlet[key], np.conj(eigVecs), eigVecs)
                rotated_singlet = np.diagonal(rotated_singlet, axis1=1, axis2=2)
                for operator in range(np.shape(rotated_singlet)[-1]):
                    opkey = (key[0], key[1], operator)
                    allsing[opkey] = rotated_singlet[:, :, operator].real

            gvdata = gv.dataset.avg_data({**nucleon, **allsing})
            gv.dump(gvdata, datapath)

            if self.params["debug"]:
                for key in gvdata:
                    if len(np.shape(gvdata[key])) == 2:
                        for corr in range(len(gvdata[key][0])):
                            print(key, corr)
                            print(gvdata[key][:, corr])
                            self.plot.simple_plot(
                                {f"{key}_{corr}": gvdata[key][:, corr].flatten()},
                                f"{key}_{corr}",
                            )
                    else:
                        self.plot.simple_plot(gvdata, key)
        return gvdata

    def set_priors(self, submaster, prior, data=None, type="auto"):
        """
        return a dictionary of gvar priors.
        If set to auto, the energy and overlap factors are inferred from meff and zeff.
        If set to auto, data (x, y) is required (data to be fit)
        If set to manual, the priors are read from priors.py
        """
        if type == "auto":
            autotime = self.params["autotime"]
            datax, datay = data
            for key in submaster:
                _, meff = self.func.meff(datay[key], datax[key])
                x, zeff = self.func.zeff(datay[key], datax[key])
                if autotime in x:
                    meff = siground(meff[x.index(autotime)].mean)
                    zeff = siground(zeff[x.index(autotime)].mean)
                else:
                    meff = siground(meff[x.index(x[0])].mean)
                    zeff = siground(zeff[x.index(x[0])].mean)
                prior[(key, "e0")] = gv.gvar(meff, 0.2 * meff)
                # use a large width for zeff in case we aren't in plateau region
                prior[(key, "z0")] = gv.gvar(zeff, 5 * zeff) 
                for n in range(1, self.params["nstates"]):
                    ampi = self.params['ampi']
                    # give a log-normal splitting of 2mpi, with a 1-sigma down fluctuation to mpi
                    prior[(key, f"e{n}")] = gv.gvar(np.log(2*ampi), 0.7)
                    if self.params["gs_factorize"]:
                        prior[(key, f"z{n}")] = gv.gvar(1, 1)
                    else:
                        prior[(key, f"z{n}")] = gv.gvar(0, zeff)
        elif type == "manual":
            pass
        return prior

    def format_data(self, subset):
        x = {
            key: range(self.params["trange"][0], self.params["trange"][1] + 1)
            for key in subset
        }
        y = dict()
        for key in subset:
            if self.params["ratio"] and key in self.ratio_denom:
                k0 = self.ratio_denom[key][0]
                k1 = self.ratio_denom[key][1]
                if self.params["ratio_type"] == "fit":
                    func1 = Functions({"nstates": 1})
                    corr0 = func1({k0: x[key]}, self.result.p)[k0]
                    corr1 = func1({k1: x[key]}, self.result.p)[k1]
                elif self.params["ratio_type"] == "data":
                    corr0 = self.data[k0][x[key]]
                    corr1 = self.data[k1][x[key]]
                denom = corr0 * corr1
            else:
                denom = 1
            y[key] = self.data[key][x[key]] / denom
            if self.params["debug"]:
                self.plot.simple_plot(y, key, x=x[key])
        return x, y

    def reconstruct_gs(self, posterior):
        """
        Reconstruct ground state energy if ratio fit is used
        """
        for subset in self.params["masterkey"]:
            for key in subset:
                if key in ["0", "1", "2", "3", "4", "5F1", "5F2"]:
                    continue
                dset = self.ratio_denom[key]
                offset = posterior[(dset[0], "e0")] + posterior[(dset[1], "e0")]
                if self.params["ratio"]:
                    posterior[(key, "egs")] = posterior[(key, "e0")] + offset
                else:
                    posterior[(key, f"ebind_{dset[0]}_{dset[1]}")] = posterior[(key, "e0")] - offset
        return posterior

    def fit(self):
        posterior = gv.BufferDict()
        for subset in self.params["masterkey"]:
            print(f"Fitting {subset}")
            x, y = self.format_data(subset)
            prior = gv.BufferDict()
            prior = self.set_priors(subset, prior, data=(x, y), type="auto")
            result = lsqfit.nonlinear_fit(
                data=(x, y), prior=prior, fcn=self.func, maxit=100000, fitter=self.params['fitter']
            )
            print(result)
            self.plot.plot_result(result, subset)
            stats = dict()
            stats[(tuple(subset), "Q")] = result.Q
            stats[(tuple(subset), "chi2")] = result.chi2
            stats[(tuple(subset), "dof")] = result.dof
            stats[(tuple(subset), "logGBF")] = result.logGBF

            posterior = {**posterior, **result.p, **stats}
        posterior = self.reconstruct_gs(posterior)
        posterior = {"masterkey": self.params["masterkey"], **posterior}
        self.posterior = posterior
        print(self.posterior)

    def save(self):
        if not self.params["save"]:
            return

        if not os.path.exists('result'):
            os.makedirs('result')
        filename = f"N_n{self.params['nstates']}_t_{self.params['trange'][0]}-{self.params['trange'][1]}.pickle"
        gv.dump(self.posterior, f"./result/{filename}")


def siground(x, sigfig=2):
    from math import log10, floor

    factor = 10 ** (sigfig - 1)
    return round(x, -int(floor(log10(abs(x / factor)))))


class Plot:
    def __init__(self, params):
        self.func = Functions(params)
        self.params = params

    def plot_result(self, fit, subset):
        for correlator in subset:
            print("CORRR")
            print(correlator)
            fig = plt.figure(f"{correlator} effective mass")
            ax = plt.axes()
            dx, dy = self.func.meff(fit.y[correlator], fit.x[correlator])
            fc = self.func(fit.x, fit.p)
            fx, fy = self.func.meff(fc[correlator], fit.x[correlator])
            ax.errorbar(x=dx, y=[i.mean for i in dy], yerr=[i.sdev for i in dy])
            fy1 = np.array([i.mean for i in fy]) - np.array([i.sdev for i in fy])
            fy2 = np.array([i.mean for i in fy]) + np.array([i.sdev for i in fy])
            ax.fill_between(x=fx, y1=fy1, y2=fy2, alpha=0.5)
            if 3 in dx:
                xmin = 3
            else:
                xmin = dx[0]
            if 15 in dx:
                xmax = 15
            else:
                xmax = dx[-1]
            xmin = dx[0]
            xmax = dx[-1]
            ax.set_xlim([xmin, xmax])
            xfilter = np.arange(fx.index(xmin), fx.index(xmax))
            dy1 = np.array([i.mean for i in dy]) - np.array([i.sdev for i in dy])
            dy2 = np.array([i.mean for i in dy]) + np.array([i.sdev for i in dy])
            ymin = min(dy1[xfilter])
            ymax = max(dy2[xfilter])
            ax.set_ylim([ymin, ymax])
            if not os.path.exists('./plots/check_fits'):
                os.makedirs('./plots/check_fits')
            scorrelator = '_'.join([str(k) for k in correlator])
            scorrelator += '_ns_%d_t_%d-%d' \
                %(self.params['nstates'],self.params['trange'][0],self.params['trange'][1])
            plt.savefig(f"./plots/check_fits/meff_{scorrelator}.pdf", transparent=True)

    def simple_plot(self, data, tag, type="meff", x=None):
        try:
            gdata = gv.dataset.avg_data(data[tag])
        except:
            gdata = data[tag]

        xm, meff = self.func.meff(gdata, x)
        xc, corr = self.func.corr(gdata, x)
        if type == "meff":
            y = meff
            x = xm
        elif type == "corr":
            y = corr
            x = xc
        print(tag)
        fig = plt.figure(f"meff {str(tag)}")
        ax = plt.axes()
        ax.errorbar(x=x, y=[i.mean for i in y], yerr=[i.sdev for i in y])
        if self.params['latex']:
            ltype = type.replace('_','\_')
            ltag  = tag#.replace('_','\_')
            #print(ltype, ltag)
            plt.title(f"{ltype} {ltag}")
        else:
            plt.title(f"{type} {tag}")
        plt.draw()
        if not os.path.exists('plots/check_average'):
            os.makedirs('plots/check_average')
        stag = '_'.join([str(k) for k in tag])

        plt.savefig(f"./plots/check_average/{type}_{stag}.pdf")
        plt.close(fig)

        x, zeff = self.func.zeff(gdata)
        fig = plt.figure(f"zeff {str(tag)}")
        ax = plt.axes()
        ax.errorbar(x=x, y=[i.mean for i in zeff], yerr=[i.sdev for i in zeff])
        if self.params['latex']:
            plt.title(f"zeff {ltag}")
        else:
            plt.title(f"zeff {tag}")
        plt.draw()
        plt.savefig(f"./plots/check_average/zeff_{stag}.pdf")
        plt.close(fig)


class Functions:
    def __init__(self, params):
        self.nstates = params["nstates"]
        self.positive_z = params["positive_z"]
        self.gs_factorize = params["gs_factorize"]

    def __call__(self, x, p):
        """
        x is a dictionary of {key: [time slices]}
        p is a dictionary of priors {(key, 'var'): gvar(m, s), ...}
        """
        rd = dict()
        for key in x.keys():
            if self.gs_factorize:
                rd[key] = self.twopoint(key, x[key], p)
            else:
                t = np.array(x[key])
                r = 0
                for n in range(self.nstates):
                    En = p[(key, "e0")]
                    En += sum([np.exp(p[(key, f"e{ni}")]) for ni in range(1, n + 1)])
                    if self.positive_z:
                        r += p[(key, f"z{n}")] ** 2 * np.exp(-En * t)
                    else:
                        r += p[(key, f"z{n}")] * np.exp(-En * t)
                rd[key] = r
        return rd

    def twopoint(self, key, x, p):
        t = np.array(x)
        r_es = self.twopoint_excited_states(key, x, p)
        r = p[(key, "z0")] ** 2 * np.exp(-p[(key, "e0")] * t) * r_es
        return r

    def twopoint_excited_states(self, key, x, p):
        t = np.array(x)
        r = 1
        for n in range(1, self.nstates):
            En = sum([np.exp(p[(key, f"e{ni}")]) for ni in range(1, n + 1)])
            r += p[(key, f"z{n}")] ** 2 * np.exp(-En * t)
        return r

    def corr(self, data, x=None):
        if not x:
            data = data[2: len(data) * 2 // 3][:-1]
            x = range(2, len(data) + 2)
        else:
            data = data[:-1]
            x = x[:-1]
        return x, data

    def meff(self, data, x=None):
        if not x:
            data = data[2: len(data) * 2 // 3]
            x = range(2, len(data) + 2)[:-1]
        else:
            print(x)
            x = x[:-1]
        y = np.log(data / np.roll(data, -1))[:-1]
        return x, y

    def zeff(self, data, x=None):
        _, corr = self.corr(data, x)
        x, meff = self.meff(data, x)
        y = corr * np.exp(meff * x)
        if self.positive_z:
            y = np.sqrt(y)
        return x, y


if __name__ == "__main__":
    fit = Fit()
    fit.fit()
    fit.save()
    # for key in fit.data:
    #    fit.plot.simple_plot(fit.data, key)
