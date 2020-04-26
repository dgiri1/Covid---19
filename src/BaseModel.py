import theano
import re, pandas as pd, datetime, numpy as np, scipy as sp, pymc3 as pm, patsy as pt, theano.tensor as tt
theano.config.compute_test_value = 'off' # BUG: may throw an error for flat RVs
from collections import OrderedDict
from sampling_utils import *


class SpatioTemporalFeature(object):
    def __init__(self):
        self._call_ = np.frompyfunc(self.call, 2, 1)

    def __call__(self, times, locations):
        # print("Type of days in {}:{}".format("__call__",type(times)))
        np_date_array = np.array([pd.Timestamp(i.year,i.month,i.day) for i in times])
        return self._call_(np_date_array.reshape((-1,1)), np.asarray(locations).reshape((1,-1))).astype(np.float32)

class SpatioTemporalDailyDemographicsFeature(SpatioTemporalFeature):
    def __init__(self, state_dict, group, scale=1.0):
        self.dict = {
            (day, state): val*scale
            for state,values in state_dict.items() if state not in ('0400000US72')
            for (g, day),val in values["demographics"].items()
            if g == group
        }
        super().__init__()

    def call(self, yearday,state):
        key = "{}/{}/20".format(yearday.month,yearday.day)
        return self.dict.get((key,state))
    
class TemporalSigmoidGrowthFeature(SpatioTemporalFeature):
    # Sigmoidal growth
    def __init__(self, t0, scale):
        self.t0 = t0
        self.scale = scale
        super().__init__()

    def call(self, t, x):
        # print("Type of days in {}:{}".format("call",type(t)))
        return sp.special.expit((t.weekofyear-self.t0)/self.scale)


class SpatialEastWestFeature(SpatioTemporalFeature):
    def __init__(self, state_dict):
        self.dict = {
            state: 1.0 if "Northeast" in values["region"] else (0.5 if "West" in values["region"] else 0.0)
            for state,values in state_dict.items()
        }
        super().__init__()

    def call(self, yearday,state):
        return self.dict.get(state)

class TemporalFourierFeature(SpatioTemporalFeature):
    # weekly cycle
    def __init__(self, i, t0, scale):
        self.t0 = t0
        self.scale = scale
        self.τ = (i//2 + 1)*2*np.pi
        self.fun = np.sin if (i%2)==0 else np.cos
        super().__init__()

    def call(self, t, x):
        # print("Type of days in {}:{}".format("call",type(t)))
        return self.fun((t.weekofyear-self.t0)/self.scale*self.τ)

class TemporalSigmoidFeature(SpatioTemporalFeature):
    # Monthly period
    def __init__(self, t0, scale):
        self.t0 = t0
        self.scale = scale
        super().__init__()

    def call(self, t, x):
        # print("Type of days in {}:{}".format("call",type(t)))
        return sp.special.expit((t.day-self.t0)/self.scale)


class IAEffectLoader(object):
    generates_stats = False
    def __init__(self, var, filenames, days, states):
        self.vars = [var]
        self.samples = []
        for filename in filenames:
            try:
                with open(filename, "rb") as f:
                    tmp=pkl.load(f)
            except FileNotFoundError:
                print("Warning: File {} not found!".format(filename))
                pass
            except Exception as e:
                print(e)
            else:
                m = tmp["ia_effects"]
                ws = list(tmp["predicted day"])
                cs = list(tmp["predicted state"])
                w_idx = np.array([ws.index(w) for w in days]).reshape((-1,1))
                c_idx = np.array([cs.index(c) for c in states])
                self.samples.append(np.moveaxis(m[w_idx,c_idx,:], -1, 0).reshape((m.shape[-1], -1)).T)

    def step(self, point):
        new = point.copy()
        # res = new[self.vars[0].name]
        new_res = self.samples[np.random.choice(len(self.samples))]
        new[self.vars[0].name] = new_res
        return new

    def stop_tuning(*args):
        pass

    @property
    def vars_shape_dtype(self):
        shape_dtypes = {}
        for var in self.vars:
            dtype = np.dtype(var.dtype)
            shape = var.dshape
            shape_dtypes[var.name] = (shape, dtype)
        return shape_dtypes

class BaseModel(object):
    """
    Model for disease prediction.
    The model has 4 types of features (predictor variables):
    * temporal (functions of time)
    * spatial (functions of space, i.e. longitude, latitude)
    * state_specific (functions of time and space, i.e. longitude, latitude)
    * interaction effects (functions of distance in time and space relative to each datapoint)
    """

    def __init__(self, trange, states, ia_effect_filenames, num_ia=16, model=None, include_ia=True, include_eastwest=True, include_demographics=True, include_temporal=True, orthogonalize=False):
        self.state_info = states
        self.ia_effect_filenames = ia_effect_filenames
        self.num_ia = num_ia if include_ia else 0
        self.include_ia = include_ia
        self.include_eastwest = include_eastwest
        self.include_demographics = include_demographics
        self.include_temporal = include_temporal
        self.trange = trange

        first_month=self.trange[0].month
        last_month=self.trange[1].month
        self.features = {
                "temporal_seasonal": {"temporal_fourier_{}".format(i): TemporalFourierFeature(i, 1, 30) for i in range(4)} if False else {},
                "temporal_trend": {"temporal_sigmoid_{}".format(i): TemporalSigmoidGrowthFeature(10, i/2) for i in range(first_month,last_month+1)} if self.include_temporal else {},
                "spatiotemporal": {"demographic_{}".format(group): SpatioTemporalDailyDemographicsFeature(self.state_info, group) for group in ["[0-5)", "[5-20)", "[20-65)"]} if self.include_demographics else {},
                "spatial": {"eastwest": SpatialEastWestFeature(self.state_info)} if self.include_eastwest else {},
                "exposure": {"exposure": SpatioTemporalDailyDemographicsFeature(self.state_info, "total", 1.0/1000000)}
            }
        
        self.Q = np.eye(16, dtype=np.float32)
        if orthogonalize:
            # transformation to orthogonalize IA features
            T = np.linalg.inv(np.linalg.cholesky(gaussian_gram([125.0,250.0,500.0,1000.0]))).T
            for i in range(4):
                self.Q[i*4:(i+1)*4, i*4:(i+1)*4] = T


    def evaluate_features(self, days, states):
        # print("Type of days in {}:{}".format("evaluate_features",type(days)))
        all_features = {}
        for group_name,features in self.features.items():
            group_features = {}
            for feature_name,feature in features.items():
                feature_matrix = feature(days, states)
                group_features[feature_name] = pd.DataFrame(feature_matrix[:,:], index=days, columns=states).stack()
            all_features[group_name] = pd.DataFrame([], index=pd.MultiIndex.from_product([days,states]), columns=[]) if len(group_features)==0 else pd.DataFrame(group_features)
        return all_features

    def init_model(self, target):
        days,states = target.index, target.columns
        # print("Type of days in {}:{}".format("init_model",type(days)))
        # extract features
        features = self.evaluate_features(days, states)
        Y_obs = target.stack().values.astype(np.float32)
        T_S = features["temporal_seasonal"].values.astype(np.float32)
        T_T = features["temporal_trend"].values.astype(np.float32)
        TS = features["spatiotemporal"].values.astype(np.float32)
        S = features["spatial"].values.astype(np.float32)

        log_exposure = np.log(features["exposure"].values.astype(np.float32).ravel())

        # extract dimensions
        num_obs = np.prod(target.shape)
        num_t_s = T_S.shape[1]
        num_t_t = T_T.shape[1]
        num_ts = TS.shape[1]
        num_s = S.shape[1]
        print((num_obs,num_s,num_t_s,num_t_t,num_ts))

        with pm.Model() as self.model:
            # interaction effects are generated externally -> flat prior
            IA    = pm.Flat("IA", testval=np.ones((num_obs, self.num_ia)),shape=(num_obs, self.num_ia))

            # priors
            #δ = 1/√α
            δ     = pm.HalfCauchy("δ", 2, testval=1.0)
            α     = pm.Deterministic("α", np.float32(1.0)/δ)
            W_ia  = pm.Normal("W_ia", mu=0, sd=3, testval=np.zeros(self.num_ia), shape=self.num_ia)
            W_t_s = pm.Normal("W_t_s", mu=0, sd=10, testval=np.zeros(num_t_s), shape=num_t_s)
            W_t_t = pm.Normal("W_t_t", mu=0, sd=10, testval=np.zeros(num_t_t), shape=num_t_t)
            W_ts  = pm.Normal("W_ts", mu=0, sd=10, testval=np.zeros(num_ts), shape=num_ts)
            W_s   = pm.Normal("W_s", mu=0, sd=10, testval=np.zeros(num_s), shape=num_s)
            self.param_names = ["δ", "W_ia", "W_t_s", "W_t_t", "W_ts", "W_s"]
            self.params = [δ, W_ia, W_t_s, W_t_t, W_ts, W_s]

            # calculate interaction effect
            IA_ef = tt.dot(tt.dot(IA, self.Q), W_ia)
            (print(f"IA:{IA.shape},T_S:{T_S.shape},W_t_s:{W_t_s.shape},T_T:{T_T.shape},W_t_t:{W_t_t.shape},TS:{TS.shape},W_ts:{W_ts.shape},S:{S.shape},W_s:{W_s.shape}"))
            # calculate mean rates
            μ = pm.Deterministic("μ", 
                # (1.0+tt.exp(IA_ef))*
                tt.exp(IA_ef + tt.dot(T_S, W_t_s) + tt.dot(T_T, W_t_t) + tt.dot(TS, W_ts) + tt.dot(S, W_s) + log_exposure)
            )

            # constrain to observations
            pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=Y_obs)


    def sample_parameters(self, target, n_init=100, samples=1000, chains=None, cores=8, init="advi", target_accept=0.8, max_treedepth=10, **kwargs):
        """
            sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)
        Samples from the posterior parameter distribution, given a training dataset.
        The basis functions are designed to be causal, i.e. only data points strictly predating the predicted time points are used (this implies "one-step-ahead"-predictions).
        """
        # model = self.model(target)

        self.init_model(target)

        if chains is None:
            chains = max(2,cores)
            

        with self.model:
            # run!
            ia_effect_loader = IAEffectLoader(self.model.IA, self.ia_effect_filenames, target.index, target.columns)
            nuts = pm.step_methods.NUTS(vars=self.params, target_accept=target_accept, max_treedepth=max_treedepth)
            steps = (([ia_effect_loader] if self.include_ia else [] ) + [nuts] )
            trace = pm.sample(samples, steps, chains=chains, cores=cores, compute_convergence_checks=False, **kwargs)
            # trace = pm.sample(0, steps, tune=samples+tune, discard_tuned_samples=False, chains=chains, cores=cores, compute_convergence_checks=False, **kwargs)
            # trace = trace[tune:]
        return trace

    def sample_predictions(self, target_days, target_states, parameters, init="auto"):
        # extract features
        features = self.evaluate_features(target_days, target_states)

        T_S = features["temporal_seasonal"].values
        T_T = features["temporal_trend"].values
        TS = features["spatiotemporal"].values
        S = features["spatial"].values
        log_exposure = np.log(features["exposure"].values.ravel())

        # extract coefficient samples
        α = parameters["α"]
        W_ia = parameters["W_ia"]
        W_t_s = parameters["W_t_s"]
        W_t_t = parameters["W_t_t"]
        W_ts = parameters["W_ts"]
        W_s = parameters["W_s"]
        

        ia_l = IAEffectLoader(None, self.ia_effect_filenames, target_days, target_states)

        num_predictions = len(target_days)*len(target_states)
        num_parameter_samples = α.size
        y = np.zeros((num_parameter_samples, num_predictions), dtype=int)
        μ = np.zeros((num_parameter_samples, num_predictions), dtype=np.float32)

        for i in range(num_parameter_samples):
            IA_ef = np.dot(np.dot(ia_l.samples[np.random.choice(len(ia_l.samples))], self.Q), W_ia[i])
            # μ[i,:] = (1.0+np.exp(IA_ef))*np.exp(np.dot(T_S, W_t_s[i]) + np.dot(T_T, W_t_t[i]) + np.dot(TS, W_ts[i]) + np.dot(S, W_s[i]) + log_exposure)
            μ[i,:] = np.exp(IA_ef + np.dot(T_S, W_t_s[i]) + np.dot(T_T, W_t_t[i]) + np.dot(TS, W_ts[i]) + np.dot(S, W_s[i]) + log_exposure)
            y[i,:] = pm.NegativeBinomial.dist(mu=μ[i,:], alpha=α[i]).random()

        return {"y": y, "μ": μ, "α": α}
