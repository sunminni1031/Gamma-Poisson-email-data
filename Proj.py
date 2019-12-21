import numpy as np
import scipy.stats
import scipy.special
from scipy.special import digamma, logsumexp, loggamma
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib


def matrx2list(matrx):
    matrx_shape = matrx.shape
    nonzero_idx = np.argwhere(matrx>0)
    nonzero_list = []
    for idx in nonzero_idx:
        nonzero_list.append(list(idx)+[matrx[tuple(idx)]])
    nonzero_list.append(list(matrx_shape)+[0])
    return np.array(nonzero_list)


def gen_rn_idx(num_nodes,max_day,model):
    if 'static' in model:
        rn_idx = np.arange(0,num_nodes)
        np.random.seed(42)
        np.random.shuffle(rn_idx)
        n_train = int(0.8*num_nodes)
        rn_idx1 = np.arange(0,num_nodes)
        np.random.seed(43)
        np.random.shuffle(rn_idx1)
        n_train1 = int(0.6*num_nodes)
    elif model=='temporal':    
        rn_idx = np.arange(0,max_day+1)
        np.random.seed(42)
        np.random.shuffle(rn_idx)
        n_train = int(0.8*(max_day+1))
        rn_idx1 = np.arange(0,num_nodes)
        np.random.seed(43)
        np.random.shuffle(rn_idx1)
        n_train1 = int(0.6*num_nodes)
    else:
        raise ValueError('model not implemented.')    
    return rn_idx,rn_idx1,n_train,n_train1


def load_data(model,file_name='email_data/email-Eu-core'):
    if 'static' in model:
        file_model = 'static'
    elif 'temporal' in model:
        file_model = 'temporal'
    try:
        data_train = np.load(file_name+'-directed-{}-train.npy'.format(file_model))
        data_test1 = np.load(file_name+'-directed-{}-test1.npy'.format(file_model))
        data_test2 = np.load(file_name+'-directed-{}-test2.npy'.format(file_model))
    except:
        with open('{}-temporal.txt'.format(file_name)) as f:
            data_sec = [np.array(line.split()).astype(int) for line in f]
        data_sec = np.array(data_sec)
        # change the format
        nodes = np.array(list(set(data_sec[:,:-1].flatten())))
        num_nodes = len(nodes)
        t_steps = np.array(list(set(data_sec[:,-1])))
        max_day = np.max(t_steps)//(24*60*60)
        data_day_ = np.zeros((num_nodes,num_nodes,max_day+1),dtype=int)
        for cur in data_sec:
            cur_day = cur[-1]//(24*60*60)
            cur_i = np.argmax(nodes==cur[0])
            cur_j = np.argmax(nodes==cur[1])
            data_day_[cur_i,cur_j,cur_day]+=1
        rn_idx,rn_idx1,n_train,n_train1 = gen_rn_idx(num_nodes,max_day,model)
        if model=='temporal':
            # temporal
            # directed-temporal
            # train and test
            mask = np.zeros(data_day_.shape,dtype=bool)
            mask[:,:,rn_idx[:n_train]] = 1
            np.save(file_name+'-directed-temporal-train.npy',\
                    matrx2list(data_day_*mask))
            # train and test in test
            mask = np.logical_not(mask)
            mask[:,:,rn_idx[n_train:]][rn_idx1[n_train1:]] = 0
            np.save(file_name+'-directed-temporal-test1.npy',\
                    matrx2list(data_day_*mask))
            mask[:,:,rn_idx[n_train:]][rn_idx1[n_train1:]] = 1
            mask[:,:,rn_idx[n_train:]][rn_idx1[:n_train1]] = 0
            np.save(file_name+'-directed-temporal-test2.npy',\
                    matrx2list(data_day_*mask))
        elif 'static' in model:
            # static  
            # directed-static
            data_static_ = np.sum(data_day_,-1)
            # train and test
            mask = np.zeros(data_static_.shape,dtype=bool)
            mask[rn_idx[:n_train]] = 1
            np.save(file_name+'-directed-static-train.npy',\
                    matrx2list(data_static_*mask))   
            # train and test in test
            mask = np.logical_not(mask)
            mask[rn_idx[n_train:]][:,rn_idx1[n_train1:]] = 0
            np.save(file_name+'-directed-static-test1.npy',\
                    matrx2list(data_static_*mask))
            mask[rn_idx[n_train:]][:,rn_idx1[n_train1:]] = 1
            mask[rn_idx[n_train:]][:,rn_idx1[:n_train1]] = 0
            np.save(file_name+'-directed-static-test2.npy',\
                    matrx2list(data_static_*mask))
            
        data_train = np.load(file_name+'-directed-{}-train.npy'.format(model))
        data_test1 = np.load(file_name+'-directed-{}-test1.npy'.format(model))
        data_test2 = np.load(file_name+'-directed-{}-test2.npy'.format(model))
    return data_train,data_test1,data_test2


# initialization
def init(seed,hyper_params,model):
    eps = hyper_params['eps']
    num_comm = hyper_params['num_comm']
    thres = hyper_params['thres']
    if model=='static' or model=='temporal':
        np.random.seed(seed=seed)
        theta_s = np.random.gamma(eps,1/eps,size=(num_nodes,num_comm))
        theta_s[theta_s<thres]=thres
        np.random.seed(seed=seed+1)
        theta_r = np.random.gamma(eps,1/eps,size=(num_nodes,num_comm))
        theta_r[theta_r<thres]=thres
        params = {'theta_s':theta_s,
                  'theta_r':theta_r}
        if model=='temporal':
            np.random.seed(seed=seed+2)
            psi = np.random.gamma(eps,1/eps,size=(num_t,num_comm))
            psi[psi<thres]=thres
            params['psi'] = psi
    elif model=='Hstatic':
        np.random.seed(seed=seed)
        beta_s = np.random.gamma(eps,1/eps,size=(num_nodes))
        beta_s[beta_s<thres]=thres
        np.random.seed(seed=seed+1)
        theta_s = np.random.gamma(eps,1/beta_s,size=(num_comm,num_nodes)).T
        theta_s[theta_s<thres]=thres
        np.random.seed(seed=seed+2)
        beta_r = np.random.gamma(eps,1/eps,size=(num_nodes))
        beta_r[beta_r<thres]=thres
        np.random.seed(seed=seed+3)
        theta_r = np.random.gamma(eps,1/beta_r,size=(num_comm,num_nodes)).T
        theta_r[theta_r<thres]=thres
        params = {'beta_s':beta_s,'theta_s':theta_s,
                  'beta_r':beta_r,'theta_r':theta_r}
    else:
        raise ValueError('model not implemented.')
    return params


#log joint
def log_joint(hyper_params,params,model):
    eps = hyper_params['eps']
    res = 0
    theta_s = params['theta_s']
    theta_r = params['theta_r']
    if model=='temporal' or model=='static':
        res += np.sum(np.log(scipy.stats.gamma.pdf(theta_s,eps,0,1/eps)))
        res += np.sum(np.log(scipy.stats.gamma.pdf(theta_r,eps,0,1/eps)))
        if model=='temporal':
            psi = params['psi']
            res += np.sum(np.log(scipy.stats.gamma.pdf(psi,eps,0,1/eps)))
            for cur_data in np.concatenate((data_train,data_test1)):
                cur_i,cur_j,cur_t,cur_cnt = cur_data
                cur_rate = (theta_s[cur_i,:]*theta_r[cur_j,:]*psi[cur_t,:]).sum(0)
                res += cur_cnt*np.log(cur_rate)-loggamma(cur_cnt+1)
            res -= (theta_s.sum(0)*theta_r.sum(0)*psi[rn_idx[:n_train]].sum(0)).sum()
            res -= (theta_s[rn_idx1[:n_train1]].sum(0)\
                    *theta_r.sum(0)*psi[rn_idx[n_train:]].sum(0)).sum()
        else: 
            for cur_data in np.concatenate((data_train,data_test1)):
                cur_i,cur_j,cur_cnt = cur_data
                cur_rate = np.sum(theta_s[cur_i,:]*theta_r[cur_j,:])
                res += cur_cnt*np.log(cur_rate)-loggamma(cur_cnt+1)
            res -= np.sum(theta_s[rn_idx[:n_train]].sum(0)*theta_r.sum(0))
            res -= np.sum(theta_s[rn_idx[n_train:]].sum(0)\
                          *theta_r[rn_idx1[:n_train1]].sum(0))
    elif model=='Hstatic':
        beta_s = params['beta_s']
        beta_r = params['beta_r']
        res += np.sum(np.log(scipy.stats.gamma.pdf(beta_s,eps,0,1/eps)))
        res += np.sum(np.log(scipy.stats.gamma.pdf(beta_r,eps,0,1/eps)))
        res += np.sum(eps*np.log(beta_s)-loggamma(eps)\
                      +(eps-1)*np.log(theta_s.T)-beta_s*theta_s.T)
        res += np.sum(eps*np.log(beta_r)-loggamma(eps)\
                      +(eps-1)*np.log(theta_r.T)-beta_r*theta_r.T)
        for cur_data in np.concatenate((data_train,data_test1)):
            cur_i,cur_j,cur_cnt = cur_data
            cur_rate = np.sum(theta_s[cur_i,:]*theta_r[cur_j,:])
            res += cur_cnt*np.log(cur_rate)-loggamma(cur_cnt+1)
        res -= np.sum(theta_s[rn_idx[:n_train]].sum(0)*theta_r.sum(0))
        res -= np.sum(theta_s[rn_idx[n_train:]].sum(0)\
                      *theta_r[rn_idx1[:n_train1]].sum(0)) 
    else:
        raise ValueError('model not implemented.')
    return res
        

def calc_gd(hyper_params,params,model):
    eps = hyper_params['eps']
    theta_s = params['theta_s']
    theta_r = params['theta_r']
    if model=='temporal' or model=='static':
        theta_s_gd = (eps-1)/theta_s-eps
        theta_r_gd = (eps-1)/theta_r-eps
        if model=='temporal':
            psi = params['psi']
            psi_gd = (eps-1)/psi-eps
            for cur_data in data_train:
                cur_i,cur_j,cur_t,cur_cnt = cur_data
                cur_rate = np.sum(theta_s[cur_i,:]*theta_r[cur_j,:]*psi[cur_t,:])
                cur_coef = cur_cnt/cur_rate
                theta_s_gd[cur_i,:] += cur_coef*(theta_r[cur_j,:]*psi[cur_t,:])
                theta_r_gd[cur_j,:] += cur_coef*(theta_s[cur_i,:]*psi[cur_t,:])
                psi_gd[cur_t,:] += cur_coef*(theta_s[cur_i,:]*theta_r[cur_j,:])
            #train
            theta_s_gd -= theta_r.sum(0)*psi[rn_idx[:n_train]].sum(0)
            theta_r_gd -= theta_s.sum(0)*psi[rn_idx[:n_train]].sum(0)
            psi_gd[rn_idx[:n_train]] -= theta_s.sum(0)*theta_r.sum(0)
            #test1
            theta_s_gd[rn_idx1[:n_train1]] -= theta_r.sum(0)*psi[rn_idx[n_train:]].sum(0)
            theta_r_gd -= theta_s[rn_idx1[:n_train1]].sum(0)*psi[rn_idx[n_train:]].sum(0)
            psi_gd[rn_idx[n_train:]] -= theta_s[rn_idx1[:n_train1]].sum(0)*theta_r.sum(0)
            
            grads = {'theta_s':theta_s_gd,'theta_r':theta_r_gd,'psi':psi_gd}
        
        else:
            for cur_data in np.concatenate((data_train,data_test1)):
                cur_i,cur_j,cur_cnt = cur_data
                cur_rate = np.sum(theta_s[cur_i,:]*theta_r[cur_j,:])
                cur_coef = cur_cnt/cur_rate
                theta_s_gd[cur_i,:] += cur_coef*theta_r[cur_j,:]
                theta_r_gd[cur_j,:] += cur_coef*theta_s[cur_i,:]
            
            theta_s_gd[rn_idx[:n_train]] -= theta_r.sum(0)
            theta_r_gd -= theta_s[rn_idx[:n_train]].sum(0)     
            
            theta_s_gd[rn_idx[n_train:]] -= theta_r[rn_idx1[:n_train1]].sum(0)
            theta_r_gd[rn_idx1[:n_train1]] -= theta_s[rn_idx[n_train:]].sum(0)
            
            grads = {'theta_s':theta_s_gd,'theta_r':theta_r_gd}
    elif model=='Hstatic':
        beta_s = params['beta_s']
        beta_r = params['beta_r']
        beta_s_gd = (eps-1)/beta_s-eps
        beta_r_gd = (eps-1)/beta_r-eps
        beta_s_gd += (eps/beta_s-theta_s.T).sum(0)
        beta_r_gd += (eps/beta_r-theta_r.T).sum(0)
        theta_s_gd = ((eps-1)/theta_s.T-beta_s).T
        theta_r_gd = ((eps-1)/theta_r.T-beta_r).T
        for cur_data in np.concatenate((data_train,data_test1)):
            cur_i,cur_j,cur_cnt = cur_data
            cur_rate = np.sum(theta_s[cur_i,:]*theta_r[cur_j,:])
            cur_coef = cur_cnt/cur_rate
            theta_s_gd[cur_i,:] += cur_coef*theta_r[cur_j,:]
            theta_r_gd[cur_j,:] += cur_coef*theta_s[cur_i,:]
        
        theta_s_gd[rn_idx[:n_train]] -= theta_r.sum(0)
        theta_r_gd -= theta_s[rn_idx[:n_train]].sum(0)     
        
        theta_s_gd[rn_idx[n_train:]] -= theta_r[rn_idx1[:n_train1]].sum(0)
        theta_r_gd[rn_idx1[:n_train1]] -= theta_s[rn_idx[n_train:]].sum(0)
        
        grads = {'beta_s':beta_s_gd,'beta_r':beta_r_gd,
                 'theta_s':theta_s_gd,'theta_r':theta_r_gd}        
    
    else:
        raise ValueError('model not implemented.')
    return grads


def score(params,model):
    res = 0
    theta_s = params['theta_s']
    theta_r = params['theta_r']
    if model=='temporal':
        psi = params['psi']
        for cur_data in data_test2:
            cur_i,cur_j,cur_t,cur_cnt = cur_data
            cur_rate = np.sum(theta_s[cur_i,:]*theta_r[cur_j,:]*psi[cur_t,:])
            res += cur_cnt*np.log(cur_rate)-loggamma(cur_cnt+1)
        res -= np.sum(theta_s[rn_idx1[n_train1:]].sum(0)\
                      *theta_r.sum(0)\
                      *psi[rn_idx[n_train:]].sum(0))
    elif 'static' in model:
        for cur_data in data_test2:
            cur_i,cur_j,cur_cnt = cur_data
            cur_rate = np.sum(theta_s[cur_i,:]*theta_r[cur_j,:])
            res += cur_cnt*np.log(cur_rate)-loggamma(cur_cnt+1)
        res -= np.sum(theta_s[rn_idx[n_train:]].sum(0)\
                      *theta_r[rn_idx1[n_train1:]].sum(0))            
    else:
        raise ValueError('model not implemented.')
    return res


def gradient_ascent(num_comm,alpha=0.1,max_iter=500,n_inits=20):
    model_str = '_{}'.format(model)
    file_name = 'result_files/result{}_ncomm={}.dict'.format(model_str,num_comm)
    hyper_params = {'eps':1,'num_comm':num_comm,'thres':1e-5}
    thres = hyper_params['thres']
    beta = [0.9,0.999]
    epsilon = 1e-8
    try:
        result = joblib.load(file_name)
        evals_inits = list(result['eval_inits'])
        scores_inits = list(result['score_inits'])
        best_score = result['best_score']
        best_params = result['best_params']
        if n_inits == -1:
            params = best_params
            evals = [log_joint(hyper_params,params,model)]
            scores = [score(params,model)]
            mmts = {} #moments
            p_list = params.keys()
            for p_nm in p_list:
                mmts[p_nm] = np.zeros((2,)+params[p_nm].shape)
            for it in np.arange(1,max_iter+1):
                cur_alpha =  alpha*np.sqrt(1-beta[1]**it)/(1-beta[0]**it)
                grads = calc_gd(hyper_params,params,model)
                for p_nm in p_list:
                    mmts[p_nm][0] = beta[0]*mmts[p_nm][0]+(1-beta[0])*grads[p_nm]
                    mmts[p_nm][1] = beta[1]*mmts[p_nm][1]+(1-beta[1])*(grads[p_nm]**2)
                    params[p_nm] += cur_alpha*mmts[p_nm][0]/(np.sqrt(mmts[p_nm][1])+epsilon)
                    params[p_nm][params[p_nm]<thres] = thres
                evals.append(log_joint(hyper_params,params,model))
                scores.append(score(params,model))            
            result['post_evals'] = evals
            result['post_scores'] = scores
            result['post_params'] = params
            joblib.dump(result,file_name)
    except:
        result = {}
        evals_inits = []
        scores_inits = []
        best_score = -np.inf
        best_params = 0
    n_inits_done = len(evals_inits)
    if n_inits>n_inits_done:
        for init_seed in np.arange(n_inits_done,n_inits):
            params = init(init_seed,hyper_params,model)
            evals = [log_joint(hyper_params,params,model)]
            scores = [score(params,model)]
            mmts = {} #moments
            p_list = params.keys()
            for p_nm in p_list:
                mmts[p_nm] = np.zeros((2,)+params[p_nm].shape)
            for it in np.arange(1,max_iter+1):
                cur_alpha =  alpha*np.sqrt(1-beta[1]**it)/(1-beta[0]**it)
                grads = calc_gd(hyper_params,params,model)
                for p_nm in p_list:
                    mmts[p_nm][0] = beta[0]*mmts[p_nm][0]+(1-beta[0])*grads[p_nm]
                    mmts[p_nm][1] = beta[1]*mmts[p_nm][1]+(1-beta[1])*(grads[p_nm]**2)
                    params[p_nm] += cur_alpha*mmts[p_nm][0]/(np.sqrt(mmts[p_nm][1])+epsilon)
                    params[p_nm][params[p_nm]<thres] = thres
                evals.append(log_joint(hyper_params,params,model))
                scores.append(score(params,model))
            evals_inits.append(evals)
            scores_inits.append(scores)
            if scores[-1]>best_score:
                best_score = scores[-1]
                best_params = params    
        result = {'eval_inits':np.array(evals_inits),
                  'score_inits':np.array(scores_inits),
                  'best_score':best_score,
                  'best_params':best_params}
        joblib.dump(result,file_name)
    return 
    
    
import platform
if platform.system()=='Linux':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_comm", help="num_comm", type=int)
    parser.add_argument("--model",help="model",type=str)
    parser.add_argument("--alg",help="alg",type=str)
    args = parser.parse_args()
    num_comm = args.num_comm
    model = args.model #temporal,static,Hstatic
    alg = args.alg #grad CAVI
    
    data_train,data_test1,data_test2 = load_data(model)
    num_nodes = data_train[-1,0]
    num_t = data_train[-1,2]
    data_train = data_train[:-1,:]
    data_test1 = data_test1[:-1,:]
    data_test2 = data_test2[:-1,:]
    rn_idx,rn_idx1,n_train,n_train1 = gen_rn_idx(num_nodes,num_t-1,model)
    if alg=='grad':
        gradient_ascent(num_comm)
    elif alg=='post_grad':
        gradient_ascent(num_comm,alpha=0.5,max_iter=5000,n_inits=-1)
    else:
        raise ValueError('algorithm not implemented.')
elif 1:
#    model = 'static'
#    data_train,data_test1,data_test2 = load_data(model)
#    num_nodes = data_train[-1,0]
#    num_t = data_train[-1,2]
#    data_train = data_train[:-1,:]
#    data_test1 = data_test1[:-1,:]
#    data_test2 = data_test2[:-1,:]
#    rn_idx,rn_idx1,n_train,n_train1 = gen_rn_idx(num_nodes,num_t-1,model)
    
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial' 
    def plot_eval_score(model,plot_len=20,comm_list=[1,2,3,4,5,6,7,8,9,10,15,20,25,30]):
        result_ncomm = {}
        final_evals = []
        best_scores = []
        fig,axes = plt.subplots(1,2,figsize=(6,1.5))
        for i_comm,num_comm in enumerate(comm_list):
            model_str = '_{}'.format(model)
            file_name = 'result_files/result{}_ncomm={}.dict'.format(model_str,num_comm)
            result_ncomm[num_comm] = joblib.load(file_name)
            cur_inits = [result_ncomm[num_comm]['eval_inits'],\
                         result_ncomm[num_comm]['score_inits']]
            final_evals.append(cur_inits[0][:,-1])
            best_scores.append(cur_inits[1][:,-1])        
            for plot_i in range(2):
                plot_x = np.arange(0,cur_inits[plot_i].shape[1])[-plot_len:]
                y = cur_inits[plot_i].mean(0)[-plot_len:]
                yerr = cur_inits[plot_i].std(0)[-plot_len:]
                axes[plot_i].plot(plot_x,y,label='n_comm={}'.format(num_comm))#color='C{}'.format(i_comm)
                axes[plot_i].fill_between(plot_x,y-yerr,y+yerr,alpha=0.2)#color='C{}'.format(i_comm))
                axes[plot_i].set_xlabel('iteration')
            axes[0].set_title('training log-joint')
            axes[1].set_title('test log-likelihood')
            #axes[1].legend()
        fig,axes = plt.subplots(1,2,figsize=(6,1.5))
        cur_inits = [np.array(final_evals),np.array(best_scores)]
        for plot_i in range(2):
            y = cur_inits[plot_i].mean(-1)
            yerr = cur_inits[plot_i].std(-1)
            ymin = cur_inits[plot_i].min(-1)
            ymax = cur_inits[plot_i].max(-1)
            axes[plot_i].plot(comm_list,y,color='C{}'.format(plot_i))
            axes[plot_i].fill_between(comm_list,ymin,ymax,alpha=0.2,color='C{}'.format(plot_i))
            axes[plot_i].set_xlabel('number of communities')
        axes[0].set_title('training log-joint')
        axes[1].set_title('test log-likelihood')
    
    def plot_CAVI_result():
        comm_list = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]
        fig,axes = plt.subplots(1,2,figsize=(6,1.5))
        CAVI_result = {}
        best_scores = []
        for num_comm in comm_list:
            file_name = 'result_files/CAVI_static_ncomm={}.dict'.format(num_comm)
            CAVI_result[num_comm] = joblib.load(file_name)
            plot_x = np.arange(0,CAVI_result[num_comm]['score_inits'].shape[1])
            y = CAVI_result[num_comm]['score_inits'].mean(0)
            yerr = CAVI_result[num_comm]['score_inits'].std(0)
            axes[0].plot(plot_x,y,label='n_c={}'.format(num_comm))
            axes[0].fill_between(plot_x,y-yerr,y+yerr,alpha=0.2)
            axes[0].set_xlabel('iteration')
            best_scores.append(CAVI_result[num_comm]['score_inits'][:,-1])
        best_scores = np.array(best_scores)
        y = best_scores.mean(-1)
        yerr = best_scores.std(-1)
        axes[1].plot(comm_list,y)
        axes[1].fill_between(comm_list,y-yerr,y+yerr,alpha=0.2)
        axes[1].set_xlabel('num_comm')
        axes[1].set_title('test_log-likelihood')
    
    def rearr(num_comm,model,side,num_act=100):    
        file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
        result = joblib.load(file_name)
        params = result['best_params']
        p_nm = 'psi' if side=='psi' else 'theta_{}'.format(side)
        x = params[p_nm]
        idx = np.argsort(-np.sum(x,1))[:num_act]
        rearr_idx = []
        for i_comm in range(num_comm):
            cur_idx = np.argwhere((np.argmax(x,1)==i_comm))[:,0]
            rearr_idx.append(np.intersect1d(cur_idx,idx))
        rearr_idx = np.concatenate(rearr_idx,0)
        return rearr_idx
    
    def plot_rearr(rearr_s_idx,rearr_r_idx,num_comm,model,color_bar=True):
        file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
        result = joblib.load(file_name)
        params = result['best_params']
        
        theta_s = params['theta_s']
        theta_r = params['theta_r']
        
        fig = plt.figure(figsize=(2,6))
        ax = fig.add_axes((0.35,0.3,0.6,0.55))
        img = ax.imshow(theta_s[rearr_s_idx],aspect='auto',cmap='Blues')
        pos_arr = np.array([0,num_comm-1])
        label_arr = pos_arr+1
        ax.set_xticks(pos_arr)
        ax.set_xticklabels(label_arr)
        if color_bar:
            fig.colorbar(img,ax=ax)
            
        fig = plt.figure(figsize=(2,6))
        ax = fig.add_axes((0.35,0.3,0.6,0.55))
        img = ax.imshow(theta_r[rearr_r_idx],aspect='auto',cmap='Blues')
        pos_arr = np.array([0,num_comm-1])
        label_arr = pos_arr+1
        ax.set_xticks(pos_arr)
        ax.set_xticklabels(label_arr)
        if color_bar:
            fig.colorbar(img,ax=ax)
    
        fig = plt.figure(figsize=(2,6))
        ax = fig.add_axes((0.35,0.3,0.6,0.55))
        img = ax.imshow(theta_r[rearr_r_idx]\
                        [:,np.linspace(num_comm-1,0,num_comm).astype(int)],\
                        aspect='auto',cmap='Blues')
        pos_arr = np.array([0,num_comm-1])
        label_arr = num_comm-pos_arr
        ax.set_xticks(pos_arr)
        ax.set_xticklabels(label_arr)
        if color_bar:
            fig.colorbar(img,ax=ax)
    
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes((0.35,0.3,0.6,0.55))
        img = ax.imshow(np.dot(theta_s[rearr_s_idx],theta_r[rearr_r_idx].T),cmap='Reds')
        if color_bar:
            fig.colorbar(img,ax=ax)
    
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_axes((0.35,0.3,0.6,0.55))
        img = ax.imshow(np.dot(theta_r[rearr_r_idx].T,theta_s[rearr_s_idx]),cmap='Reds')
        pos_arr = np.array([0,num_comm-1])
        label_arr = pos_arr+1
        ax.set_xticks(pos_arr)
        ax.set_xticklabels(label_arr,rotation='vertical')
        ax.set_yticks(pos_arr)
        ax.set_yticklabels(label_arr)
        if color_bar:
            fig.colorbar(img,ax=ax)
    
        fig,axes = plt.subplots(1,4,figsize=(28,6))
        if 'static' in model:
            matrx = np.dot(theta_s[rearr_s_idx],theta_r[rearr_r_idx].T)
        elif model=='temporal':
            psi = params['psi']
            matrx = np.dot(theta_s[rearr_s_idx],np.diag(psi.sum(0)))
            matrx = np.dot(matrx,theta_r[rearr_r_idx].T)
        img = axes[0].imshow(matrx,cmap='Reds')
        if color_bar:
            fig.colorbar(img,ax=axes[0])
        data_static_ = np.load('email_data/data_static_.npy')
        img = axes[1].imshow(data_static_[rearr_s_idx][:,rearr_r_idx],cmap='Reds')
        if color_bar:
            fig.colorbar(img,ax=axes[1])
        img = axes[2].imshow(np.log(matrx+1),cmap='Reds')
        if color_bar:
            fig.colorbar(img,ax=axes[2])
        img = axes[3].imshow(np.log(data_static_[rearr_s_idx][:,rearr_r_idx]+1),cmap='Reds')
        if color_bar:
            fig.colorbar(img,ax=axes[3])
    
    
    def cmp_model(num_comm_cmp=[4,25]):
        model = 'static'
        rearr_s_idx = [rearr(num_comm,model,'s',num_nodes+1) for num_comm in num_comm_cmp]
        data_static_ = np.load('email_data/data_static_.npy')
        fig,axes = plt.subplots(2,3,figsize=(3*4,2*4))
        file_names = ['result_files/result_{}_ncomm={}.dict'.format(model,num_comm) for num_comm in num_comm_cmp]
        for ir, rearr_idx in enumerate(rearr_s_idx):
            for im, num_comm in enumerate(num_comm_cmp):
                result = joblib.load(file_names[im])
                params = result['best_params']
                theta_s = params['theta_s']
                theta_r = params['theta_r']
                matrx = np.dot(theta_s[rearr_idx],theta_r[rearr_idx].T)
                img = axes[ir,im].imshow(np.log(matrx+1),cmap='Reds')
                fig.colorbar(img,ax=axes[ir,im])
                if im==0:
                    axes[ir,im].set_ylabel('arrangement {}'.format(ir+1))
                if ir==0:
                    axes[ir,im].set_title('model ({} communities)'.format(num_comm))
            cur_data = data_static_[rearr_idx][:,rearr_idx]
            img = axes[ir,im+1].imshow(np.log(cur_data+1),cmap='Reds')
            fig.colorbar(img,ax=axes[ir,im+1])
            if ir==0:
                axes[ir,im+1].set_title('actual data')
    
    
    def plot_time():
        num_comm = 10
        plot_comm = np.arange(num_comm)
        model = 'static'
        plot_len = 500
        side='s'
        file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
        result = joblib.load(file_name)
        params = result['best_params']
        x = params['theta_{}'.format(side)]
        #idx = np.argsort(-np.sum(x,1))[:100]
        y_ncomm = []
        for i_comm in range(num_comm):
            cur_idx = np.argwhere((np.argmax(x,1)==i_comm))[:,0]
            #cur_idx = (np.intersect1d(cur_idx,idx))
            y = data_day_[cur_idx,:,:][:,cur_idx,:].sum((0,1))
            y_ncomm.append(y)
        mpl.rcParams['font.size'] = 10    
        fig,axes = plt.subplots(len(plot_comm),1,figsize=(7,1.5*len(plot_comm)))
        for ic,i_comm in enumerate(plot_comm):
            y = y_ncomm[i_comm]
            axes[ic].plot(y[:plot_len],label='community {}'.format(i_comm),color='C{}'.format(ic))
            axes[ic].legend()
        axes[ic].set_xlabel('time/day')    
        axes[0].set_title('communities\' total activity per day')
        
        model = 'temporal'
        side='s'
        file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
        result = joblib.load(file_name)
        params = result['best_params']
        x = params['theta_{}'.format(side)]
        #idx = np.argsort(-np.sum(x,1))[:100]
        fig,axes = plt.subplots(num_comm,1,figsize=(15,1.5*num_comm))
        for i_comm in range(num_comm):
            cur_idx = np.argwhere((np.argmax(x,1)==i_comm))[:,0]
            #cur_idx = (np.intersect1d(cur_idx,idx))
            y = data_day_[cur_idx,:,:][:,cur_idx,:].sum((0,1))
            axes[i_comm].plot(y[:plot_len],label='nc_{}'.format(i_comm),color='C{}'.format(i_comm))
            axes[i_comm].legend()
        
        model = 'temporal'
        file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
        result = joblib.load(file_name)
        params = result['best_params']
        psi = params['psi']
        fig,axes = plt.subplots(num_comm,1,figsize=(15,1.5*num_comm))
        for i_comm in range(num_comm):
            y = psi[:,i_comm]
            axes[i_comm].plot(y[:plot_len],label='nc_{}'.format(i_comm),color='C{}'.format(i_comm))
            axes[i_comm].legend()

    def plot_tmp_corr():
        model = 'static'
        side='s'    
        if model=='static':
            comm_list = [2,3,4,5,6,7,8,9,10,20,30,40,50,60,80,100,200]
        else:
            comm_list = [2,3,4,5,6,7,8,9,15,20,25,30]
        comm_corr = []
        for num_comm in comm_list:
            file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
            result = joblib.load(file_name)
            params = result['best_params']
            x = params['theta_{}'.format(side)]
            y_ncomm = []
            for i_comm in range(num_comm):
                cur_idx = np.argwhere((np.argmax(x,1)==i_comm))[:,0]
                #y = data_day_[cur_idx,:,:][:,cur_idx,:].sum((0,1))
                y = data_day_[np.ix_(cur_idx,cur_idx)].sum((0,1))
                y_ncomm.append(y)
            y_ncomm = np.array(y_ncomm)
            y_corrs = []
            for i_comm in np.arange(0,num_comm-1):
                for j_comm in np.arange(i_comm+1,num_comm):
                    x1 = y_ncomm[i_comm]
                    x1 = x1 - x1.mean()
                    x2 = y_ncomm[j_comm]
                    x2 = x2 - x2.mean()
                    cur_corr = (x1*x2).sum()/(np.sqrt((x1**2).sum()*(x2**2).sum())+1e-8)
                    y_corrs.append(cur_corr)
            comm_corr.append(np.mean(y_corrs))  
        fig = plt.figure(figsize=(3,1.5))
        ax = fig.add_axes((0.35,0.3,0.6,0.55))
        ax.plot(comm_list,comm_corr)
        ax.set_title('temporal correlation between communities')
        ax.set_xlabel('num of communities in model')
        ax.set_ylim([0,1])
#        fig,axes = plt.subplots(2,1,sharey=True,figsize=(4,4))
#        ax = axes[0]
#        ax.plot(comm_list,comm_corr,label='static model')
#        ax.set_ylim([0,1])
#        ax.legend()
#        ax = axes[1]
#        ax.plot(comm_list[:11],comm_corr[:11],label='static model')
#        ax.plot(tmp_comm_list,tmp_comm_corr,label='temporal model')
#        ax.set_xlabel('num of communities in model')
#        ax.set_ylim([0,1])
#        ax.legend()


    def plot_act_hist():
        fig,axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=(9,3))
        for idx_s,side in enumerate(['s','r']):
            sum_axis = 1 if side=='s' else 0
            side_nm = 'senders' if side=='s' else 'receivers'
            data_static_ = np.load('email_data/data_static_.npy')
            
            ax = axes[idx_s,0]
            _ = ax.hist(np.log(data_static_.sum(sum_axis)+1),bins=20)
            if idx_s==1:
                ax.set_xlabel('log(activity+1)')
            ax.set_ylabel('num of {}'.format(side_nm))
            if idx_s==0:
                ax.set_title('actual data')
            
            model='static'
            num_comm = 4
            file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
            result = joblib.load(file_name)
            params = result['best_params']
            theta_s = params['theta_s']
            theta_r = params['theta_r']
            ax=axes[idx_s,1]
            _ = ax.hist(np.log(np.dot(theta_s,theta_r.T).sum(sum_axis)+1),bins=20)
            if idx_s==1:
                ax.set_xlabel('log(activity+1)')
            if idx_s==0:
                ax.set_title('model (4 communities)')
            
            model='static'
            num_comm = 25
            file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
            result = joblib.load(file_name)
            params = result['best_params']
            theta_s = params['theta_s']
            theta_r = params['theta_r']
            ax=axes[idx_s,2]
            _ = ax.hist(np.log(np.dot(theta_s,theta_r.T).sum(sum_axis)+1),bins=20)
            if idx_s==1:
                ax.set_xlabel('log(activity+1)')
            if idx_s==0:
                ax.set_title('model (25 communities)')
    
    
    #def plot_sender_day():
    model = 'temporal'
    num_comm = 15
    color_bar = 0
    plot_t = 500
    
    file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
    result = joblib.load(file_name)
    params = result['best_params']
    
    theta_s = params['theta_s']
    theta_r = params['theta_r']
    psi = params['psi']
    
    act_idx = np.argsort(-data_day_.sum((1,2)))[:50]
    act_theta_s = theta_s[act_idx]
    
    rearr_idx = []
    for i_comm in range(num_comm):
        cur_idx = np.argwhere((np.argmax(act_theta_s,1)==i_comm))[:,0]
        rearr_idx.append(cur_idx)
    rearr_idx = np.concatenate(rearr_idx,0)
    
    matrx1 = act_theta_s[rearr_idx]
    fig = plt.figure(figsize=(2,6))
    ax = fig.add_axes((0.35,0.3,0.6,0.55))
    img = ax.imshow(matrx1,aspect='auto',cmap='Blues')
    pos_arr = np.array([0,num_comm-1])
    label_arr = pos_arr+1
    ax.set_xticks(pos_arr)
    ax.set_xticklabels(label_arr)
    if color_bar:
        fig.colorbar(img,ax=ax)
    
    matrx = np.dot(matrx1,np.diag(theta_r.sum(0)))
    matrx = np.dot(matrx,psi[:plot_t].T)
    data_mtrx = data_day_.sum(1)
    data_mtrx = data_mtrx[act_idx][rearr_idx][:,:plot_t]
    
    fig,axes = plt.subplots(1,2,figsize=(9,4.5))
    img = axes[0].imshow(np.log(matrx+1),aspect='auto',cmap='Reds')
    fig.colorbar(img,ax=axes[0])
    axes[1].imshow(np.log(data_mtrx+1),aspect='auto',cmap='Reds')
    axes[1].set_xlabel('day')
    fig.colorbar(img,ax=axes[1])
    
    
    mx = np.dot(matrx1,theta_r[act_idx][rearr_idx].T)
    data_mx = data_static_[act_idx][rearr_idx][:,act_idx][:,rearr_idx]
    fig,axes = plt.subplots(1,2,figsize=(9,4))
    img = axes[0].imshow(np.log(mx+1),aspect='auto',cmap='Reds')
    fig.colorbar(img,ax=axes[0])
    axes[1].imshow(np.log(data_mx+1),aspect='auto',cmap='Reds')
    fig.colorbar(img,ax=axes[1])
        #np.dot(matrx1,np.diag(theta_r.sum(0)))

#    model='temporal'
#    num_comm = 10
#    rearr_s_idx = rearr(num_comm,'temporal','s',num_nodes+1)
#    rearr_r_idx = rearr(num_comm,'temporal','r',num_nodes+1)
#    plot_rearr(rearr_s_idx,rearr_r_idx,num_comm,'temporal',color_bar=False)

#    fig = plt.figure(figsize=(2,6))
#    ax = fig.add_axes((0.35,0.3,0.6,0.55))
#    img = ax.imshow(matrx2\
#                    [:,np.linspace(num_comm-1,0,num_comm).astype(int)],\
#                    aspect='auto',cmap='Blues')
#    pos_arr = np.array([0,num_comm-1])
#    label_arr = num_comm-pos_arr
#    ax.set_xticks(pos_arr)
#    ax.set_xticklabels(label_arr)
#    if color_bar:
#        fig.colorbar(img,ax=ax)

#    fig = plt.figure(figsize=(2,2))
#    ax = fig.add_axes((0.35,0.3,0.6,0.55))
#    img = ax.imshow(np.dot(matrx2.T,matrx1),cmap='Reds')
#    pos_arr = np.array([0,num_comm-1])
#    label_arr = pos_arr+1
#    ax.set_xticks(pos_arr)
#    ax.set_xticklabels(label_arr,rotation='vertical')
#    ax.set_yticks(pos_arr)
#    ax.set_yticklabels(label_arr)
#    if color_bar:
#        fig.colorbar(img,ax=ax)



    
#    model='static'
#    num_comm = 10
#    file_name = 'result_files/result_{}_ncomm={}.dict'.format(model,num_comm)
#    result = joblib.load(file_name)
#    params = result['best_params']
#    theta_s = params['theta_s']