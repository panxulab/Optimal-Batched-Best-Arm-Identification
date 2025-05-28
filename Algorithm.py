import math
import random

import numpy as np
from scipy import stats


def bernouli_kl_divergence(mu1, mu2):
    P = [1 - mu1, mu1]
    Q = [1 - mu2, mu2]
    kldivergence = stats.entropy(P, Q)
    return kldivergence

def chernoff(pull_num, mu, maxindex):
    mubest=mu[maxindex]
    numbest=pull_num[maxindex]
    MuMid = [(numbest * mubest + pull_num[i] * mu[i]) / (numbest + pull_num[i]) for i in range(len(mu))]
    Index= [i for i in range(len(mu))]
    Index.pop(maxindex)
    zj=[bernouli_kl_divergence(mubest, MuMid[i]) * numbest + bernouli_kl_divergence(mu[i], MuMid[i]) * pull_num[i] for i in Index]
    return min(zj)


def beta_func(tau, delta, alpha):
    beta = np.log(2 * np.log(1 / delta) * np.power(tau, alpha) / delta)
    return beta

def beta_func_TS(tau, delta):
    beta = np.log((np.log(tau)+1) / delta)
    return beta

def Ifonc(alpha, mu1, mu2):
    if (alpha == 0) or (alpha == 1):
        return 0
    else:
        mid = alpha * mu1 + (1 - alpha) * mu2
        return alpha * bernouli_kl_divergence(mu1, mid) + (1 - alpha) * bernouli_kl_divergence(mu2, mid)

def cost(mu1, mu2, nu1, nu2):
    if (nu1 == 0) & (nu2 == 0):
        return 0
    else:
        alpha = nu1 / (nu1 + nu2)
        return ((nu1 + nu2) * Ifonc(alpha, mu1, mu2))

def xkofy(y, k, sortb):
    # return x_k(y), i.e. finds x such that g_k(x)=y
    xMax = 1
    while (1 + xMax) * cost(sortb[0], sortb[k], 1 / (1 + xMax), xMax / (1 + xMax)) - y < 0:
        xMax = 2 * xMax
    l = 0
    u = xMax
    pre = 1e-11
    sgn = cost(sortb[0], sortb[k], 1, 0) - y
    while u - l > pre:
        m = (u + l) / 2
        if ((1 + m) * cost(sortb[0], sortb[k], 1 / (1 + m), m / (1 + m)) - y)* sgn > 0:
            l = m
        else:
            u = m
    return (u + l) / 2


def muddle(mu1, mu2, nu1, nu2):
    result=(nu1*mu1 + nu2*mu2)/(nu1+nu2)
    if result==1:
        result=result-1e-16
    return result

def aux(y, sortb):
    # return F_mu(y)-1
    K = len(sortb)
    x = [xkofy(y, k, sortb) for k in range(1, K)]
    m = [muddle(sortb[0], sortb[k], 1, x[k-1]) for k in range(1,K)]
    return (sum([bernouli_kl_divergence(sortb[0],m[k-1])/(bernouli_kl_divergence(sortb[k], m[k-1])) for k in range(1,K)])-1)

def optimal_weights(b):
    sortb = -np.sort(-b)
    Index=np.argsort(-b)
    Index=Index.tolist()
    ymax=0.5
    if bernouli_kl_divergence(sortb[0], sortb[1])==np.inf:
        while aux(ymax, sortb) < 0:
            ymax = ymax * 2
    else:
        ymax = bernouli_kl_divergence(sortb[0], sortb[1])
    l=0
    u=ymax
    pre=1e-11
    sgn=aux(l,sortb)
    while (u - l > pre):
        m = (u + l) / 2
        if (aux(m,sortb) * sgn > 0):
            l = m
        else:
            u = m
    y=(u + l) / 2
    x = [xkofy(y, k, sortb) for k in range(1,len(sortb))]
    x.insert(0, 1)
    invindex=[ Index.index(i) for i in range(len(Index))]
    nuOpt = [i / sum(x) for i in x]
    NuOpt = [nuOpt[i] for i in invindex]
    vOpt = y/sum(x)
    if vOpt==0:
        Tb = np.finfo(np.float64).max
    else:
        Tb=1 / vOpt
    return Tb, NuOpt



def compute_mu(score, sim):
    mu = np.zeros(sim.num())
    for i in range(sim.num()):
        mu[i] = score[i] / sim.pull_num(i)
    return mu


def Tri_BBAI(epsilon, delta, L1, L2, L3, alpha, sim):
    # stage 1
    batch = 0
    score = np.zeros(sim.num())
    for i in range(sim.num()):
        for j in range(L1):
            score[i] += sim.pull(i)
    batch += 1
    q = 1
    wb_list = []
    flag = True
    while flag:
        # stage 2
        b = compute_mu(score, sim)
        maxindex = np.argmax(b)
        for i in range(sim.num()):
            if i == maxindex:
                b[i] = b[i] - epsilon
            else:
                b[i] = b[i] + epsilon
        Tb, wb = optimal_weights(b)
        wb_list.append(wb)
        for i in range(sim.num()):
            if np.power(2, q) * alpha * wb[i] * Tb * np.log(1 / delta) == np.inf:
                Ti = L2
            else:
                Ti = min(math.ceil(np.power(2, q) * alpha * wb[i] * Tb * np.log(1 / delta)), L2)
            for j in range(Ti):
                score[i] += sim.pull(i)
        batch += 1
        if q == 2:
            for i in range(sim.num()):
                if wb_list[q - 1][i] - wb_list[q - 2][i] <= 1 / np.sqrt(sim.num()):
                    flag = False
                    break
        q += 1
    mu = compute_mu(score, sim)
    maxindex = np.argmax(mu)
    # stage 3
    tau = sim.total_pull_num()
    minZj = chernoff(sim.pull_num_array(), mu, maxindex)
    beta = beta_func_TS(tau, delta)
    if minZj >= beta:
        #print('total_sample:', sim.total_pull_num())
        return sim.total_pull_num(), mu, maxindex+1, True, batch

    # stage 3
    batch+=1
    for i in range(sim.num()):
        if sim.pull_num(i) < L3:
            for j in range(int(L3 - sim.pull_num(i))):
                score[i] += sim.pull(i)
    mu = compute_mu(score, sim)
    maxindex = np.argmax(mu)
    return sim.total_pull_num(), mu, maxindex+1, False, batch


def Opt_BBAI(epsilon, delta, L1, L2, L3, alpha, sim):
    # stage 1
    score = np.zeros(sim.num())
    for i in range(sim.num()):
        for j in range(L1):
            score[i] += sim.pull(i)
    b = score / L1
    maxindex = np.argmax(b)
    for i in range(sim.num()):
        if i == maxindex:
            b[i] = b[i] - epsilon
        else:
            b[i] = b[i] + epsilon
    Tb,wb = optimal_weights(b)
    # stage 2
    for i in range(sim.num()):
        if alpha * wb[i] * Tb * np.log(1 / delta)==np.inf:
            Ti=L2
        else:
            Ti = min(math.ceil(alpha * wb[i] * Tb * np.log(1 / delta)), L2)
        # print(Ti)
        for j in range(Ti):
            score[i] += sim.pull(i)
    mu = compute_mu(score, sim)
    maxindex = np.argmax(mu)
    # stage 3
    tau = sim.total_pull_num()
    minZj = chernoff(sim.pull_num_array(), mu, maxindex)
    beta = beta_func_TS(tau, delta)
    batch =2
    if minZj >= beta:
        return sim.total_pull_num(), mu, maxindex + 1, True, batch

    # stage 4
    setlist = []
    B_list = []
    set_0=[i for i in range(sim.num())]
    set_r=set(set_0)
    r=1
    B_r = 0
    setlist.append(set_r)
    B_list.append(B_r)
    while len(setlist[r - 1]) > 1:
        epsilon_r=np.power(0.5,r)/4
        delta_r=delta/(40*np.pi*np.pi*sim.num()*np.power(r,2))
        batch += 1
        for i in setlist[r - 1]:
            d_r = 1 / (2 * np.power(epsilon_r, 2)) * np.log(2 / delta_r) / 100
            for j in range(math.ceil(d_r)):
                score[i] += sim.pull(i)
        mu = compute_mu(score, sim)
        mu_set = [mu[i] for i in setlist[r - 1]]
        p_best = max(mu_set)
        set_k = setlist[r - 1]
        B_list.append(B_list[r - 1] + len(setlist[r - 1]) * math.ceil(d_r))
        for i in setlist[r - 1].copy():
            if mu[i] < p_best - epsilon_r:
                set_k.remove(i)
        setlist.append(set_k)
        for j in range(r - 1):
            list_i = []
            delta_j = delta / (40 * np.pi ** 2 * sim.num() * np.power(j + 1, 2))
            batch += 1
            for i in setlist[j]:
                if B_list[r] * delta_j > B_list[j]:
                    delta_j = delta_j ** 2
                    d_r = 2 * d_r
                    for k in range(math.ceil(d_r)):
                        score[i] += sim.pull(i)
                p_ij = score[i] / sim.pull_num_array()[i]
                list_i.append(p_ij)
            if len(list_i) > 0:
                if max(list_i) >= p_best - epsilon_r:
                    mu = compute_mu(score, sim)
                    maxindex = np.argmax(mu)
                    return sim.total_pull_num(), mu, maxindex + 1, False, batch
        r = r +1
    mu = compute_mu(score, sim)
    maxindex = setlist[r - 1].pop()
    return sim.total_pull_num(), mu, maxindex + 1, False, batch


def Opt_BBAI_New(epsilon, delta, L1, L2, L3, alpha, sim):
    batch =0
    # stage 1
    score = np.zeros(sim.num())
    for i in range(sim.num()):
        for j in range(L1):
            score[i] += sim.pull(i)
    batch += 1
    q = 1
    wb_list = []
    flag = True
    while flag:
        # stage 2
        b = compute_mu(score, sim)
        maxindex = np.argmax(b)
        for i in range(sim.num()):
            if i == maxindex:
                b[i] = b[i] - epsilon
            else:
                b[i] = b[i] + epsilon
        Tb, wb = optimal_weights(b)
        wb_list.append(wb)
        for i in range(sim.num()):
            if np.power(2, q) * alpha * wb[i] * Tb * np.log(1 / delta) == np.inf:
                Ti = L2
            else:
                Ti = min(math.ceil(np.power(2, q) * alpha * wb[i] * Tb * np.log(1 / delta)), L2)
            for j in range(Ti):
                score[i] += sim.pull(i)
        batch += 1
        if q == 2:
            for i in range(sim.num()):
                if wb_list[q - 1][i] - wb_list[q - 2][i] <= 1 / np.sqrt(sim.num()):
                    flag = False
                    break
        q += 1

    mu = compute_mu(score, sim)
    maxindex = np.argmax(mu)
    # stage 3
    tau = sim.total_pull_num()
    minZj = chernoff(sim.pull_num_array(), mu, maxindex)
    beta = beta_func_TS(tau, delta)
    if minZj >= beta:
        return sim.total_pull_num(), mu, maxindex+1, True, batch

    # stage 4
    setlist=[]
    B_list=[]
    set_0=[i for i in range(sim.num())]
    set_r=set(set_0)
    r=1
    B_r=0
    setlist.append(set_r)
    B_list.append(B_r)
    gamma_list = []
    l_list = []
    while len(setlist[r-1])>1:
        epsilon_r=np.power(0.5,r)/4
        delta_r=delta/(40*np.pi*np.pi*sim.num()*np.power(r,2))
        gamma_list.append(delta_r)
        l_list.append(0)
        batch+=1
        for i in setlist[r-1]:
            d_r = 1 / np.power(epsilon_r, 2) * np.log(2 / delta_r) / 20
            for j in range(math.ceil(d_r)):
                score[i] += sim.pull(i)
        mu = compute_mu(score, sim)
        mu_set=[mu[i] for i in setlist[r-1]]
        p_best=max(mu_set)
        set_k=setlist[r-1]
        B_list.append(B_list[r-1]+len(setlist[r-1])*math.ceil(d_r))
        for i in setlist[r-1].copy():
            if mu[i]<p_best-epsilon_r:
                set_k.remove(i)
        setlist.append(set_k)
        for j in range(r-1):
            list_i=[]
            batch += 1
            for i in setlist[j]:
                if B_list[r] * gamma_list[j] * np.power(2, l_list[j]) > B_list[j]:
                    gamma_list[j] = gamma_list[j] * gamma_list[j]
                    t_j = 1 / np.power(epsilon_r, 2) * np.log(2 / gamma_list[j]) / 10
                    if t_j == float('inf'):
                        break
                    for k in range(math.ceil(t_j)):
                        score[i] += sim.pull(i)
                    l_list[j] += 1
                    p_ij = score[i] / sim.pull_num_array()[i]
                    list_i.append(p_ij)
            if len(list_i)>0:
                if max(list_i) > p_best - epsilon_r / 2:
                    mu = compute_mu(score, sim)
                    maxindex = np.argmax(mu)
                    return sim.total_pull_num(), mu, maxindex+1, False, batch
        r=r+1
    mu = compute_mu(score, sim)
    maxindex = setlist[r-1].pop()
    return sim.total_pull_num(), mu, maxindex+1, False, batch


def TrackandStop(delta, alpha, sim):
    score = np.zeros(sim.num())
    for i in range(sim.num()):
        score[i] += sim.pull(i)
    condition=True
    while condition:
        #print('total_sample:', sim.total_pull_num())
        mu = compute_mu(score, sim)
        MaxIndexs = np.argwhere(mu == np.amax(mu))
        MaxIndexs=MaxIndexs.reshape(-1)
        best=np.random.choice(MaxIndexs)
        arm=0
        if len(MaxIndexs)>1:
            arm=best
        else:
            maxindex=MaxIndexs[0]
            tau = sim.total_pull_num()
            minZj = chernoff(sim.pull_num_array(), mu, maxindex)
            beta = beta_func_TS(tau, delta)
            if minZj>beta:
                print(minZj)
                print(beta)
                condition=False
            else:
                if min(sim.pull_num_array())<=max(np.sqrt(tau) - sim.num()/2,0):
                    arm=np.argmin(sim.pull_num_array())
                else:
                    T, w = optimal_weights(mu)
                    arm=np.argmax(w-sim.pull_num_array()/tau)
        score[arm]+=sim.pull(arm)
    mu = compute_mu(score, sim)
    maxindex = np.argmax(mu)
    print('total_sample:',sim.total_pull_num())
    return sim.total_pull_num(), mu, maxindex+1


def ExponentialGapElimination(delta, sim):
    score = np.zeros(sim.num())
    set_0 = [i for i in range(sim.num())]
    set_r = set(set_0)
    r = 1
    batch = 0
    while len(set_r) > 1:
        epsilon_r = np.power(0.5, r) / 4
        delta_r = delta / (50 * r * r * r)
        batch += 1
        for i in set_r:
            t_r = (2 / np.power(epsilon_r, 2)) * np.log(2 / delta_r) / 3900
            # d_r=1/(2*np.power(epsilon_r,2))*np.log(2/delta_r)
            for j in range(math.ceil(t_r)):
                score[i] += sim.pull(i)
        maxindex, score, batch = MedianElimination(set_r, epsilon_r / 2, delta_r, score, sim, batch)
        mu = compute_mu(score, sim)
        for i in set_r.copy():
            if mu[i] < mu[maxindex] - epsilon_r:
                set_r.remove(i)
        r += 1
    mu = compute_mu(score, sim)
    maxindex = set_r.pop()
    return sim.total_pull_num(), mu, maxindex + 1, batch


def MedianElimination(set_r, epsilon, delta, score, sim, batch):
    l = 1
    set_l = set_r.copy()
    epsilon_l = epsilon / 4
    delta_l = delta
    score_l = score
    while len(set_l) > 1:
        batch += 1
        for i in set_l:
            t_l = 1 / np.power(epsilon_l / 2, 2) * np.log(3 / delta_l) / 3900
            # d_r=1/(2*np.power(epsilon_r,2))*np.log(2/delta_r)
            for j in range(math.ceil(t_l)):
                score_l[i] += sim.pull(i)
        mu = compute_mu(score_l, sim)
        m_l = np.median([mu[k] for k in set_l])
        for i in set_l.copy():
            if mu[i] < m_l:
                set_l.remove(i)
        l += 1
        epsilon_l = 3 * epsilon_l / 4
        delta_l = delta_l / 2
    maxindex = set_l.pop()
    return maxindex, score_l, batch


def Top1DeltaEliminate(delta, sim):
    score = np.zeros(sim.num())
    set_r = [i for i in range(sim.num())]
    r = 1
    beta_r = 1
    delta_r = delta / 4
    c = 8
    epsilon = 0.1
    Q = c / epsilon / epsilon
    S_prime = []
    S_top = []
    batch = 0
    while len(set_r) > 0:
        Q_r = beta_r * Q * np.log(1 / delta_r) / 65
        batch += 1
        for i in set_r:
            for j in range(math.ceil(Q_r)):
                score[i] += sim.pull(i)
        mu = compute_mu(score, sim)
        # print(mu)
        mu_r = [mu[k] for k in set_r]
        mu_r_sorted_index = np.argsort(-np.array(mu_r))
        k_prime = min(1, len(set_r))
        top_rank = np.ceil(pow(delta_r / 1, beta_r) * len(set_r) / 2.0) + k_prime - 1;
        batch += 1
        for i in range(k_prime):
            random_arm = random.randint(0, top_rank - 1)
            for j in range(math.ceil(Q_r)):
                score[mu_r_sorted_index[random_arm]] += sim.pull(mu_r_sorted_index[random_arm])
        mu = compute_mu(score, sim)
        # print(mu)
        top_value = mu[mu_r_sorted_index[random_arm]]
        # print(mu_r_sorted_index)
        S_prime.append(top_value)
        S_top.append(mu_r_sorted_index[random_arm])
        set_r_prime = set_r.copy()
        for i in set_r_prime:
            if mu[i] < top_value + 3 * epsilon / 4:
                set_r.remove(i)
        # print(len(set_r))
        if len(set_r) != 0:
            if mu_r_sorted_index[random_arm] in set_r:
                set_r.remove(mu_r_sorted_index[random_arm])
            if len(set_r_prime) <= 2 * delta * len(set_r):
                beta_r = beta_r * len(set_r_prime) / 2 / len(set_r)
            else:
                beta_r = beta_r * len(set_r_prime) / len(set_r)
            delta_r = delta / 2.0 / pow(2.0, r)
            r += 1
    maxindex = S_top[S_prime.index(max(S_prime))]
    return sim.total_pull_num(), mu, maxindex + 1, batch


def MultiRound(delta, sim, epsilon):
    score = np.zeros(sim.num())
    set_r = [i for i in range(sim.num())]
    r = 0
    t = []
    k = 1
    t.append(0)
    batch = 0
    while True:
        r += 1
        e_r = pow(2, -r)
        t_r = (2 / (k * e_r * e_r)) * np.log(4 * len(set_r) * r * r / delta)
        t.append(t_r)
        batch += 1
        for i in range(k):
            for i in set_r:
                sample_times = t[r] - t[r - 1]
                for j in range(math.ceil(sample_times)):
                    score[i] += sim.pull(i)
        mu = compute_mu(score, sim)
        mu_set = [mu[i] for i in set_r]
        p_best = max(mu_set)
        set_r_prime = set_r.copy()
        for i in set_r_prime:
            if mu[i] < p_best - e_r:
                set_r.remove(i)
        if e_r == 0 or len(set_r) == 1:
            break
    maxindex = set_r[0]
    return sim.total_pull_num(), mu, maxindex + 1, batch


def epsilon_BAI(epsilon, delta, arm_set, sim, score, C=100):
    arm_star = 0
    j = 1
    scores = score

    # Initialize arm° with s1 pulls
    s1 = int(np.ceil(16 / epsilon ** 2 * np.log(C / delta)))
    for _ in range(s1):
        scores[arm_star] += sim.pull(arm_star)

    for i in range(1, sim.num()):
        alpha = epsilon / 4 if np.random.rand() < 1 / np.log(j + 1) else epsilon / 2
        ell = 1

        while True:
            # Calculate s_ell according to the provided formula
            s_ell = int(np.ceil((16 / epsilon ** 2) * np.log(C / delta) * (2 ** ell)))
            s_ell_pre = int(np.ceil((16 / epsilon ** 2) * np.log(C / delta) * (2 ** (ell - 1))))
            pulls = s_ell - s_ell_pre
            score_i = 0
            pull_count = 0

            for _ in range(pulls):
                score_i += sim.pull(i)
                pull_count += 1

            mu_hat_i = score_i / pull_count if pull_count > 0 else 0
            mu_hat_star = scores[arm_star] / sim.pull_num(arm_star)
            tau_j = (32 / epsilon ** 2) * np.log(C * j ** 2 / delta)

            if mu_hat_i >= mu_hat_star + alpha and s_ell > tau_j:
                arm_star = i
                j = 1
                break
            elif mu_hat_i < mu_hat_star + alpha:
                j += 1
                break
            else:
                ell += 1

    return arm_star, scores


def ID_BAI(delta, sim):
    r = 1
    S_r = set(range(sim.num()))
    score = np.zeros(sim.num())

    while len(S_r) > 1:
        epsilon_r = 2 ** (-r) / 4
        delta_r = delta / (40 * r ** 2)
        h = 1
        arm_star, score = epsilon_BAI(epsilon_r, delta_r, S_r, sim, score)

        pulls = int(2 / epsilon_r ** 2 * np.log(1 / delta_r))
        for _ in range(pulls):
            score[arm_star] += sim.pull(arm_star)
        I_r = score[arm_star] / sim.pull_num(arm_star)

        B_r = (6 * len(S_r) / epsilon_r ** 2) * np.log(40 / delta_r)

        for arm in S_r.copy():
            if arm == arm_star:
                continue
            if B_r > 0:
                s_i = 0
                l = 1
                pulls = int((2 / epsilon_r ** 2) * np.log(40 / delta_r))
                while s_i <= (2 / epsilon_r ** 2) * np.log(40 * h ** 2 / delta_r):
                    for _ in range(pulls):
                        score[arm] += sim.pull(arm)
                    s_i = s_i + pulls
                    B_r = B_r - pulls
                    p_hat_i = score[arm] / sim.pull_num(arm)

                    if p_hat_i < I_r - epsilon_r:
                        S_r.remove(arm)
                        h = h + 1
                        break
                    l = l + 1
            else:
                pulls = int((2 / epsilon_r ** 2) * np.log(40 / delta_r))
                for _ in range(pulls):
                    score[arm] += sim.pull(arm)
                p_hat_i = score[arm] / sim.pull_num(arm)
                if p_hat_i < I_r - epsilon_r:
                    S_r.remove(arm)
        r = r + 1
    arm_best = S_r.pop()
    return sim.total_pull_num(), score[arm_best] / sim.pull_num(arm_best), arm_best + 1


def collaborative_algorithm(sim, delta, m=1, K=1):
    I0 = set(range(sim.num()))
    m0 = m
    Acc = set()
    Rej = set()
    r = 0
    Tr_1 = 0
    score = np.zeros(sim.num())
    while I0:
        er = 2 ** (-(r + 1))
        Tr = int(8 * np.log(4 * len(I0) * (r + 1) ** 2 / delta) / (K * er ** 2))

        for arm_idx in I0:
            for _ in range(Tr - Tr_1):
                score[arm_idx] += sim.pull(arm_idx)

        # Estimate the mean of each arm
        theta_hat = np.zeros(sim.num())
        for arm_idx in I0:
            theta_hat[arm_idx] = score[arm_idx] / sim.pull_num(arm_idx)

        sorted_arms = sorted(I0, key=lambda i: theta_hat[i], reverse=True)

        if len(sorted_arms) == 1:
            Acc.add(sorted_arms[0])
            break
        for i in sorted_arms:
            if theta_hat[i] > theta_hat[sorted_arms[m]] + er:
                Acc.add(i)

        for i in sorted_arms:
            if theta_hat[i] < theta_hat[sorted_arms[m-1]] - er:
                Rej.add(i)

        m = m - len(Acc)
        I0 = I0 - Acc - Rej
        r += 1
        Tr_1 = Tr
    arm_best = Acc.pop()

    return sim.total_pull_num(), score[arm_best] / sim.pull_num(arm_best), arm_best +1, r
