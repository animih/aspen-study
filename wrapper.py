# just a brunch of functions, that are usefull for code reduction
import numpy as np
import inspect
import matplotlib.pyplot as plt
import time

# func that sums times on each time step
# and returns overall solver work time
def tim(clas):   
    if type(clas).__name__ == 'aspen_log':
        t = clas.gb_res+clas.gb_jac \
      +clas.gb_lin+np.sum(clas.lc_res) \
      +np.sum(clas.lc_jac)+np.sum(clas.lc_lin)
    elif type(clas).__name__ == 'newton_log':
        t = clas.lin+clas.jac+clas.res
    
    return t

# a decorator for test-function with output results
def test_decorator(obj):
    def wraper(*args, **kwargs):
        print('test started')
        solver, X, mes, t, delta = obj(*args, **kwargs)
        print('verdict : ' + mes)
        print('time : {:.4} +- {:.4}'.format(t, delta))

        if type(solver.timelog).__name__ == 'newton_log':
            print('mean newton iterations: ', np.mean(solver.timelog.kn))
        elif type(solver.timelog).__name__ == 'aspen_log':
            print('mean aspen iterations: ', np.mean(solver.timelog.aspen_iters))

        return X, mes, t, delta
    return wraper

# run problem sample_size times, finds mean time and std
def test(solver, sample_size = 5, tmax=1, dyn_bd = False):
    t = 0
    t = np.zeros(sample_size)

    if dyn_bd:
        buf = np.copy(solver.solver.partion)
    for k in range(sample_size):
        if dyn_bd:
            solver.prob.func.reset_jac(solver.solver.partion)
            solver.solver.partion = np.copy(buf)
        solver.init_log()
        #t[k] -= time.time()
        X, mes = solver.solve(tmax)
        t[k] = tim(solver.timelog)
        #t[k] += time.time()
    
    return solver, X, mes, np.mean(t),  np.sqrt(np.std(t))

# ploter func for provided solver
# optional: save = desired filepath to save plot
def show_res(solver, save=None):
    x = np.linspace(0, 1, solver.Nx)
    plt.figure(figsize= (8, 6))
    plt.xlabel('x')
    plt.ylabel('t')
    if type(solver.nl_solver).__name__ == 'aspen':
        plt.title('ASPEN')
    else:
        plt.title('Newton')
    t = solver.t
    x_grid, t_grid = np.meshgrid(x, t)
    if type(solver.nl_solver).__name__ == 'aspen':
        if not(solver.dyn_bd):
            borders = [domain[0] for domain in solver.nl_solver.partion[1:]]
            for bd in borders:
                plt.axvline(bd/solver.Nx, linestyle = '--', color='k')
        else:
            for i in range(solver.freq_ch):
                for bd in solver.timelog.borders[:, i]:
                    step = 1/(solver.freq_ch)
                    plt.plot([bd/solver.Nx, bd/solver.Nx], 
                        [step*i, step*(i+1)],
                        linestyle = '--', color='k')
            for i in range(solver.freq_ch+1):
                step = 1/(solver.freq_ch)
                plt.axhline(step*i, linestyle = '--', color='k')

    cs = plt.contourf(x_grid, t_grid, solver.X.T, cmap='RdBu_r')
    cbar = plt.colorbar(cs)
    if save != None:
        plt.savefig('data/'+save, dpi=400)

# ploter func for mean local iteration of provided solver with ASPEN
# optional: save = desired filepath to save plot
def bar_loc(solver, Nd, save = None):

    plt.title('mean iters')
    plt.ylim([0, 6])
    plt.bar(np.arange(1, Nd+1), np.mean(solver.timelog.domain_iters/solver.timelog.aspen_iters, axis = 1))
    if save != None:
        plt.savefig('data/' + save, dpi=400)
    plt.show()

# ploter func for local iteration on step = 'step' of provided solver with ASPEN
# optional: save = desired filepath to save plot
def bar_loc_step(solver, Nd, step, save = None):
    plt.title('liters on step = {}'.format(step))
    plt.ylim([0, 6])
    plt.bar(np.arange(1, Nd+1), solver.timelog.domain_iters[:, step-1]/solver.timelog.aspen_iters[step-1])
    if save != None:
        plt.savefig(save, dpi=400)
    plt.show()

# ploter func wich comapres number of iterations of several solvers
# uses in case of ASPEN: global iteration, Newton: iterations
# optional: save = desired filepath to save plot
def compare(list_of_solvers, list_of_names, save=None):
    title = ''

    for i in range(len(list_of_names)-1):
        title += list_of_names[i] + ' vs '
    title += list_of_names[-1]
    plt.title(title)
    print('--time comparision--')

    step = np.arange(1, list_of_solvers[0].Nt+1)
    for solver, name in zip(list_of_solvers, list_of_names):
        if type(solver.nl_solver).__name__ == 'aspen':
            plt.plot(step, solver.timelog.aspen_iters, label=name)
        elif  type(solver.nl_solver).__name__ == 'newton':
            plt.plot(step, solver.timelog.kn, label=name)
        print(name+ ' :', tim(solver.timelog) )
    plt.legend()
    if save != None:
        plt.savefig('data/'+save, dpi=400)
    plt.show()