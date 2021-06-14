# just a brunch of functions, that are usefull for code reduction
import numpy as np
import inspect
import matplotlib.pyplot as plt
import time

def tim(clas):   
    if type(clas).__name__ == 'aspen_log':
        t = clas.gb_res+clas.gb_jac \
      +clas.gb_lin+np.sum(clas.lc_res) \
      +np.sum(clas.lc_jac)+np.sum(clas.lc_lin)
    elif type(clas).__name__ == 'newton_log':
        t = clas.lin+clas.jac+clas.res
    
    return t

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

def show_res(solver, save=None):
    x = np.linspace(0, 1, solver.param.Nx)
    plt.figure(figsize= (8, 6))
    plt.xlabel('x')
    plt.ylabel('t')
    if type(solver.solver).__name__ == 'aspen':
        plt.title('ASPEN')
    else:
        plt.title('Newton')
    t = solver.t
    x_grid, t_grid = np.meshgrid(x, t)
    if type(solver.solver).__name__ == 'aspen':
        if not(solver.dyn_bd):
            for bd in solver.solver.partion[1:-1]:
                plt.axvline(bd/solver.param.Nx, linestyle = '--', color='k')
        else:
            for i in range(solver.freq_ch+1):
                for bd in solver.timelog.borders[:, i]:
                    step = 1/(solver.freq_ch+1)
                    plt.plot([bd/solver.param.Nx, bd/solver.param.Nx], 
                        [step*i, step*(i+1)],
                        linestyle = '--', color='k')
            for i in range(solver.freq_ch+1):
                step = 1/(solver.freq_ch+1)
                plt.axhline(step*i, linestyle = '--', color='k')

    cs = plt.contourf(x_grid, t_grid, solver.X.T, cmap='RdBu_r')
    cbar = plt.colorbar(cs)
    if save != None:
        plt.savefig('data/'+save, dpi=400)
    plt.show()

def bar_loc(solver, Nd, save = None):

    plt.title('mean iters')
    
    plt.bar(np.arange(1, Nd+1), np.mean(solver.timelog.domain_iters, axis = 1))
    if save != None:
        plt.savefig('data/' + save, dpi=400)
    plt.show()

def bar_loc_step(solver, Nd, step, save = None):
    plt.title('liters on step = {}'.format(step))
    plt.bar(np.arange(1, Nd+1), solver.timelog.domain_iters[:, step])
    if save != None:
        plt.savefig(save, dpi=400)
    plt.show()

def compare(list_of_solvers, list_of_names, save=None):
    title = ''

    for i in range(len(list_of_names)-1):
        title += list_of_names[i] + ' vs '
    title += list_of_names[-1]
    plt.title(title)
    print('--time comparision--')
    for solver, name in zip(list_of_solvers, list_of_names):
        if type(solver.solver).__name__ == 'aspen':
            plt.plot(solver.timelog.aspen_iters, label=name)
        elif  type(solver.solver).__name__ == 'newton':
            plt.plot(solver.timelog.kn, label=name)
        print(name+ ' :', tim(solver.timelog) )
    plt.legend()
    if save != None:
        plt.savefig('data/'+save, dpi=400)
    plt.show()


def decorator2(obj):
    def wraper(start_bd, metrics, l_solver):
        print('--before--')
        print(start_bd, '{:.2E}'.format(metrics(start_bd)))
        bd = obj(start_bd, metrics, l_solver)
        print('--after--')
        opt = metrics(bd)
        print(bd, '{:.2E}'.format(opt))
        return bd, opt
    return wraper

@decorator2
def local_search(start_bd, metrics, l_solver):
    borders = np.copy(start_bd)
    borders = l_solver.find(borders, metrics, steps = 220)
    borders = l_solver.find(borders, metrics, steps = 180, cl = False)
    borders = l_solver.find(borders, metrics, steps = 220)

    return borders