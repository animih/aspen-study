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
        solver, X, mes, t = obj(*args, **kwargs)
        print('verdict : ' + mes)
        print('mean time : {}'.format(t))

        if type(solver.timelog).__name__ == 'newton_log':
            print('mean newton iterations: ', np.mean(solver.timelog.kn))
        elif type(solver.timelog).__name__ == 'aspen_log':
            print('mean aspen iterations: ', np.mean(solver.timelog.aspen_iters))

        return X, mes, t
    return wraper

def test(solver, sample_size = 5):
    t = 0
    for k in range(sample_size):
        solver.init_log()
        t -= time.time()
        X, mes = solver.solve()
        t += time.time()
    t /= sample_size

    return solver, X, mes, t

def show_res(solver, save=None):
    x = np.linspace(0, 1, solver.param.Nx)
    plt.figure(figsize= (8, 6))
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Newton method')
    t = solver.t
    x_grid, t_grid = np.meshgrid(x, t)
    if type(solver.solver).__name__ == 'aspen':
        if not(solver.dyn_bd):
            for bd in solver.solver.partion[1:-1]:
                plt.axvline(bd/solver.param.Nx, linestyle = '--', color='k')
        else:
            for i in range(4):
                for bd in solver.timelog.borders[:, i]:
                    plt.plot([bd/solver.param.Nx, bd/solver.param.Nx], 
                        [0.25*i, 0.25*(i+1)],
                        linestyle = '--', color='k')
            for i in range(4):
                plt.axhline(0.25*i, linestyle = '--', color='k')

        print('iters :', solver.timelog.domain_iters)
    cs = plt.contourf(x_grid, t_grid, solver.X.T, cmap='RdBu_r')
    cbar = plt.colorbar(cs)
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
        plt.savefig(save, dpi=400)
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