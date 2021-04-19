# just a brunch of functions, that are usefull for code reduction
import numpy as np
import inspect
import matplotlib.pyplot as plt

def time(clas):   
    if type(clas).__name__ == 'aspen_log':
        t = clas.gb_res+clas.gb_jac \
      +clas.gb_lin+np.sum(clas.lc_res) \
      +np.sum(clas.lc_jac)+np.sum(clas.lc_lin)
    elif type(clas).__name__ == 'newton_log':
        t = clas.lin+clas.jac+clas.res
    
    return t

def decorator1(obj):
    def wraper(*args, **kwargs):
        print('test started')
        solver, X, message, time = obj(*args, **kwargs)
        print('verdict : ' + message)
        print('mean time : {}'.format(time))

        if type(solver.timelog).__name__ == 'newton_log':
            print('mean newton iterations: ', np.mean(solver.timelog.kn))
        elif type(solver.timelog).__name__ == 'aspen_log':
            print('mean aspen iterations: ', np.mean(solver.timelog.aspen_iters))

        return X, message, time
    return wraper

@decorator1
def calc(solver, x0, bd1, bd2, sample_size = 10):
    t = 0
    for k in range(1):
        solver.init_log()
        solver.setBoundary(bd1, bd2)
        solver.x0 = np.copy(x0)
        X, message = solver.solve()
        t += time(solver.timelog)
    t /= 1#sample_size
    return solver, X, message, t

def show_res(solver):
    x = np.linspace(0, 1, solver.param.Nx)
    plt.figure(figsize= (8, 6))
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Newton method')
    t = solver.t
    x_grid, t_grid = np.meshgrid(x, t)
    if type(solver.solver).__name__ == 'aspen':
        for bd in solver.solver.partion[1:-1]:
            plt.axvline(bd/solver.param.Nx, linestyle = '--', color='k')
        print('iters :', solver.timelog.domain_iters)
    cs = plt.contourf(x_grid, t_grid, solver.X.T, cmap='RdBu_r')
    cbar = plt.colorbar(cs)
    plt.show()

def compare(list_of_solvers, list_of_names):
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
        print(name+ ' :', time(solver.timelog) )
    plt.legend()
    plt.show()

def decorator2(obj):
    def wraper(start_bd, m_sim, metrics, l_solver):
        print('--before--')
        print(start_bd, '{:.2E}'.format(metrics(start_bd)))
        bd = obj(start_bd, m_sim, metrics, l_solver)
        print('--after--')
        opt = metrics(bd)
        print(bd, '{:.2E}'.format(opt))
        return bd, opt
    return wraper

@decorator2
def local_search(start_bd, m_sim, metrics, l_solver):
    borders = np.copy(start_bd)
    borders = l_solver.find(borders, m_sim, metrics, steps = 120)
    borders = l_solver.find(borders, m_sim, metrics, steps = 80, cl = False)
    borders = l_solver.find(borders, m_sim, metrics, steps = 120)

    return borders