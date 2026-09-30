import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from pathlib import Path

OUT = Path(__file__).with_name('project1_figures')
OUT.mkdir(exist_ok=True)


def cheb_diff_matrix(N):
    """Chebyshev-Lobatto nodes x_j=cos(pi*j/N) and first differentiation matrix."""
    if N == 0:
        return np.array([1.0]), np.array([[0.0]])
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[[0, -1]] = 2.0
    c *= (-1.0) ** j
    X = x[:, None]
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D -= np.diag(np.sum(D, axis=1))
    return x, D


def homogeneous_basis_operator(N, alpha, a4, a2, a1, a0):
    """
    Build collocation matrix for u=(1-z^2)^2*q(z), q in P_{N-4},
    at z_j=cos(pi*j/N), j=2,...,N-2.
    Operator is a4*d4 + a2*d2 + a1*d1 + a0.
    """
    z = np.cos(np.pi * np.arange(N + 1) / N)
    zi = z[2:-2]
    qdeg = N - 4
    nb = qdeg + 1

    # (1-z^2)^2 in Chebyshev coefficients.
    one_minus_z2 = np.array([0.5, 0.0, -0.5])
    factor = C.chebmul(one_minus_z2, one_minus_z2)

    A = np.zeros((zi.size, nb))
    basis_coeffs = []
    for k in range(nb):
        Tk = np.zeros(k + 1)
        Tk[-1] = 1.0
        b = C.chebmul(factor, Tk)
        basis_coeffs.append(b)
        b1 = C.chebder(b, 1)
        b2 = C.chebder(b, 2)
        b4 = C.chebder(b, 4)
        Lb = a0 * b
        if a1 != 0:
            Lb = C.chebadd(Lb, a1 * b1)
        if a2 != 0:
            Lb = C.chebadd(Lb, a2 * b2)
        if a4 != 0:
            Lb = C.chebadd(Lb, a4 * b4)
        A[:, k] = C.chebval(zi, Lb)
    return z, zi, A, factor


def solve_problem5(N):
    # u'''' - 4u'' + 3u = g, u=u'=0 at z=+-1.
    z, zi, A, factor = homogeneous_basis_operator(
        N, alpha=1.0, a4=1.0, a2=-4.0, a1=0.0, a0=3.0
    )
    g = (np.pi**4 + 4*np.pi**2 + 3.0) * np.cos(np.pi * zi) + 3.0
    q = np.linalg.solve(A, g)
    ucoef = C.chebmul(factor, q)
    return ucoef


def p6_lift_coeffs():
    """Chebyshev coefficients of the cubic boundary lift b(z) on z in [-1,1]."""
    # b(z) = p0(z)*1 + q0(z)*0 + p1(z)*(1/26) + q1(z)*(-25/676)
    # Expanded exactly: (625 z^3 - 25 z^2 - 1925 z + 1429)/2704.
    power_ascending = np.array([
        1429/2704,
        -1925/2704,
        -25/2704,
        625/2704,
    ])
    return C.poly2cheb(power_ascending)


def g6(x):
    return ((24 - 240*x**2 + 120*x**4) / (1 + x**2)**5
            + 1/(1 + x**2)
            - 2*x/(1 + x**2)**2)


def solve_problem6(N):
    # Map x in [0,5] to z=2x/5-1, so d/dx = alpha*d/dz.
    alpha = 2.0 / 5.0
    z, zi, A, factor = homogeneous_basis_operator(
        N, alpha=alpha, a4=alpha**4, a2=0.0, a1=alpha, a0=1.0
    )
    x_i = 2.5 * (zi + 1.0)
    G = g6(x_i)

    b = p6_lift_coeffs()
    b1 = C.chebder(b, 1)
    b4 = C.chebder(b, 4)
    Lb = C.chebadd(C.chebadd(alpha**4 * b4, alpha * b1), b)
    rhs = G - C.chebval(zi, Lb)

    q = np.linalg.solve(A, rhs)
    v = C.chebmul(factor, q)
    U = C.chebadd(v, b)
    return U


def exact5(x):
    return np.cos(np.pi*x) + 1.0


def exact6(x):
    return 1.0/(1.0 + x*x)


def eval5(N, x):
    return C.chebval(x, solve_problem5(N))


def eval6(N, x):
    z = 2.0*x/5.0 - 1.0
    return C.chebval(z, solve_problem6(N))


def make_figures():
    # Problem 5 solutions.
    xx = np.linspace(-1, 1, 2001)
    Ns5 = [8, 12, 16, 20, 22]
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(xx, exact5(xx), linewidth=2.4, label='exact')
    for N in Ns5:
        plt.plot(xx, eval5(N, xx), label=f'N={N}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Problem 5: exact and Chebyshev spectral solutions')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUT/'problem5_solutions.pdf')
    plt.savefig(OUT/'problem5_solutions.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for N in Ns5:
        err = np.abs(eval5(N, xx) - exact5(xx))
        plt.semilogy(xx, np.maximum(err, 1e-18), label=f'N={N}')
    plt.xlabel('x')
    plt.ylabel('absolute error')
    plt.title('Problem 5: pointwise error')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUT/'problem5_errors.pdf')
    plt.savefig(OUT/'problem5_errors.png', dpi=200)
    plt.close()

    # Problem 6 solutions.
    xx6 = np.linspace(0, 5, 3001)
    Ns6 = [16, 24, 32, 48, 64]
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(xx6, exact6(xx6), linewidth=2.4, label='exact')
    for N in Ns6:
        plt.plot(xx6, eval6(N, xx6), label=f'N={N}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Problem 6: exact and Chebyshev spectral solutions')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUT/'problem6_solutions.pdf')
    plt.savefig(OUT/'problem6_solutions.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for N in Ns6:
        err = np.abs(eval6(N, xx6) - exact6(xx6))
        plt.semilogy(xx6, np.maximum(err, 1e-18), label=f'N={N}')
    plt.xlabel('x')
    plt.ylabel('absolute error')
    plt.title('Problem 6: pointwise error')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUT/'problem6_errors.pdf')
    plt.savefig(OUT/'problem6_errors.png', dpi=200)
    plt.close()

    # Convergence summary.
    N5all = np.arange(8, 27, 2)
    e5=[]
    for N in N5all:
        e5.append(np.max(np.abs(eval5(int(N), xx) - exact5(xx))))
    N6all = np.arange(16, 81, 4)
    e6=[]
    for N in N6all:
        e6.append(np.max(np.abs(eval6(int(N), xx6) - exact6(xx6))))
    plt.figure(figsize=(7.2, 4.8))
    plt.semilogy(N5all, e5, marker='o', label='Problem 5')
    plt.semilogy(N6all, e6, marker='s', label='Problem 6')
    plt.axhline(np.finfo(float).eps, linestyle='--', label='machine epsilon')
    plt.xlabel('N (polynomial degree; N+1 Lobatto nodes)')
    plt.ylabel('max absolute error')
    plt.title('Spectral convergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT/'convergence.pdf')
    plt.savefig(OUT/'convergence.png', dpi=200)
    plt.close()

    return N5all, np.array(e5), N6all, np.array(e6)


def main():
    N5all,e5,N6all,e6 = make_figures()
    print('Problem 5 convergence')
    for N,e in zip(N5all,e5):
        print(f'N={N:3d}, max error={e:.6e}')
    print('\nProblem 6 convergence')
    for N,e in zip(N6all,e6):
        print(f'N={N:3d}, max error={e:.6e}')

    print('\nSelected thresholds:')
    for thresh in [1e-12, 1e-14, 2e-15, 1e-15]:
        p5 = next((int(N) for N,e in zip(N5all,e5) if e <= thresh), None)
        p6 = next((int(N) for N,e in zip(N6all,e6) if e <= thresh), None)
        print(f'error <= {thresh:.1e}: P5 N={p5}, P6 N={p6}')


if __name__ == '__main__':
    main()
