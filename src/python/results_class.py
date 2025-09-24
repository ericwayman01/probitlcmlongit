# like a C-language struct
class Results():
    def __init__(self, beta=None,
                 gammas=dict(), kappas=dict(), my_lambda=None,
                 Rmat=None, xi=None, omega=None):
        self.beta = beta
        self.gammas = gammas
        self.kappas = kappas
        self.my_lambda = my_lambda
        self.Rmat = Rmat
        self.xi = xi
        self.omega = omega
