from numpy import clip, empty, inf
def d_positive (B, separate=True) :
    Bp = clip(B, 0, inf); Bn = clip(B, -inf, 0)
    if separate :
        return Bp, Bn
    else :
        n,m = B.shape
        ret = empty((2*n,2*m))
        ret[:n,:m] = Bp; ret[n:,m:] = Bp
        ret[:n,m:] = Bn; ret[n:,:m] = Bn
        return ret