#!/usr/bin/env python3
"""Linear algebra: Gaussian elimination, eigenvalues (power iteration), SVD-lite."""
import sys,math

def gauss_solve(A,b):
    n=len(b);aug=[A[i]+[b[i]]for i in range(n)]
    for i in range(n):
        mx=max(range(i,n),key=lambda r:abs(aug[r][i]))
        aug[i],aug[mx]=aug[mx],aug[i]
        for j in range(i+1,n):
            f=aug[j][i]/aug[i][i]
            aug[j]=[aug[j][k]-f*aug[i][k]for k in range(n+1)]
    x=[0]*n
    for i in range(n-1,-1,-1):
        x[i]=(aug[i][n]-sum(aug[i][j]*x[j]for j in range(i+1,n)))/aug[i][i]
    return x

def power_iteration(A,iters=100):
    n=len(A);v=[1/math.sqrt(n)]*n
    for _ in range(iters):
        Av=[sum(A[i][j]*v[j]for j in range(n))for i in range(n)]
        norm=math.sqrt(sum(x**2 for x in Av))
        v=[x/norm for x in Av]
    eigenvalue=sum(sum(A[i][j]*v[j]for j in range(n))*v[i]for i in range(n))
    return eigenvalue,v

def gram_schmidt(V):
    U=[]
    for v in V:
        u=list(v)
        for prev in U:
            proj=sum(a*b for a,b in zip(u,prev))/sum(a**2 for a in prev)
            u=[a-proj*b for a,b in zip(u,prev)]
        norm=math.sqrt(sum(x**2 for x in u))
        if norm>1e-10:U.append([x/norm for x in u])
    return U

def main():
    if len(sys.argv)>1 and sys.argv[1]=="--test":
        # Solve 2x+y=5, x+3y=7
        x=gauss_solve([[2,1],[1,3]],[5,7])
        assert abs(x[0]-1.6)<1e-10 and abs(x[1]-1.8)<1e-10
        # Power iteration on symmetric matrix
        ev,vec=power_iteration([[2,1],[1,2]])
        assert abs(ev-3)<0.1  # dominant eigenvalue is 3
        # Gram-Schmidt
        orth=gram_schmidt([[1,1],[1,0]])
        assert abs(sum(orth[0][i]*orth[1][i]for i in range(2)))<1e-10  # orthogonal
        print("All tests passed!")
    else:
        x=gauss_solve([[3,2,1],[1,1,1],[2,1,3]],[10,6,13])
        print(f"Solution: {[f'{v:.2f}' for v in x]}")
if __name__=="__main__":main()
