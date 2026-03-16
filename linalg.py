import math
def lu_decompose(A):
    n=len(A);L=[[0]*n for _ in range(n)];U=[r[:] for r in A]
    for i in range(n): L[i][i]=1
    for j in range(n):
        for i in range(j+1,n):
            if abs(U[j][j])<1e-12: continue
            f=U[i][j]/U[j][j];L[i][j]=f
            for k in range(n): U[i][k]-=f*U[j][k]
    return L,U

def solve_lu(L,U,b):
    n=len(b);y=[0]*n;x=[0]*n
    for i in range(n): y[i]=b[i]-sum(L[i][j]*y[j] for j in range(i))
    for i in range(n-1,-1,-1): x[i]=(y[i]-sum(U[i][j]*x[j] for j in range(i+1,n)))/U[i][i]
    return x

def power_iteration(A,iters=100):
    n=len(A);v=[1/math.sqrt(n)]*n
    for _ in range(iters):
        w=[sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
        norm=math.sqrt(sum(x**2 for x in w));v=[x/norm for x in w]
    eigenval=sum(sum(A[i][j]*v[j] for j in range(n))*v[i] for i in range(n))
    return eigenval,v

def demo():
    A=[[2,1,0],[1,3,1],[0,1,2]];L,U=lu_decompose(A)
    x=solve_lu(L,U,[1,2,3]);print(f"Solve Ax=[1,2,3]: x={[round(v,4) for v in x]}")
    ev,vec=power_iteration(A);print(f"Largest eigenvalue: {ev:.4f}")
if __name__=="__main__": demo()
