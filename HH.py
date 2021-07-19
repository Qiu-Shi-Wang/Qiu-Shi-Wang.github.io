#Hochschild cohomology of a finite dimensional algebra over a field k.
#December 8 2020.
'''We will use numpy arrays as our main data type. They can be added elementwise, multiplied by a scalar, etc.'''

'''Say we have a finite dimensional algebra A; we do not yet require it to be graded or Frobenius. Choosing a basis, we 
represent it by a square matrix, the multiplication matrix on the basis elements. Each of the entries of the matrix is
a coordinate vector representing an element of A.'''

'''Each element of CC^n(A) is a homomorphism from A tensor itself n times to A. We will represent this function by 
a Python dictionary taking a basis element of A^otimes n to a vector in A. In this sense, letting d=dim(A), CC^n(A) 
has k-dimension d^{n+1}'''
import numpy as np
import math
from itertools import product
from copy import copy
import time

def basislist(d,n): #creates a list of all basis element pairs (1d arrays)
    if n==0:
        return [(0,)]
    else:
        L=[]
        for i in range(d):
            L.append(i)
        A=np.array(L)
        return list(product(L,repeat=n))

def algbasis(d):
    B=[]
    for i in range(d):
        v=[]
        for j in range(d):
            if j==i:
                v.append(1)
            else:
                v.append(0)
        B.append(v)
    return B
            
def dim(A):
    return len(A)

def algprod(x,y,A): #elements of the algebra are 1-dimensional numpy arrays
    arr=np.array(emptylist(dim(A)))
    for i in range(len(x)):
        for j in range(len(y)):
            #print(x,y)
            arr=arr+A[i,j]*x[i]*y[j]
    return arr

def component(x,i,dim): #x is an basis element of A tensor itself n times, and we want the ith element
    Index=x[i]
    comp=[]
    for j in range(dim):
        if j==Index:
            comp.append(1)
        else:
            comp.append(0)
    return np.array(comp)

def truncate(x,n,A):
    if n==1:
        return tuple(list(x)[1:])
    elif n==len(x)+1:
        return tuple(list(x)[:-1])
    else:
        lincomb=[]
        L=list(x)[:(n-1)]+list(x)[n:]
        C=component(x,n-2,dim(A))
        D=component(x,n-1,dim(A))
        P=algprod(C,D,A)
        for i in range(len(P)):
            Lcopy=copy(L)
            Lcopy[n-2]=i
            lincomb.append((tuple(Lcopy),P[i]))
        return lincomb
    

def diff(A,n,f): #this is the Hochschild differential
    L=basislist(dim(A),n+1)
    if n==0:#HH0 case
        df={}
        for i in np.identity(dim(A)):
            #print(f)
            df[tuple(i)]=algprod(i,f[(0,)],A)-algprod(f[(0,)],i,A)
    else:
        df={}
        for x in L:
            
            #print("x is " +str(x)+ " and f is " + str(f))
            #print("To multiply for first term: " + str((component(x,0,dim(A)),f[truncate(x,1,A)])))
            df[x]=algprod(component(x,0,dim(A)),f[truncate(x,1,A)],A)
            
            #print("first term" +str(df[x]))
            
            for i in range(1,n+1):
                for vec in truncate(x,i+1,A):
                    
                    #print("middle term" + str((-1)**i*vec[1]*f[vec[0]]))
                    
                    df[x]+=(-1)**i*vec[1]*f[vec[0]]
            
            #print("To multiply for last term: " +str((f[truncate(x,n+2,A)],component(x,n,dim(A)))))
            
            df[x]+=(-1)**(n+1)*algprod(f[truncate(x,n+2,A)], component(x,n,dim(A)),A)
            
            #print("last term" + str((-1)**(n+1)*algprod(f[truncate(x,n+2,A)], component(x,n,dim(A)),A)))
            #print("total" +str(df[x]))
            #print("\n")
            
    return df

def vector(f): #takes a homomorphism (dictionary) and makes it into a vector
    v=[]
    for i in f:
        v+=list(f[i])
    return v

def dictio(v,d,n): #does the inverse of vector()
    D={}
    if d==1:
        if n==0 or n==1:
            D[(0,)]=np.array([v[0]])
        else:
            D[tuple(emptylist(n))]=np.array([v[0]])
        return D
    else:
        #print("len(v) is " +str(len(v)))
        #print("d is " + str(d))
        N=round(math.log(len(v),d))-1
        #print("N is " +str(N))
        B=basislist(d,N)
        for i in range(len(v)//d):
            D[B[i]]=np.array(v[i*d:(i+1)*d])
        return D

def emptylist(n):
    L=[]
    for i in range(n):
        L.append(0)
    return L

def basisCC(d,n):
    return np.identity(d**(n+1))

def nullityd(A,n):
    L=[]
    for i in basisCC(dim(A),n):
        L.append(vector(diff(A,n,dictio(i,dim(A),n))))

    M=np.array(L)
    r=np.linalg.matrix_rank(M)
    return dim(A)**(n+1)-r
    
def dimHH(A,n):
    t=time.time()
    if n==0:
        HH=nullityd(A,0)
        print(str(time.time()-t)+ " seconds" )
        return HH
    else:
        #print(nullityd(A,n), nullityd(A,n-1), dim(A)**n)
        HH=nullityd(A,n)+nullityd(A,n-1)-dim(A)**n
        print(str(time.time()-t)+ " seconds" )
        return HH


'''Graded Hochschild cohomology'''



A=np.array([[[1]]]) #trivial algebra isomorphic to the field
B=np.array([[[1,0],[0,1]],[[0,1],[0,0]]]) #k[x]/x^2
C=np.array([[[1,0,0],[0,1,0],[0,0,1]],[[0,1,0],[0,0,1],[0,0,0]],[[0,0,1],[0,0,0],[0,0,0]]]) #k[x]/x^3


#for i in range(10):
    #print("dim HH^"+str(i)+"(A)=" + str(dimHH(C,i)))
