
#Ex 1 partial

ε = 10**(-7)

import math as mat

def functie(x):
    return (x**7+5*x+1)

def este_radacina(a,b):
    ax=functie(a)
    bx=functie(b)
    if (ax < 0 and bx > 0) or (ax > 0 and bx < 0):
        return True
    return False

a = -1
b = 0

while abs(a-b)>ε:
    c=(a+b)/2
    if este_radacina(a,c):
        b=c
    if este_radacina(c,b):
        a=c

print(f"radacina este {float((a+b)/2)} cu eroarea {ε}")

