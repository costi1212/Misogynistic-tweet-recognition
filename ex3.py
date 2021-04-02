# In[3]:
ε=1E-6

import math
def functie(x):
    return (-1)/(math.factorial(x)*(x-1)*x)

resultat = 3
x=2
while (abs(resultat-math.e) > ε):
    resultat+= functie(x)
    x+= 1
print(f"Valoarea evaluata a lui e {resultat} cu abaterea de {ε} corespunde cu valoarea"
      f" nominala {math.e} dupa {x-1} iteratii" )
