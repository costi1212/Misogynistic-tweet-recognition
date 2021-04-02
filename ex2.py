#Ex 2 partial

import math
resultat = []

el = 0.0
while True:
    try:
        el = float(input("Introduceti elementul initial al listei: "))
        if el > math.pi/2 or el < 0:
            print("Valoarea initiala trebuie sa fie cuprinsa intre 0 si pi/2")
        else:
            break
    except Exception as e:
        print("Valoarea introdusa nu este buna ", e)

resultat.append(el)
resultat.append(math.atan(-3*el+10))
i = 1
while abs(resultat[-1]-resultat[-2]) > 1E-7:
    resultat.append(math.atan(-3*resultat[-1]+10))
    i += 1
print(f"Sirul si-a atins convergenta dorita {1E-7} dupa {i} iteratii")
print(resultat)
print("%.14f" % float(1E-10))
