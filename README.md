# EVO projekt 
## Implementace Graph Coloring Problému za pomocí GA
> @autor Tadeáš Kachyňa, <xkachy00@stud.fit.vutbr.cz>

> @datum 08/05/2022

> @source projekt.py


Zadáním projektu bylo provést implementaci genetického algoritmu řešící graph coloring problem. Cílem bylo uskutečnit nekolik experimentů. Výsledky z nich jsou dostupné v souboru statistiky.xlsx a taktéž v prezentaci k obhajobě. V projektu jsem implementoval několik variant dvou genetických operátorů - mutace(3) a křížení(2). Jedno z křížení je reprezentována jednoduchým algoritmem ve kterém se náhodně určí bod rozdělení, následně se tyto dvě poloviny v chromozomech vymění. Druhé křížení je již o něco pokročilejší a mění prvky jen  u kterých je fitness reprezentována hodnotou 1. Mutace je reprezentována ve třech variantách. První z nich přiřazuje náhodně jakoukoliv barvu, druhá dbá na sousední uzly a přirazuje jen barvy, aby nedošlo k její následné kolizi. Třetí varianta přiřazuje barvu na základě výskytu všech barev v daném chromozomu. Nejrpve se spočítá jejich výskyt, následně se vezme polovina s nižším výskytem a náhodně se jedna barva genu přiřadí.

## Spuštětní projektu
```
$ python3 projekt.py
```

### Volitelné argumenty
Argumenty lze libovolně kombinovat.
```
--size - velikost matice (size x size) reprezentující graf
--colors - počáteční počet barev, kterým se graf začne obarvovat, pokud argument nebude zadán, bude vypočítána horní hranice barev
--graph - pokud je zadána hodnota "yes" (jiné hodnoty jsou ignorovány) poskytne algoritmus graf fitness funkce pro každý počet barev, který je schopen graf obarvit
--file - výstup bude přesměrován do souboru stats.txt
```

Hodnocení 21 bodů z 22
