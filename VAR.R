
#katalog roboczy
setwd("C:/sciezka")

#import danych
#####
df = data.frame(N = c(1:79),
                bezrobocie = danebezrobocie1[9:87,],
                danemieszkania=danemieszkania[87:9,3],
                wig20 = danewig[44:122,"Zamkniecie"],
                WIBOR = danewibor[29:107,5],
                PKB = danePKB
)
df

#####

model<-read.csv("dane_w_excelu.csv",sep=";")
model
dane<-model
danelog<-log(dane[,2:8])
dane<-danelog

tytuly <- c("Dochod BP",
            "Eks towarów",
            "Należności ",
            "P wynagrodzenie"
) 
#nazwy zmiennych, t(dane)=transpozycja danych


#zapis danych w postaci szeregu czasowego
#?ts

tsdane <-ts(dane,start=2009,end=2019 ,frequency=4)
tsdane<-tsdane[,-8]
tsdane<-tsdane[,-7]

tsdane
#liczba op?znien do ACF i PACF


dane<-as.matrix(dane[,2:5])
colnames(dane)<-tytuly

#Wybrane dane !!!
tsdane
dane

# Rysunki dla poziomów !!
par(mfrow=c(1,1))
par(mfrow=c(4, 3))

n=ncol(dane) #liczba analizowanych zmiennych
T_obs=nrow(dane) #liczba zebranych obserwacji
op=ceiling(T_obs/4)

i=1
while (i<=4) {
  plot.ts(tsdane[,i+2],main=tytuly[i],ylab="")#szereg
  acf(dane[,i],lag.max=op,main=tytuly[i],ylab="")#ACF
  pacf(dane[,i],lag.max=op,main=tytuly[i],ylab="")#PACF
  i=i+1
}

# Rysunki dla przyrostów !!

par(mfrow=c(1,1))
par(mfrow=c(4, 3))

n=ncol(dane) #liczba analizowanych zmiennych
T_obs=nrow(dane) #liczba zebranych obserwacji
op=ceiling(T_obs/4)

tsdd=diff(tsdane)
dd=diff(dane)#przyrosty
i=1
while (i<=4) {
  plot.ts(tsdd[,i+2],main=tytuly[i],ylab="")#szereg
  acf(dd[,i],lag.max=op,main=tytuly[i],ylab="")#ACF
  pacf(dd[,i],lag.max=op,main=tytuly[i],ylab="")#PACF
  i=i+1
}


##############  estymacja VAR(k) #########################


#kryteria informacyjne z pakietu vars
#Wybieram możliwe opóźnienia z kryteriów informacyjnych
VARselect(dane,lag.max=5,type="both",season=4)

#Dla wybranego stopnia opóźinienia badam stabilność
VAR4L=VAR(dane,p=5,type="both",season=4)

roots(VAR4L)

par(mfrow=c(1,1))
plot(roots(VAR4L,F),xlim=c(-1,1),ylim=c(-1,1),asp=1,
     main="Wykres dla 5 stopni opóźnień dla trendu liniowego",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)

#zmienia stałą deteministyczną
VAR5c=VAR(dane,p=5,type="const",season=4)

roots(VAR5c)

par(mfrow=c(1,1))
plot(roots(VAR5c,F),xlim=c(-1.3,1.3),ylim=c(-1,1),asp=1,
     main="Wykres dla 5 stopni opóźnień dla modelu ze stałą",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)

#Wybieram teraz dla stopnie opóźnien, niebiore 1 bo jak rpzejdena przyrosty
#zostani mi wtedy biały szum
VAR2L=VAR(dane,p=2,type="both",season=4)

par(mfrow=c(1,1))
plot(roots(VAR2L,F),xlim=c(-1.1,1.1),ylim=c(-1,1),asp=1,
     main="Wykres dla 2 stopni opóźnień dla trendu liniowego",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)

VAR2C=VAR(dane,p=2,type="const",season=4)

par(mfrow=c(1,1))
plot(roots(VAR2C,F),xlim=c(-1.1,1.1),ylim=c(-1,1),asp=1,
     main="Wykres dla 2 stopni opóźnień dla modelu ze stałą",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)

# Stacjonarnosc dla przyrostów
VAR4L=VAR(dd,p=4,type="const",season=4)

roots(VAR4L)

par(mfrow=c(1,1))
plot(roots(VAR4L,F),xlim=c(-1,1),ylim=c(-1,1),asp=1,
     main="Wykres dla 4 stopni opóźnień dla modelu ze stałą",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)

VAR5c=VAR(dd,p=4,type="none",season=4)

roots(VAR5c)

par(mfrow=c(1,1))
plot(roots(VAR5c,F),xlim=c(-1.3,1.3),ylim=c(-1,1),asp=1,
     main="Wykres dla 4 stopni opóźnień dla modelu bez składowej deterministycznej",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)


VAR1C=VAR(dd,p=1,type="const",season=4)

par(mfrow=c(1,1))
plot(roots(VAR1C,F),xlim=c(-1.1,1.1),ylim=c(-1,1),asp=1,
     main="Wykres dla 1 stopnia opóźnień dla modelus ze stałą",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)

VAR1B=VAR(dd,p=1,type="none",season=4)

par(mfrow=c(1,1))
plot(roots(VAR1B,F),xlim=c(-1.1,1.1),ylim=c(-1,1),asp=1,
     main="Wykres dla 1 stopnia opóźnień dla modelu bez składowej deterministycznej",xlab="",ylab="")#wykres wartosci wlasnych macierzy stowarzyszonej
draw.circle(0,0,1)


#####
#Najwyzszy modul >0.9 -> na wszelki wypadek przechodzimy do analizy przyrostow
#VAR(4) dla poziomow -> VAR(3) dla przyrostow
#trend liniowy (a+bt) dla poziomow -> stala dla przyrostow




#test VAR(2) vs VAR(3)

#H0: VAR(2) vs H1:VAR(3) (zapis ukladu hipotez za pomoca testowanych modeli)
#H0: A3=0  vs H1: A3<>0 (zapis ukladu hipotez za pomoca testowanych restrykcji zerowych, "<>" - rozne)

#VAR2p=VAR(dd[2:nrow(dd),],p=1,type="const",season=4)#szacowanie VAR(2)
#roots(VAR2p) #warunek stabilnosci, najwyzszy modul <0.9 -> bezpiecznie


lemp=2*(logLik(VAR1C)-logLik(VAR1B))#empiryczna wartosc statystyki testowej
al=0.05#poziom istotnosci
df=n^2#liczba stopni swobody
qchisq(1-al,df)#kwantyl z rozkladu chi^2 o 16 stopniach swobody 
#(16 - poniewaz n=4, a wiec macierz A3 ma 6 elementow i sprawdzamy 
#ich laczna istotnosc)
p_value=pchisq(lemp, df, lower.tail = F)
p_value



#Statystyla neleży do przedziału krytycznego wiec wybieramy VAR1B



############## testowanie zalozen modelu VAR  #########################

#model VAR(1) bez skladowej deterministycznej dla przyrostów

par(mfrow=c(1,1))
plot(VAR1B)

k=1#liczba opoznien VAR
Tp=nrow(dd)-k   #liczba modelowanych obserwacji

h=ceiling(Tp/4)#liczba testowanych macierzy korelacji
serial.test(VAR1B,lags.pt=h,type="PT.adjusted")#test walizkowy z korekta na mala probe

par(mfrow=c(1,1))
plot(serial.test(VAR1C,lags.pt=h,type="PT.adjusted"))


serial.test(VAR1C,lags.bg=2,type="BG") #mnozniki Lagrangera

normality.test(VAR1C,F)

arch.test(VAR1C,lags.multi= 1)

#!!!!!! tylko jedna wartość 0.01
h=1
while (h<=5) {
  print(serial.test(VAR1B,lags.bg=h,type="BG"))
  h=h+1
}

#Rezultaty testu Breuscha i Godfreya wskazuja, ze w resztach zostala autokorelacj 
# ponieważ wartości są w tysięcznych->
# -> dokladamy 1 opoznienie -> VAR(2)

#dodaje opoznienia. Zauwazam, ze nic to nie daje
#model VAR(2) dla przyrostow bez skladnika determnistycznego
VAR2B=VAR(dd,p=2,type="none",season=4)
roots(VAR2B)

par(mfrow=c(1,1))
plot(VAR2B)

k=2#liczba opoznien VAR
Tp=nrow(dd)-k#liczba modelowanych obserwacji

h=ceiling(Tp/4)#liczba testowanych macierzy korelacji
serial.test(VAR2B,lags.pt=h,type="PT.adjusted")#test walizkowy z korekta na mala probe
serial.test(VAR2B,lags.bg=1,type="BG")#test walizkowy z korekta na mala probe

par(mfrow=c(1,1))
plot(serial.test(VAR2B,lags.pt=h,type="PT.adjusted"))

#sekwencja testow Breuscha i Godfreya 
h=1

while (h<=6) {
  print(h)
  print(serial.test(VAR1B,lags.bg=h,type="BG"))
  h=h+1
}
#Koniec

#test normlanosci reszt w modelu VAR(4) dla przyrostow
normality.test(VAR1B,F)


#test efektu ARCH
h=1
while (h<=5) {
  print(arch.test(VAR1B, lags.multi = h))
  h=h+1
}

arch.test(VAR2B, lags.multi=4, lags.single = 6, F)

#punktowe oceny paramerow modelu VAR(4) dla przyrostow
coef(VAR2B)

#model VAR(4) bez trendu i stalej dla przyrost?w
VAR2B_bd=VAR(dd,p=4,type="none",season=4)
roots(VAR2B_bd)

par(mfrow=c(1,1))
plot(roots(VAR2B_bd,F),xlim=c(-1,1),ylim=c(-1,1),asp=1)
draw.circle(0,0,1)
#model VAR(2) bez trendu i stalej dla przyrost?w nie spelnia warunku stabilnosci


#Wyznaczam

VAR1C
m <- ca.jo(dane,type = "trace",ecdet = "trend", K = 2, spec = "transitory", season = 4)
summary(m)

#Wybieram 2

#Weights W oznaczaja współczynniki alfa. Korekta bledu nastepuje gdy
#mamy przeciwne znaki

m1 <- ca.jo(dane,type = "eigen",ecdet = "none", K = 2, spec = "transitory", season = 4)
summary(m1)
#DLa drugiej relacji mam trzy korekty błędów

vec.r2 <- cajorls(m,r=2)
?cajorls
summary(vec.r2$rlm)
#Dla Dochocow BP otrzymujemy istotne, Esk. towarów nieistotny, 
#Nalezności ogółem istotne dostosowania, przyszle wynagrodzenia nieistotne
#dwie zmienne dostosowuja sie do dwoch relacji



wektory <- vec.r2$beta



Z1=dane[5:nrow(dane),]
Z1=cbind(Z1,seq(0,nrow(Z1)-1,1))

odchylenia <- Z1%*%wektory


ts_odch<-ts(odchylenia,start=c(2002, 1), frequency=4)

par(mfrow=c(1,1))
plot.ts(ts_odch)


B <- matrix(c(1.00000000, 1.0000000, -2.07133326, 6.4587693, -0.45098166, -6.9808011, 0.13994817,  0.6725278, 0.03670909,  0.1258719), nrow = 5, ncol = 2, byrow = T)
od_B <- Z1%*%B
ts_od_B<-ts(od_B,start=c(2001, 4), frequency=4)

par(mfrow=c(1,1))
plot.ts(ts_od_B)

plot.ts(cbind(ts_odch,ts_od_B))

PB <- B%*%solve(crossprod(B,B))%*%t(B)

PW <- wektory%*%solve(crossprod(wektory,wektory))%*%t(wektory)

PB-PW

######## wykorzystanie oszacowanego modelu ################

?predict

?fanchart

?irf

?fevd

tytuly <- c("Dochod BP",
            "Eks towarów",
            "Należności ",
            "P wynagrodzenie"
) 

plot(irf(VAR1C,response=c("Dochod.BP"),n.ahead = 40, boot = T))

  
plot(irf(VAR1C,c("Eks.towarów","Należności.","P.wynagrodzenie"),
         n.ahead = 40, boot = T))


?irf
plot(irf(VAR4p, response="r",  n.ahead = 20, boot = T))

?ur.df


0. Jak weryfikujemy wiarygodność opinii eksperckich SRO (outliery, sprzeczne oceny, brakujące pola)?

1. Dlaczego kategorie z < 10 obserwacjami dostają Risk 1:200 = 0, skoro istnieją realne straty? Zerowanie VaR dla kategorii, która po „pożyczce” wciąż ma < 10 obserwacji. Ile kategorii ma Risk 1:200 = 0? CO jeśli te straty zaczęły pojawiać się w ostatnich miesiącach i będą pojawiać się dalej. Wtedy w żaden sposób ich nie uwzględniamy.

2. Jeśli weźmiemy dane dla P z P życie i później bierzemy te same dane w P życie. Czy tutaj nie będziemy mieli zbyt dużej korelacji między tymi kolumnami?
Jak uzasadniono statystycznie, że profile strat obu spółek są wymienne? | Czy podczas agregacji grupowej nie korelujecie w ten sposób zdarzeń 1-do-1?

3. Ile realnie punktów wpada w ogon każdej kategorii?. Bo jeśli weźmiemy 32 obserwacje i 2 będą w ogonie, to dopasowujemy rozkład do 2 punktów. Ile punktów nad progiem ma każda kategoria? 

4. Czy testowano alternatywne rozkłady ciężkoogonowe (Weibull, Gamma, Burr) dla strat? 
Czy przeprowadzono testy wskazujące, że lognormal będzie odpowiedni.

6. freq_var ≤ freq_mean × 1.10  10 % to arbitralny próg. Dla ilu z M = 100 bootstrapów kończy z Poissonem, a ilu z NB?


4. Wielkość każdej próbki bywa bardzo różna; może zdarzyć się, że w 60 losowych miesiącach nie trafi żadna strata wektor pusty, a mimo to dalej dopasowujemy log-normal. Losowe próbki 60 m-cy mogą mieć 0–2 zdarzeń; dopasowanie LN do takiej próbki.
Jeśli warunek jest wieksze niż 10 a to jest we wszystkich miesiącach (czyli 120 miesiącach) to może się zdarzyć, że nie mamy nic.

5. Jeśli curr_xi ma 0 obserwacji, to log-normal się dopasuje. log(0) → -inf; MLE zwróci nan, a ponieważ wyjątki są wyciszone, lgnm_ksz=0, lgnm_skal=0 → dalsza symulacja generuje same zera zamiast strat. warnings.filterwarnings("ignore", category=RuntimeWarning)


7. par_p = 0.1 czy było to testowane.

8. Tylko log-Normal poniżej progu; testowano Weibull/Gamma? Jeżeli prawdziwe straty są asymetryczne (Weibull) lub z częstymi małymi outliers (Gamma), LN może dać za gruby/za cienki body. Czy wykonano GOF-test (KS/AD) porównujący LN, Weibull, Gamma? | Rozszerzyć fitting: pick best AIC/BIC lub mixture

10. Jaki jest błąd Monte-Carlo VaR(99.5 %) przy zmianie seeda RNG (np. 500 powtórzeń) – czy mieści się w granicach ± 5 %?

11. Dlaczego korzystamy wyłącznie z kopuły t-Studenta/Gaussa z jednym ν dla wszystkich par – rozważano vine-copul. Czy wykonywali państwo testy dopasowania kopuli? W jaki sposób uzasadniamy, że to właśnie tStudenta jest (jak rozumiem) najlepszym wyborem.

12. Czy przeprowadzano testy wrażliwości na macierz Korelacji Spermana?

13. Jak w modelu rozwiązano następującą sytuację: dla kategorii z liczbą obserwacji < 10 wchodzi ona do kalibracji macierzy korelacji (kopuła „widzi” kolumnę), natomiast moduł symulacji zwraca pusty wektor strat. Co robi algorytm w takim momencie?

14. W dokumentacji (§ 3.4.1.2 – 3.4.2) przewidziano, że kategorie z liczbą obserwacji < min_obs mają zostać zsumowane w „sztuczny koszyk 0”, który jest komonotoniczny z ryzykami z Mᶜ, a dopiero następnie uwzględniony przy kalibracji macierzy Σ i liczby stopni swobody kopuły t-Studenta. W aktualnym kodzie CopulaJoin nie znajduję tworzenia takiego koszyka ani zastąpienia brakujących kolumn korelacją = 1; macierz Spearmana jest budowana na wszystkich oryginalnych kolumnach, a ujemne korelacje są jedynie obcinane do zera. Czy planujecie (lub gdzie w innym miejscu implementacji) dodać etap tworzenia komonotonicznego koszyka 0 dla kategorii z małą liczbą zdarzeń oraz kalibrację kopuły wyłącznie na zbiorze {0} ∪ M, aby spełnić wymogi § 3.4?
