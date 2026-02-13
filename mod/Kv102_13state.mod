NEURON {
SUFFIX Kv102_13
USEION k READ ek WRITE ik
RANGE g, gbar
}

UNITS { (mV) = (millivolt) }

PARAMETER {
Hkc=-0.23007 (/degC)
Hki=0.73963 (/degC)
Hkn=0.1194 (/degC)
Hri=-0.015537 (/degC)
Hric_n=-0.33936 (/degC)
Hrn=0.23669 (/degC)
Hvc=1.0 (/degC)
Hvic_c=-0.90905 (/degC)
Hvic_n=0.94289 (/degC)
Hvn_ic=1.0 (/degC)
Ri=4.6749 ()
Rn=0.059481 ()
Ro=10.0 ()
Vc= -6.2187 (mV)
Zc= 0.38395 ()
kc= 0.99997 (/ms)
ki= 1.0 (/ms)
kn= 1.5658 (/ms)
ko= 0.0011199 (/ms)
ric_c=0.619 ()
ric_n=129.9917 ()
vc_ic=10.0 ()
vic_c=0.064507 ()
vic_n=3.6886 ()
vn_ic=0.1 ()
gbar = 36     (millimho/cm2)
}

ASSIGNED {
v    (mV)
ek   (mV)
g    (millimho/cm2)
ik   (milliamp/cm2)
kco  (/ms)
koc  (/ms)
}

STATE { C4 C3 C2 C1 C0 OS IC1 C4IC1 C3IC1 C2IC1 C1IC1 IN INC1 }

LOCAL F, A,rc_ic,rn_ic,kcf_ic,Hrn_ic

BREAKPOINT {
SOLVE kin METHOD sparse
g = gbar*OS
ik = g*(v - ek)*(1e-3)
}

INITIAL { 
F= 96.485(joule/mV)
A= 8.134(joule/degC)*(celsius+273.15)
rc_ic=ric_c
rn_ic=ric_n
kcf_ic=Ro
Hrn_ic=Hric_n
SOLVE kin STEADYSTATE sparse 
}

KINETIC kin {
rates(v)
~ C4 <-> C3     (4*kco, 1*koc)
~ C3 <-> C2     (3*kco, 2*koc)
~ C2 <-> C1     (2*kco, 3*koc)
~ C1 <-> C0     (1*kco, 4*koc)
~ C4IC1 <-> C3IC1  (4*vc_ic*kco,1*rc_ic*vc_ic*koc)
~ C3IC1 <-> C2IC1  (3*vc_ic*kco,2*rc_ic*vc_ic*koc)
~ C2IC1 <-> C1IC1  (2*vc_ic*kco,3*rc_ic*vc_ic*koc)
~ C1IC1 <-> IC1    (1*vc_ic*kco,4*kcf_ic*rc_ic*vc_ic*koc)
~ C0 <-> OS     (ko, ko*Ro)

~ C4 <-> C4IC1   (ric_c^4*vic_c^4*ki*exp(Hki*(celsius-25))*exp(Hvic_c*(celsius-25)), vic_c^4*ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25))*exp(Hvic_c*(celsius-25)))
~ C3 <-> C4IC1   (ric_c^3*vic_c^3*ki*exp(Hki*(celsius-25))*exp(Hvic_c*(celsius-25)), vic_c^3*ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25))*exp(Hvic_c*(celsius-25)))
~ C2 <-> C4IC1   (ric_c^2*vic_c^2*ki*exp(Hki*(celsius-25))*exp(Hvic_c*(celsius-25)), vic_c^2*ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25))*exp(Hvic_c*(celsius-25)))
~ C1 <-> C4IC1   (ric_c^1*vic_c^1*ki*exp(Hki*(celsius-25))*exp(Hvic_c*(celsius-25)), vic_c^1*ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25))*exp(Hvic_c*(celsius-25)))

~ OS <-> IC1    (ki*exp(Hki*(celsius-25)), ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25)))

~ OS <-> IN     (kn*exp(Hkn*(celsius-25)), kn*Rn*exp(Hkn*(celsius-25))*exp(Hrn*(celsius-25)))
~ IC1 <-> INC1  (vn_ic*rn_ic*kn*exp(Hkn*(celsius-25))*exp(Hrn_ic*(celsius-25))*exp(Hvn_ic*(celsius-25)), vn_ic*kn*Rn*exp(Hkn*(celsius-25))*exp(Hrn*(celsius-25))*exp(Hvn_ic*(celsius-25)))

~ IN <-> INC1    (ric_n*vic_n*ki*exp(Hki*(celsius-25))*exp(Hric_n*(celsius-25))*exp(Hvic_n*(celsius-25)), vic_n*ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25))*exp(Hvic_n*(celsius-25)))

CONSERVE C4+C3+C2+C1+C0+C4IC1+C3IC1+C2IC1+C1IC1+OS+IC1+IN+INC1 = 1
}

PROCEDURE rates(v(millivolt)) {
kco=kc*exp(Zc*F/A*(v-Vc-Hvc*(celsius-25)))*exp(Hkc*(celsius-25))
koc=kc*exp(-Zc*F/A*(v-Vc-Hvc*(celsius-25)))*exp(Hkc*(celsius-25))
}




