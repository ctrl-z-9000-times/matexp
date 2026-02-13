NEURON {
SUFFIX Kv41_13
USEION k READ ek WRITE ik
RANGE g, gbar
}

UNITS { (mV) = (millivolt) }

PARAMETER {
Ri=1.3567 ()
Rn=0.031678 ()
Ro=2.6831 ()
Vc= -30.9315 (mV)
Zc= 1.4398 ()
kc=0.2463 (/ms)
ki= 0.0001 (/ms)
kn= 0.42513 (/ms)
ko= 0.49899 (/ms)
ric_c=0.18414 ()
ric_n=123.2257 ()
vc_ic=4.5526 ()
vic_c=1000.0 ()
vic_n=0.001 ()
vn_ic=10.0 ()
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

LOCAL F, A,rc_ic,rn_ic,kcf_ic

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

~ C4 <-> C4IC1   (ric_c^4*vic_c^4*ki, vic_c^4*ki*Ri)
~ C3 <-> C4IC1   (ric_c^3*vic_c^3*ki, vic_c^3*ki*Ri)
~ C2 <-> C4IC1   (ric_c^2*vic_c^2*ki, vic_c^2*ki*Ri)
~ C1 <-> C4IC1   (ric_c^1*vic_c^1*ki, vic_c^1*ki*Ri)

~ OS <-> IC1    (ki, ki*Ri)

~ OS <-> IN     (kn, kn*Rn)
~ IC1 <-> INC1  (vn_ic*rn_ic*kn, vn_ic*kn*Rn)

~ IN <-> INC1    (ric_n*vic_n*ki, vic_n*ki*Ri)

CONSERVE C4+C3+C2+C1+C0+C4IC1+C3IC1+C2IC1+C1IC1+OS+IC1+IN+INC1 = 1
}

PROCEDURE rates(v(millivolt)) {
kco=kc*exp(Zc*F/A*(v-Vc))
koc=kc*exp(-Zc*F/A*(v-Vc))
}




