def params(m0=0.310810/.0849):
    """
    Paramter file for fits to V(r) vs r.
    m0 = pion mass

    Long-range potential terms: ((1-self.gaussian(x,c,1))**2)*(-np.exp(a) * np.exp(-np.exp(m) * x)/ x )
    where gaussian(x,gamma,b) = b * np.exp(-((np.exp(gamma))**2 * x ** 2))

    Short-range potential terms: harmosc(x,gamma,b,n) =  b * np.exp(-(np.exp(gamma)**2) * x**2) * self.herm_w(np.exp(gamma)*x,n)
                                 herm_g2(x,gamma1,gamma2,b,n) = b * np.exp(-(np.exp(gamma1)**2) * x**2) * self.herm_w(np.exp(gamma2)*x,n)
                                 osc(x,gamma1,gamma2,ph,b) = b * np.cos(gamma2*x + ph) * np.exp(-(np.exp(gamma1)**1)*x**1)
   """
    import numpy as np
    import gvar as gv

    p = dict()
    p["prior_master_gaussian"] = gv.BufferDict(
        c_0=gv.gvar("-3.8(0.1)"),
        c_1=gv.gvar("-7.(0.1)"),
        c_2=gv.gvar("0.3(0.1)"),
        c_3=gv.gvar("-1.(0.1)"),
        c_4=gv.gvar("-0.3(0.1)"),
        a_0=gv.gvar(-1.95,0.1),
        a_1=gv.gvar(-3.0,1),
        a_2=gv.gvar(-9.0,1),
        a_3=gv.gvar(1.3,0.5),
        a_4=gv.gvar(-15.0,1),
        m_0=gv.gvar(1.30,0.01),
        m_1=gv.gvar(-.2,.1),
        m_2=gv.gvar(1.,.1),
        m_3=gv.gvar(1.,.1),
        m_4=gv.gvar(.02,.1),
        gamma1_0=gv.gvar("0.2(0.1)"),
        gamma1_1=gv.gvar("1(1)"),
        gamma1_2=gv.gvar("9(1)"),
        gamma1_3=gv.gvar("9(1)"),
        gamma1_4=gv.gvar("9(1)"),
        b_3=gv.gvar("7(5)"),
        b_0=gv.gvar("-.5(.1)"),
        b_1=gv.gvar("2(1)"),
        b_2=gv.gvar("7(5)"),
        b_4=gv.gvar("7(5)"),
    )

    p["prior_master_harmosc"] = gv.BufferDict(
	c_0=gv.gvar("-3.8(0.1)"),
        c_1=gv.gvar("-7.(0.1)"),
        c_2=gv.gvar("0.3(0.1)"),
        c_3=gv.gvar("-1.3(0.1)"),
	c_4=gv.gvar("-0.3(0.1)"),
        a_0=gv.gvar(-1.95,0.1),
        a_1=gv.gvar(-3.0,1),
	a_2=gv.gvar(-9.0,1),
        a_3=gv.gvar(1.3,.1),
        a_4=gv.gvar(-15.0,1),
        m_0=gv.gvar(1.297708,0.0001),
        m_1=gv.gvar(-.2,.1),
        m_2=gv.gvar(1.,.1),
	m_3=gv.gvar(1.,.1),
	m_4=gv.gvar(.02,.1),
	gamma1_0=gv.gvar(".2(.1)"),
	gamma1_1=gv.gvar("1.2(1)"),
	gamma1_2=gv.gvar("12(5)"),
        gamma1_3=gv.gvar("12(5)"),
        b_0=gv.gvar("-.6(.1)"),
	b_1=gv.gvar("1.7(.1)"),
	b_2=gv.gvar("15(20)"),
        b_3=gv.gvar("15(20)"),
    )

    p["prior_master_herm_g2"] = gv.BufferDict(
	c_0=gv.gvar("-3.8(0.1)"),
        c_1=gv.gvar("-7.(0.1)"),
        c_2=gv.gvar("0.3(0.1)"),
        c_3=gv.gvar(".15(0.1)"),
        c_4=gv.gvar("-0.3(0.1)"),
        a_0=gv.gvar(-1.95,0.1),
        a_1=gv.gvar(-3.0,1),
        a_2=gv.gvar(-9.0,1),
        a_3=gv.gvar(1.8,.1),
        a_4=gv.gvar(-15.0,1),
        m_0=gv.gvar(1.297708,0.0001),
        m_1=gv.gvar(-.2,.1),
        m_2=gv.gvar(1.,.1),
        m_3=gv.gvar(1.1,.1),
        m_4=gv.gvar(.02,.1),
        gamma1_0=gv.gvar(".7(.5)"),
        gamma2_0=gv.gvar("1(1)"),
	gamma2_1=gv.gvar("1(1.0)"),
	gamma2_2=gv.gvar("12(1)"),
	b_0=gv.gvar("8(1)"),
	b_1=gv.gvar("-20(10)"),
	b_2=gv.gvar("0(0.1)"),
    )

    p["prior_master_osc"] = gv.BufferDict(
	c_0=gv.gvar("-3.8(0.2)"),
        c_1=gv.gvar("-7.(0.1)"),
        c_2=gv.gvar("0.3(0.1)"),
        c_3=gv.gvar("-3.(0.1)"),
        c_4=gv.gvar("-0.3(0.1)"),
        a_0=gv.gvar(-1.95,0.1),
        a_1=gv.gvar(-3.0,1),
        a_2=gv.gvar(-9.0,1),
        a_3=gv.gvar(1.3,.1),
        a_4=gv.gvar(-15.0,1),
        m_0=gv.gvar(1.297708,0.0001),
        m_1=gv.gvar(-.2,.1),
        m_2=gv.gvar(1.,.1),
        m_3=gv.gvar(1.,.1),
        m_4=gv.gvar(.02,.1),
	gamma1_0=gv.gvar("1(1)"),
	gamma1_1=gv.gvar("1(1)"),
	gamma1_2=gv.gvar("1(1)"),
	gamma2_0=gv.gvar("3(1)"),
	gamma2_1=gv.gvar("1(1)"),
	gamma2_2=gv.gvar("12(1)"),
	b_0=gv.gvar("20(20)"),
	b_1=gv.gvar("2.(5)"),
	b_2=gv.gvar("2(1)"),
        ph_0=gv.gvar("1.(1)"),
        ph_1=gv.gvar("1.(1)"),
        ph_2=gv.gvar("1.(1)")
    )

    return p

