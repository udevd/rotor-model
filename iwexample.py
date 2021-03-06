from rotormodel import jnrcalculator, isingwitness
def calculator(normal):
    (Jxx,Jyy,Y)=normal
    hz=0
    with jnrcalculator.suppress_stdout():
        r=aklt.AKLTChain(J=J,Y=Y,hz=hz).calculate_expvals()
    return normal,r
g=jnrcalculator.GeneralJNRCalculator(calculator,3,'aklt.pickle')
g.process_random_points(100)

#in jupyter: %matplotlib notebook
g.draw_3d(0,1,2)

g.dump_csv('iw_points.csv') #each line contains: normal_x, normal_y, normal_z, p_x, p_y, p_z
