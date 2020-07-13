from rotormodel import jnrcalculator, majumdarghosh
def calculator(normal):
    (Jn,Jnn,hz)=normal
    with jnrcalculator.suppress_stdout():
        r=majumdarghosh.MGChain(Jn=Jn,Jnn=Jnn,hz=hz).calculate_expvals()
    return normal,r
g=jnrcalculator.GeneralJNRCalculator(calculator,3,'mghosh.pickle')
g.process_random_points(100)
# in jupyter: %matplotlib notebook 
g.draw_3d(0,1,2)
g.dump_csv('mghosh_pts.csv') 
