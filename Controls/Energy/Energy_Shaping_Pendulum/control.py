class Controller:

    def __init__(self, sim):
        self.model = sim.model
        self.data = sim.data
        self.option = sim.option

    def init_controller(self,):
        #initialize the controller here. This function is called once, in the beginning
        pass    

    def energy_shaping_controller(self,model, data):
        #put the controller here. This function is called inside the simulation.
        te = self.data.energy[0] + self.data.energy[1]
        max_pe = 1*abs(self.option.gravity[2])*3
        #print(max_pe)
        e = te - max_pe
        #print(e)
        k = 0.01
        #print(-k*self.data.qvel[0]*e)
        self.data.ctrl[0] = -k*self.data.qvel[0]*e

    def pid_controller(self,model, data):
        pass

    def swing_up_and_stabilize_controller(self,mode,data):
        pass