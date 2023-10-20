import os
import mujoco as mj
class SimUtils:

    def __init__(self,xml_name, simend):
        self.full_path = os.path.realpath(__file__)
        self.path = os.path.dirname(self.full_path)
        self.xml_path = '/'+self.path[1:]+'/' +xml_name #xml file (assumes this is in the same folder as this file)'
        self.simend = simend
        #print(self.xml_path)
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options
        self.option = mj.MjOption()
