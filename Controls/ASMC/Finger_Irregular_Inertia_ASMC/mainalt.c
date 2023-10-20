#include<stdbool.h> //for bool
//#include<unistd.h> //for usleep
//#include <math.h>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

//simulation end time
double simend = 20;

//related to writing data to a file
FILE *fid;
int loop_index = 0;
const int data_frequency = 10; //frequency at which data is written to a file


// char xmlpath[] = "../myproject/template_writeData/pendulum.xml";
// char datapath[] = "../myproject/template_writeData/data.csv";


//Change the path <template_writeData>
//Change the xml file
char path[] = "";
char xmlfile[] = "SARHARC.xml";


char datafile[] = "data.csv";


// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
double_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum ctrl;

//trajectories

double q_ref[40][3]={0};
double qd_ref[40][3]={0};
double qdd_ref[40][3]={0};




// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


//****************************
//This function is called once and is used to get the headers
void init_save_data()
{
  //write name of the variable here (header)
   fprintf(fid,"t, ");
   fprintf(fid,"mcp_t, ");
   fprintf(fid,"pip_t, ");
   fprintf(fid,"dip_t, ");
   fprintf(fid,"mcp_q, ");
   fprintf(fid,"pip_q, ");
   fprintf(fid,"dip_q, ");
   fprintf(fid,"mcp_qd, ");
   fprintf(fid,"pip_qd, ");
   fprintf(fid,"dip_qd, ");
   fprintf(fid,"mcp_qdd, ");
   fprintf(fid,"pip_qdd, ");
   fprintf(fid,"dip_qdd, ");
   //Don't remove the newline
   fprintf(fid,"\n");
}

//***************************
//This function is called at a set frequency, put data here
void save_data(const mjModel* m, mjData* d)
{
  //data here should correspond to headers in init_save_data()
  //seperate data by a space %f followed by space
  fprintf(fid,"%f, ",d->time);
  fprintf(fid,"%f, ",d->ctrl[4]);
  fprintf(fid,"%f, ",d->ctrl[5]);
  fprintf(fid,"%f, ",d->ctrl[6]);
  fprintf(fid,"%f, ",d->qpos[0]);
  fprintf(fid,"%f, ",d->qpos[1]);
  fprintf(fid,"%f, ",d->qpos[2]);
  fprintf(fid,"%f, ",d->qvel[0]);
  fprintf(fid,"%f, ",d->qvel[1]);
  fprintf(fid,"%f, ",d->qvel[2]);
  fprintf(fid,"%f, ",d->qacc[0]);
  fprintf(fid,"%f, ",d->qacc[1]);
  fprintf(fid,"%f, ",d->qacc[2]);

  //Don't remove the newline
  fprintf(fid,"\n");
}

/******************************/
void set_torque_control(const mjModel* m,int actuator_no,int flag)
{
  if (flag==0)
    m->actuator_gainprm[10*actuator_no+0]=0;
  else
    m->actuator_gainprm[10*actuator_no+0]=1;
}
/******************************/


/******************************/
void set_position_servo(const mjModel* m,int actuator_no,double kp)
{
  m->actuator_gainprm[10*actuator_no+0]=kp;
  m->actuator_biasprm[10*actuator_no+1]=-kp;
}
/******************************/

/******************************/
void set_velocity_servo(const mjModel* m,int actuator_no,double kv)
{
  m->actuator_gainprm[10*actuator_no+0]=kv;
  m->actuator_biasprm[10*actuator_no+2]=-kv;
}
/******************************/

//**************************

void read_traj(void){


    FILE* fp_q_ref = fopen("q_ref.csv", "r");


    if (!fp_q_ref)
        printf("Can't open file\n");

    else {
        // Here we have taken size of
        // array 1024 you can modify it
        char buffer[120];
        int row = 0;
        int column = 0;
        while (fgets(buffer,
                     120, fp_q_ref)) {
            column = 0;
            // Splitting the data
            char* value = strtok(buffer, ",");
            while (value) {
                /*// Column 1
                if (column == 0) {
                    printf("theta_1 :");
                }
                // Column 2
                if (column == 1) {
                    printf(" theta_2 :");
                }
                // Column 3
                if (column == 2) {
                    printf(" theta_3 :");
                }
                printf("%s", value);*/
                q_ref[row][column] = atof(value);
                value = strtok(NULL, ",");
                column++;
            }
/*            printf("\n");*/
            row++;
        }
        // Close the file
        fclose(fp_q_ref);
    }

    FILE* fp_qd_ref = fopen("qd_ref.csv", "r");

    // qd_ref
    if (!fp_qd_ref)
        printf("Can't open file\n");

    else {
        // Here we have taken size of
        // array 1024 you can modify it
        char buffer[120];
        int row = 0;
        int column = 0;
        while (fgets(buffer,
                     120, fp_qd_ref)) {
            column = 0;
            // Splitting the data
            char* value = strtok(buffer, ",");
            while (value) {
                // Column 1
               /* if (column == 0) {
                    printf("thetad_1 :");
                }
                // Column 2
                if (column == 1) {
                    printf(" thetad_2 :");
                }
                // Column 3
                if (column == 2) {
                    printf(" thetad_3 :");
                }
                printf("%s", value);*/
                qd_ref[row][column] = atof(value);
                value = strtok(NULL, ",");
                column++;
            }
/*            printf("\n");*/
            row++;
        }
        // Close the file
        fclose(fp_qd_ref);
    }

    FILE* fp_qdd_ref = fopen("qdd_ref.csv", "r");

    if (!fp_qdd_ref)
        printf("Can't open file\n");

    else {
        // Here we have taken size of
        // array 1024 you can modify it
        char buffer[120];
        int row = 0;
        int column = 0;
        while (fgets(buffer,
                     120, fp_qdd_ref)) {
            column = 0;
            // Splitting the data
            char* value = strtok(buffer, ",");
            while (value) {
                // Column 1
                /*if (column == 0) {
                    printf("thetadd_1 :");
                }
                // Column 2
                if (column == 1) {
                    printf(" thetadd_2 :");
                }
                // Column 3
                if (column == 2) {
                    printf(" thetadd_3 :");
                }
                printf("%s", value);*/
                qdd_ref[row][column] = atof(value);
                value = strtok(NULL, ",");
                column++;
            }
/*            printf("\n");*/
            row++;
        }
        // Close the file
        fclose(fp_qdd_ref);
    }

}

void init_controller(const mjModel* m, mjData* d)
{
    // path = ../myproject/SARHARC/
    read_traj();

}



//**************************
void mycontroller(const mjModel* m, mjData* d)
{
  //write control here
  double t;
  int t_sec;
  t = d->time;
  double res[4]={0,0,0,0};
  double atp[12] = {-112.7650,0,0,112.7650,0,0,-225.5300,215.2120,0,0,-215.2120,59.2944};
  double tvec[3]={0,0,0};

  double dense_M[9] = {0,0,0,0, 0, 0, 0, 0, 0}; //mass matrix is square, for 2 dof nv=2 we have a 4 or 2X2 M matrix
  mj_fullM(m, dense_M, d->qM); // the function to access the mass matrix in Mujoco
  double M[3][3];
  M[0][0] = dense_M[0];
  M[0][1] = dense_M[1];
  M[0][2] = dense_M[2];
  M[1][0] = dense_M[3];
  M[1][1] = dense_M[4];
  M[1][2] = dense_M[5];
  M[2][0] = dense_M[6];
  M[2][1] = dense_M[7];
  M[2][2] = dense_M[8];


  double c_mcp, c_pip, c_dip;
  double e_q[3] = {d->qpos[0] - q_ref[t_sec][0], d->qpos[1] - q_ref[t_sec][1], d->qpos[2] - q_ref[t_sec][2]};
  double e_qd[3] = {d->qvel[0] - qd_ref[t_sec][0], d->qvel[1] - qd_ref[t_sec][1], d->qvel[2] - qd_ref[t_sec][2]};
  double e_qdd[3] = {d->qacc[0] - qdd_ref[t_sec][0], d->qacc[1] - qdd_ref[t_sec][1], d->qacc[2] - qdd_ref[t_sec][2]};
  //printf("%d",(int)(t/0.5));
  //printf("\n");
  t_sec = (int)(t/0.5);
  double kp = 0.0005, kv = 0.00005;
  double kde_qd[3]={0,0,0};
  double kpe_q[3]={0,0,0};
  double h[3]={0,0,0};
  double mqdd[3]={0,0,0};
  double mqdd_d[3]={0,0,0};
  double mkde_qd[3] = {0,0,0};
  double mkpe_q[3] = {0,0,0};
  double qdd_d[3] = {0,0,0};


  // variables for the trajectory planning stack

  // the position velocity and acceleration profiles will be sampled and directly injected into the simulation
  double q[3] = {d->qpos[0],d->qpos[1],d->qpos[2]};
  double qd[3] = {d->qvel[0], d->qvel[1], d->qvel[2]};
  double qdd[3] = {d->qacc[0],d->qacc[1],d->qacc[2]};

  double q_init[3] = {d->qpos[0],d->qpos[1],d->qpos[2]};
  double qd_init[3] = {0,0,0};
  double qdd_init[3] = {0,0,0};

  double q_fin[3] = {1.5708,1.309,1.22173};
  double qd_fin[3] = {0,0,0};
  double qdd_fin[3] = {0,0,0};






  if(t_sec <=40) {

      qdd_d[0] = qdd_ref[t_sec][0];
      qdd_d[1] = qdd_ref[t_sec][1];
      qdd_d[2] = qdd_ref[t_sec][2];
      // get the (Kd . e_qd) term
      mju_scl(kde_qd, e_qd, kv, 3);
      // get the (Kp . e_q) term
      mju_scl(kpe_q, e_q, kp, 3);
      // get the (M . qdd_d) term
      mju_mulMatVec(mqdd_d , dense_M, qdd_d, 3, 3);
      // get the (M . Kd . e_qd) term
      mju_mulMatVec(mkde_qd, dense_M, kde_qd, 3, 3);
      // get the (M . Kp . e_q) term
      mju_mulMatVec(mkpe_q, dense_M, kpe_q, 3, 3);


      h[0] = d->qfrc_bias[0];
      h[1] = d->qfrc_bias[1];
      h[2] = d->qfrc_bias[2];


      /*d->ctrl[4] = -kp * (d->qpos[0] - q_ref[t_sec][0]) - kv * (d->qvel[0] - qd_ref[t_sec][0]);
      d->ctrl[5] = -kp * (d->qpos[1] - q_ref[t_sec][1]) - kv * (d->qvel[1] - qd_ref[t_sec][1]);
      d->ctrl[6] = -kp * (d->qpos[2] - q_ref[t_sec][2]) - kv * (d->qvel[2] - qd_ref[t_sec][2]);*/

/*      c_mcp = -kp * (d->qpos[0] - q_ref[t_sec][0]) - kv * (d->qvel[0] - qd_ref[t_sec][0]);
      c_pip = -kp * (d->qpos[1] - q_ref[t_sec][1]) - kv * (d->qvel[1] - qd_ref[t_sec][1]);
      c_dip = -kp * (d->qpos[2] - q_ref[t_sec][2]) - kv * (d->qvel[2] - qd_ref[t_sec][2]);*/
    // feedback linearized torques
      c_mcp = h[0] + mqdd_d[0] - mkde_qd[0] - mkpe_q[0];
      c_pip = h[1] + mqdd_d[1] - mkde_qd[1] - mkpe_q[1];
      c_dip = h[2] + mqdd_d[2] - mkde_qd[2] - mkpe_q[2];

      tvec[0]=c_mcp;
      tvec[1]=c_pip;
      tvec[2]=c_dip;

      d->ctrl[4] = tvec[0];
      d->ctrl[5] = tvec[1];
      d->ctrl[6] = tvec[2];






        
        
        
    }








  //write data here (dont change/dete this function call; instead write what you need to save in save_data)
  if (loop_index%data_frequency==0)
    {
      save_data(m,d);

    }
  loop_index = loop_index + 1;
}





void myasmc(const mjModel* m, mjData* d){


  //write control here
  double t;
  int t_sec;
  t = d->time;
  double res[4]={0,0,0,0};
  double atp[12] = {-112.7650,0,0,112.7650,0,0,-225.5300,215.2120,0,0,-215.2120,59.2944};
  double tvec[3]={0,0,0};
  //Access the terms of the Mass Matrix
  double dense_M[9] = {0,0,0,0,0,0,0,0,0}; //mass matrix is square, for 2 dof nv=2 we have a 4 or 2X2 M matrix
  mj_fullM(m, dense_M, d->qM); // the function to access the mass matrix in Mujoco
  double M[3][3];
  M[0][0] = dense_M[0];
  M[0][1] = dense_M[1];
  M[0][2] = dense_M[2];
  M[1][0] = dense_M[3];
  M[1][1] = dense_M[4];
  M[1][2] = dense_M[5];
  M[2][0] = dense_M[6];
  M[2][1] = dense_M[7];
  M[2][2] = dense_M[8];



  
  double c_mcp, c_pip, c_dip;
  //position error
  double e_q[3] = {d->qpos[0] - q_ref[t_sec][0], d->qpos[1] - q_ref[t_sec][1], d->qpos[2] - q_ref[t_sec][2]};
  //velocity error
  double e_qd[3] = {d->qvel[0] - qd_ref[t_sec][0], d->qvel[1] - qd_ref[t_sec][1], d->qvel[2] - qd_ref[t_sec][2]};
  //acceleration error
  double e_qdd[3] = {d->qacc[0] - qdd_ref[t_sec][0], d->qacc[1] - qdd_ref[t_sec][1], d->qacc[2] - qdd_ref[t_sec][2]};

  t_sec = (int)(t/0.5);

  // what are these Kp Kv values fpr fucks sake
  double kp = 0.0005, kv = 0.00005;

  //
  double kde_qd[3]={0,0,0};
  double kpe_q[3]={0,0,0};
  double h[3]={0,0,0};
  double mqdd[3]={0,0,0};
  double mqdd_d[3]={0,0,0};
  double mkde_qd[3] = {0,0,0};
  double mkpe_q[3] = {0,0,0};
  double qdd_d[3] = {0,0,0};


  // variables for the trajectory planning stack

  // the position velocity and acceleration profiles will be sampled and directly injected into the simulation
  double q[3] = {d->qpos[0],d->qpos[1],d->qpos[2]};
  double qd[3] = {d->qvel[0], d->qvel[1], d->qvel[2]};
  double qdd[3] = {d->qacc[0],d->qacc[1],d->qacc[2]};

  double q_init[3] = {d->qpos[0],d->qpos[1],d->qpos[2]};
  double qd_init[3] = {0,0,0};
  double qdd_init[3] = {0,0,0};

  double q_fin[3] = {1.5708,1.309,1.22173};
  double qd_fin[3] = {0,0,0};
  double qdd_fin[3] = {0,0,0};


  double phi[9] = {1,0,0,0,1,0,0,0,1}; // let the sliding surface multiplier be an identity matrix

  double phie[3]={0,0,0};

  mjc_mulMatVec(phie, phi, e_q, 3, 3);

  double s[3] = {e_qd[0]+ phie[0],e_qd[1]+ phie[1],e_qd[2]+ phie[2]};

  double zeta[6] = {e_q[0],e_q[1],e_q[2], e_qd[0],e_qd[1],e_qd[2]};

  





  //this if condition exists to cut off simulation after 40 seconds
  if(t_sec <=40) {

      //desired acceleration at that time step
      qdd_d[0] = qdd_ref[t_sec][0];
      qdd_d[1] = qdd_ref[t_sec][1];
      qdd_d[2] = qdd_ref[t_sec][2];





      // get the (Kd . e_qd) term and store it in kde_qd
      mju_scl(kde_qd, e_qd, kv, 3);
      // get the (Kp . e_q) term and store it in kpe_q
      mju_scl(kpe_q, e_q, kp, 3);
      // get the (M . qdd_d) term
      mju_mulMatVec(mqdd_d , dense_M, qdd_d, 3, 3);
      // get the (M . Kd . e_qd) term
      mju_mulMatVec(mkde_qd, dense_M, kde_qd, 3, 3);
      // get the (M . Kp . e_q) term
      mju_mulMatVec(mkpe_q, dense_M, kpe_q, 3, 3);

      // extracting the H(q,q_d) terms values
      h[0] = d->qfrc_bias[0];
      h[1] = d->qfrc_bias[1];
      h[2] = d->qfrc_bias[2];

      //control input for each joint
      c_mcp = h[0] + mqdd_d[0] - mkde_qd[0] - mkpe_q[0];
      c_pip = h[1] + mqdd_d[1] - mkde_qd[1] - mkpe_q[1];
      c_dip = h[2] + mqdd_d[2] - mkde_qd[2] - mkpe_q[2];

      //store the control input in the redundant tvec
      tvec[0]=c_mcp;
      tvec[1]=c_pip;
      tvec[2]=c_dip;

      //inject control into the simulation
      d->ctrl[4] = tvec[0];
      d->ctrl[5] = tvec[1];
      d->ctrl[6] = tvec[2];






        
        
        
    }








  //write data here (dont change/dete this function call; instead write what you need to save in save_data)
  if (loop_index%data_frequency==0)
    {
      save_data(m,d);

    }
  loop_index = loop_index + 1;


    
}



//************************
// main function
int main(int argc, const char** argv)
{

    // activate software
    mj_activate("mjkey.txt");

    char xmlpath[100]={0};
    char datapath[100]={0};

    strcat(xmlpath,path);
    strcat(xmlpath,xmlfile);

    strcat(datapath,path);
    strcat(datapath,datafile);


    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 )
        m = mj_loadXML(xmlpath, 0, error, 1000);

    else
        if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
            m = mj_loadModel(argv[1], 0);
        else
            m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);


    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1244, 700, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    double arr_view[] = {-104.449080, -19.016950, 0.397198, 0.009315, -0.023338, -0.045315};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    d->qpos[0] = 0.3;
    d->qpos[1] = 0.3;
    d->qpos[2] = 0.3;

    // install control callback
    mjcb_control = mycontroller;


    fid = fopen(datapath,"w");
    init_save_data();
    init_controller(m,d);

    // use the first while condition if you want to simulate for a period.
    while( !glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
        {
            mj_step(m, d);
        }

        if (d->time>=simend)
        {
           fclose(fid);
           break;
         }

       // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

          // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        //printf("{%f, %f, %f, %f, %f, %f};\n",cam.azimuth,cam.elevation, cam.distance,cam.lookat[0],cam.lookat[1],cam.lookat[2]);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}
