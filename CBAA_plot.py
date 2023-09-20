from CBAA import CBAA_agent

import numpy as np
import matplotlib.pyplot as plt


task_num = 4
robot_num = 4

task = np.random.uniform(low=0,high=1,size=(task_num,2))

robot_list = [CBAA_agent(id=i, task=task) for i in range(robot_num)]

fig, ax = plt.subplots()
ax.set_xlim((-0.1,1.1))
ax.set_ylim((-0.1,1.1))

ax.plot(task[:,0],task[:,1],'rx')
robot_pos = np.array([r.state[0].tolist() for r in robot_list])
ax.plot(robot_pos[:,0],robot_pos[:,1],'b^')

# Network Initialize
G = np.ones((robot_num, robot_num)) # Fully connected network
G[2,3]=0
G[3,2]=0
G[1,2]=0
G[2,1]=0
G[1,3]=0
G[3,1]=0
# for i in range(robot_num-1):
#   for j in range(i+1,robot_num):
#     ax.plot([robot_pos[i][0],robot_pos[j][0]],[robot_pos[i][1],robot_pos[j][1]],'g--',linewidth=1)

t = 0 # Iteration number
assign_plots = []

while True:
  converged_list = []
  print("==Iteration {}==".format(t))
  ## Phase 1: Auction Process
  print("Auction Process")
  for robot_id, robot in enumerate(robot_list):
    # select task by local information
    robot.select_task()
    # print(robot.x)

    if t == 0:
      assign_line, = ax.plot([robot.state[0][0],task[robot.J,0]],[robot.state[0][1],task[robot.J,1]],'k-',linewidth=1)
      assign_plots.append(assign_line)
    else:
      assign_plots[robot_id].set_data([robot.state[0][0],task[robot.J,0]],[robot.state[0][1],task[robot.J,1]])

  plt.pause(0.5)

  ## Phase 2: Consensus Process
  print("Consensus Process")
  # Send winning bid list to neighbors (depend on env)
  message_pool = [robot.send_message() for robot in robot_list]

  for robot_id, robot in enumerate(robot_list):
    # Recieve winning bidlist from neighbors
    g = G[robot_id]
    connected, = np.where(g==1)
    connected = list(connected)
    connected.remove(robot_id)

    if len(connected) > 0:
        Y = {neighbor_id:message_pool[neighbor_id] for neighbor_id in connected}
    else:
        Y = None

    # Update local information and decision
    if Y is not None:
      converged = robot.update_task(Y)
      converged_list.append(converged)
      print(converged)
    # print(robot.x)

    if any(robot.x): # (list)
      assign_plots[robot_id].set_data([robot.state[0][0],task[robot.J,0]],[robot.state[0][1],task[robot.J,1]])
    else:
      assign_plots[robot_id].set_data([robot.state[0][0],robot.state[0][0]],[robot.state[0][1],robot.state[0][1]])



  plt.pause(0.5)

  t += 1

  # 모든 로봇 agent 수와 최적의 임무계획 결과 수가 같으면 모든게 합의됨.
  # if sum(converged_list)==robot_num:
  #   break

print("CONVERGED")
plt.show()