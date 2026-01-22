import MAPF
import numpy as np
from typing import Dict, List, Tuple,Set
import datetime
import random
from collections import defaultdict


class pyTaskScheduler:
    def __init__(self, env):
        self.env=env
  
  
    def initialize(self, preprocess_time_limit:int):
        """
        Initialize the task scheduler
        """
        random.seed(0)
        np.random.seed(0)
        print("Initializing Auction-based Python Scheduler...")
    
    def plan(self, time_limit: int):
        # Safety buffer (20ms) for python to cpp
        start_time = self.env.plan_current_time()
        max_duration = datetime.timedelta(milliseconds=time_limit - 20)
        
        proposed_schedule = list(self.env.curr_task_schedule)
        
        # 1. Identify Idle Agents
        idle_agents = []
        try:
            for i in range(self.env.num_of_agents):
                if proposed_schedule[i] == -1:
                    idle_agents.append(i)
        except:
            return proposed_schedule

        if not idle_agents:
            return proposed_schedule

        # 2. Identify Available Tasks
        available_tasks = []
        for tid, task in self.env.task_pool.items():
            if task.agent_assigned == -1:
                available_tasks.append(task)
        
        if not available_tasks:
            return proposed_schedule

        # GRID-BASED AUCTION OPTIMIZATION
        # Instead of every agent checking every task, bucket tasks into grid cells.
        # Agents only bid on tasks in their local neighborhood.
        try:
            agent_locations = [self.env.curr_states[i].location for i in range(self.env.num_of_agents)]
        except:
            return proposed_schedule
        
        # Determine Grid Size (divide map into roughly 8x8 sectors)
        cols = self.env.cols
        rows = self.env.rows
        
        # Dynamic cell size: aim for ~8 divisions along the width
        cell_size = max(5, cols // 8) 
        
        # Bucket tasks: key = (grid_x, grid_y), value = list of tasks
        task_grid = defaultdict(list)
        
        for task in available_tasks:
            if task.locations:
                t_loc = task.locations[0]
                tx, ty = t_loc % cols, t_loc // cols
                grid_key = (tx // cell_size, ty // cell_size)
                task_grid[grid_key].append(task)

        bids = []
        
        # 3. Generate Bids (Restricted by Grid)
        for agent_idx in idle_agents:
            agent_loc = agent_locations[agent_idx]
            ax, ay = agent_loc % cols, agent_loc // cols
            gx, gy = ax // cell_size, ay // cell_size
            
            # Check 3x3 neighbors
            candidate_tasks = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (gx + dx, gy + dy)
                    if neighbor_key in task_grid:
                        candidate_tasks.extend(task_grid[neighbor_key])

            # Fallback: If agent is in an empty corner of the map, 
            # pick a few random tasks so it doesn't stay idle forever.
            if not candidate_tasks:
                if len(available_tasks) > 5:
                    candidate_tasks = random.sample(available_tasks, 5)
                else:
                    candidate_tasks = available_tasks

            # Generate bids ONLY for these candidates
            for task in candidate_tasks:
                if task.locations:
                    # Calculate the average position (center of gravity) of all errands in the task
                    sum_x = 0
                    sum_y = 0
                    n_locs = len(task.locations)
                    
                    for t_loc in task.locations:
                        sum_x += t_loc % cols
                        sum_y += t_loc // cols
                    
                    avg_x = sum_x / n_locs
                    avg_y = sum_y / n_locs
                    
                    # Calculate distance from Agent to Task Centroid
                    dist = abs(ax - avg_x) + abs(ay - avg_y)
                    
                    bids.append((dist, agent_idx, task.task_id))
            
            # Time Check: Break early if we are running out of time
            # Check every 50 agents to minimize overhead
            if agent_idx % 50 == 0:
                 if (self.env.plan_current_time() - start_time) > max_duration:
                    break

        # 4. The Auction (Sort by cost) closest agent-task pair wins
        bids.sort(key=lambda x: x[0])

        # 5. Assign
        assigned_agents = set()
        assigned_tasks = set()

        for cost, agent_idx, task_id in bids:
            # Strict Time Check inside the assignment loop
            if len(assigned_agents) % 20 == 0:
                if (self.env.plan_current_time() - start_time) > max_duration:
                    break

            if agent_idx in assigned_agents:
                continue
            if task_id in assigned_tasks:
                continue

            proposed_schedule[agent_idx] = task_id
            assigned_agents.add(agent_idx)
            assigned_tasks.add(task_id)

        return proposed_schedule
