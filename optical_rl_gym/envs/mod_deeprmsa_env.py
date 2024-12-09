
from typing import Tuple

import gym
import numpy as np
import copy
import networkx as nx
from .mod_rmsa_env import ModifiedRMSAEnv




class ModifiedDeepRMSAEnv(ModifiedRMSAEnv):
    def __init__(
        self,
        topology=None,
        j=1,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        num_spectrum_resources=50,
        node_request_probabilities=None,
        seed=None,
        allow_rejection=False,
    ):
        # Create index mappings (0-based indices for internal use)
        self.node_map = {
            int(node): idx for idx, node in enumerate(sorted(topology.nodes()))
        }
        self.node_ids = {  # Changed from inv_node_map to node_ids
            idx: str(node) for node, idx in self.node_map.items()
        }
        
        # Store original topology
        self.topology = copy.deepcopy(topology)
        
        print("Node mappings:", self.node_map)
        print("Node IDs:", self.node_ids)
        # Calculate the longest path length correctly
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(self.topology, weight='weight'))
        self.topology.graph['longest_path_length'] = max(
            max(lengths.values()) for lengths in path_lengths.values()
        )

        super().__init__(
            topology=self.topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            reset=False,
        )
        self.j = j
        # Path finding specific parameters
        self.max_path_length = self.topology.number_of_nodes() - 1
        
        # Define observation space components
        self.path_obs_size = (
            2 * self.topology.number_of_nodes() +  # Source and destination one-hot
            2 * self.topology.number_of_edges()    # Link distance and fragmentation
        )
        
        # Define observation spaces for both phases
        self.path_observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.path_obs_size,),
            dtype=np.float32
        )
        
        self.spectrum_observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2 * self.j + 3,),  # Correct shape based on j
            dtype=np.float32
        )
        
        # Combined observation space
        self.observation_space = gym.spaces.Dict({
            'phase': gym.spaces.Discrete(2),  # 0 for path finding, 1 for spectrum
            'path_obs': self.path_observation_space,
            'spectrum_obs': self.spectrum_observation_space
        })
        
        # Action spaces for each phase
        self.action_space = gym.spaces.Dict({
            'path_action': gym.spaces.Discrete(self.topology.number_of_nodes()),
            'spectrum_action': gym.spaces.Discrete(self.num_spectrum_resources)
        })
        
        self.reset(only_episode_counters=False)
        
    def _get_path_observation(self):
        """Get observation for path finding phase"""
        if self.current_service is None:
            return np.zeros(self.path_observation_space.shape, dtype=np.float32)
            
        obs = []
        
        # Source and destination one-hot encodings
        source_one_hot = np.zeros(self.topology.number_of_nodes())
        source_one_hot[self.current_service.source_id] = 1
        
        dest_one_hot = np.zeros(self.topology.number_of_nodes())
        dest_one_hot[self.current_service.destination_id] = 1
        
        obs.extend(source_one_hot)
        obs.extend(dest_one_hot)
        
        # Link states
        for i, j in self.topology.edges():
            # Normalized distance
            distance = self.topology[i][j]['weight']
            norm_distance = distance / self.topology.graph['longest_path_length']
            obs.append(norm_distance)
            
            # Link fragmentation
            fragmentation = self._calculate_link_fragmentation(i, j)
            obs.append(fragmentation)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_spectrum_observation(self):
        """Get observation for spectrum assignment phase"""
        path_object = self._create_path_object()
        
        # Get available blocks
        initial_indices, lengths = self.get_available_blocks(path_object)
        
        # Create spectrum observation similar to original DeepRMSA
        spectrum_obs = np.full((2 * self.j + 3,), fill_value=-1.0)
        
        # Fill spectrum information
        for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
            if idb >= self.j:
                break
            # Initial slot index normalized
            spectrum_obs[idb * 2] = (
                2 * (initial_index - 0.5 * self.num_spectrum_resources)
                / self.num_spectrum_resources
            )
            # Number of contiguous slots normalized
            spectrum_obs[idb * 2 + 1] = (length - 8) / 8
            
        # Add path specific metrics
        slots_needed = self.get_number_slots(path_object)
        spectrum_obs[self.j * 2] = (slots_needed - 5.5) / 3.5
        
        # Add total available slots
        available_slots = self.get_available_slots(path_object)
        spectrum_obs[self.j * 2 + 1] = np.sum(available_slots) / self.num_spectrum_resources
        
        return np.array(spectrum_obs, dtype=np.float32)
        
    def observation(self):
        """Get current observation based on phase"""
        if self.current_service is None:
            return {
                'phase': 0,
                'path_obs': np.zeros(self.path_observation_space.shape, dtype=np.float32),
                'spectrum_obs': np.zeros(self.spectrum_observation_space.shape, dtype=np.float32)
            }
            
        path_obs = self._get_path_observation()
        spectrum_obs = self._get_spectrum_observation() if not self.path_finding_phase else np.zeros_like(self.spectrum_observation_space.low)
        
        return {
            'phase': 0 if self.path_finding_phase else 1,
            'path_obs': path_obs,
            'spectrum_obs': spectrum_obs
        }
    
    def _calculate_link_fragmentation(self, node1: int, node2: int) -> float:
        """Calculate fragmentation of a link"""
        # Get link index
        link_idx = self.topology[node1][node2]["index"]
        
        # Get slots array
        slots = self.topology.graph['available_slots'][link_idx]
        
        if np.sum(slots) == 0:  # If no slots available
            return 1.0
            
        # Count fragments
        initial_indices, values, lengths = self.rle(slots)
        
        # Get unused blocks
        unused_blocks = [i for i, x in enumerate(values) if x == 1]
        
        if len(unused_blocks) > 0:
            max_block = max(lengths[unused_blocks])
            total_free = np.sum(slots)
            return 1 - (max_block / total_free)
        else:
            return 1.0  # Maximum fragmentation when no free blocks

    def step(self, action):
        """Handle both path finding and spectrum assignment"""
        if self.path_finding_phase:
            if isinstance(action, dict):
                action = action['path_action']
                
            # print(f"Step - Action: {action}")
            # print(f"Current path (indices): {self.current_path}")
            # print(f"Current path (nodes): {[self.node_ids[idx] for idx in self.current_path]}")
            
            next_state, reward, done, info = super()._handle_path_finding(action)
            
            if done and not info.get('path_complete', False):
                # Path finding failed
                return next_state, -1, True, info
                
            if info.get('path_complete', False):
                # Successfully found path, continue to spectrum assignment
                return next_state, reward, False, info
        else:
            # Spectrum assignment phase
            if isinstance(action, dict):
                action = action['spectrum_action']
                
            next_state, reward, done, info = super()._handle_spectrum_assignment(action)
            
            if done:
                # Move to next request
                self._next_service()
                self._init_path_finding()
            
        return next_state, reward, done, info
    
    def get_available_blocks(self, path):
        """Get available spectrum blocks for the path"""
        available_slots = self.get_available_slots(path)
        
        # Get the blocks
        initial_indices, values, lengths = self.rle(available_slots)
        
        # Find blocks with enough slots
        slots_needed = self.get_number_slots(path)
        valid_blocks = [
            (idx, length) 
            for idx, (val, length) in zip(initial_indices, zip(values, lengths))
            if val == 1 and length >= slots_needed
        ]
        
        if not valid_blocks:
            return [], []
            
        return zip(*valid_blocks)
    
    def reset(self, only_episode_counters=True):
        """Reset environment with handling for only_episode_counters"""
        if only_episode_counters:
            return super().reset(only_episode_counters=only_episode_counters)
        
        # Full reset
        observation = super().reset(only_episode_counters=only_episode_counters)
        
        # Generate first service if needed
        if self.current_service is None:
            self._next_service()
        
        # Initialize path finding state
        self.path_finding_phase = True
        self.current_path = [self.current_service.source_id]
        self.visited_nodes = {self.current_service.source_id}
        
        return self.observation()