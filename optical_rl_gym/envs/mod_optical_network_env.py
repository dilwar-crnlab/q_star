import copy
import heapq
import random
from typing import List, Optional, Tuple, Set

import gym
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from optical_rl_gym.utils import Service, Path

class ModifiedOpticalNetworkEnv(gym.Env):
    def __init__(
        self,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10.0,
        mean_service_holding_time: float = 10800.0,
        num_spectrum_resources: int = 80,
        allow_rejection: bool = False,
        node_request_probabilities: Optional[np.array] = None,
        seed: Optional[int] = None,
        channel_width: float = 12.5,
    ):
        # Keep existing initialization
        super().__init__()
        assert topology is None or "ksp" in topology.graph
        assert topology is None or "k_paths" in topology.graph
        self._events: List[Tuple[float, Service]] = []
        self.current_time: float = 0
        self.episode_length: int = episode_length
        self.services_processed: int = 0
        self.services_accepted: int = 0
        self.episode_services_processed: int = 0
        self.episode_services_accepted: int = 0
        
        # Add path finding related attributes
        self.path_finding_phase = False
        self.current_path: List[int] = []
        self.visited_nodes: Set[int] = set()
        self.destination_id: Optional[int] = None
        self.path_action_mask = None
        
        # Initialize existing attributes
        self.current_service: Service = None
        self._new_service: bool = False
        self.allow_rejection: bool = allow_rejection
        
        self.load = load
        self.mean_service_holding_time = mean_service_holding_time
        self.set_load(load=load, mean_service_holding_time=mean_service_holding_time)
        
        self.rand_seed = seed
        self.rng = random.Random(self.rand_seed)
        
        self.topology: nx.Graph = copy.deepcopy(topology)
        self.topology_name: str = topology.graph["name"]
        self.k_paths: int = self.topology.graph["k_paths"]
        # just as a more convenient way to access it
        self.k_shortest_paths = self.topology.graph["ksp"]

        self.num_spectrum_resources: int = num_spectrum_resources
        self.channel_width: float = channel_width

        # Initialize spectrum resources first
        
        
        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges(), num_spectrum_resources), 
            dtype=int
        )

        self.topology.graph["num_spectrum_resources"] = num_spectrum_resources
        
        self.topology.graph["available_spectrum"] = np.full(
            (self.topology.number_of_edges()),
            fill_value=self.num_spectrum_resources,
            dtype=int,
        )


        
        # Initialize node probabilities
        if node_request_probabilities is not None:
            assert len(node_request_probabilities) == self.topology.number_of_nodes()
            self.node_request_probabilities = node_request_probabilities
        else:
            self.node_request_probabilities = np.full(
                self.topology.number_of_nodes(),
                fill_value=1.0 / self.topology.number_of_nodes(),
            )
            
    def set_load(self, load: float = None, mean_service_holding_time: float = None) -> None:
        """
        Sets the load to be used to generate requests.
        :param load: The load to be generated, in Erlangs
        :param mean_service_holding_time: The mean service holding time to be used to
        generate the requests
        :return: None
        """
        if load is not None:
            self.load = load
        if mean_service_holding_time is not None:
            self.mean_service_holding_time = (mean_service_holding_time)  # current_service holding time in seconds)
        self.mean_service_inter_arrival_time = 1 / float(self.load / float(self.mean_service_holding_time))
    def _get_path_finding_state(self) -> np.ndarray:
        """Get state representation for path finding phase"""
        state = []
        
        # Add current node one-hot encoding
        current_node_enc = np.zeros(self.topology.number_of_nodes())
        current_node_enc[self.current_path[-1]] = 1
        state.extend(current_node_enc)
        
        # Add destination one-hot encoding
        dest_node_enc = np.zeros(self.topology.number_of_nodes())
        dest_node_enc[self.destination_id] = 1
        state.extend(dest_node_enc)
        
        # Add link states (could be extended with more features)
        for _, _, data in self.topology.edges(data=True):
            state.append(data.get('weight', 1.0))  # Link distance
            
        return np.array(state)
        
    def _get_valid_next_nodes(self) -> List[int]:
        """Get list of valid next nodes during path finding"""
        if not self.current_path:
            return []
            
        current = self.current_path[-1]
        valid_nodes = []
        
        for neighbor in self.topology.neighbors(current):
            if neighbor not in self.visited_nodes:
                valid_nodes.append(neighbor)
                
        return valid_nodes
        
    def _update_action_mask(self):
        """Update the action mask for path finding"""
        self.path_action_mask = np.zeros(self.topology.number_of_nodes())
        valid_nodes = self._get_valid_next_nodes()
        self.path_action_mask[valid_nodes] = 1
        
    def _init_path_finding(self):
        """Initialize path finding phase"""
        self.path_finding_phase = True
        self.current_path = [self.current_service.source_id]
        self.visited_nodes = {self.current_service.source_id}
        self.destination_id = self.current_service.destination_id
        self._update_action_mask()
        
    def _calculate_link_cost(self, node1: int, node2: int) -> float:
        """Calculate cost of using a link"""
        # Basic implementation using distance
        return self.topology[node1][node2].get('weight', 1.0)
        
    def get_path_state(self):
        """Get the current state for path finding"""
        return {
            'observation': self._get_path_finding_state(),
            'action_mask': self.path_action_mask
        }
        
    def is_path_valid(self) -> bool:
        """Check if current path is valid"""
        if len(self.current_path) < 2:
            return False
            
        # Check if path is connected
        for i in range(len(self.current_path) - 1):
            if not self.topology.has_edge(self.current_path[i], self.current_path[i + 1]):
                return False
                
        return True
        
    def get_path_length(self) -> float:
        """Calculate total length of current path"""
        if not self.is_path_valid():
            return float('inf')
            
        length = 0.0
        for i in range(len(self.current_path) - 1):
            length += self._calculate_link_cost(
                self.current_path[i], 
                self.current_path[i + 1]
            )
        return length

    def reset(self, only_episode_counters=True):
        """Reset environment"""
        if only_episode_counters:
            # Only reset episode counters
            self.episode_services_processed = 0
            self.episode_services_accepted = 0
            return self.observation()
            
        # Full reset
        self._events = []
        self.current_time = 0
        self.services_processed = 0
        self.services_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        
        self.topology.graph["services"] = []
        self.topology.graph["running_services"] = []
        
        return self.observation()
    

    def _add_release(self, service: Service) -> None:
        """
        Adds an event to the event list of the simulator.
        This implementation is based on the functionalities of heapq:
        https://docs.python.org/2/library/heapq.html

        :param event:
        :return: None
        """
        heapq.heappush(self._events, (service.arrival_time + service.holding_time, service))
        

    def _get_node_pair(self) -> Tuple[str, int, str, int]:
        """
        Uses the `node_request_probabilities` variable to generate a source and a destination.

        :return: source node, source node id, destination node, destination node id
        """
        src = self.rng.choices([x for x in self.topology.nodes()], weights=self.node_request_probabilities)[0]
        src_id = self.topology.graph["node_indices"].index(src)
        new_node_probabilities = np.copy(self.node_request_probabilities)
        new_node_probabilities[src_id] = 0.0
        new_node_probabilities = new_node_probabilities / np.sum(new_node_probabilities)
        dst = self.rng.choices(
            [x for x in self.topology.nodes()], weights=new_node_probabilities
        )[0]
        dst_id = self.topology.graph["node_indices"].index(dst)
        return src, src_id, dst, dst_id

    def observation(self):
        return {"topology": self.topology, "service": self.current_service}

    def reward(self):
        return 1 if self.current_service.accepted else 0
    
    def seed(self, seed=None):
        if seed is not None:
            self.rand_seed = seed
        else:
            self.rand_seed = 41
        self.rng = random.Random(self.rand_seed)
