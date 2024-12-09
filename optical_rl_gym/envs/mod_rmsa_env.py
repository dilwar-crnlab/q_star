import copy
import functools
import heapq
import logging
import math
from collections import defaultdict
from typing import Optional, Sequence, Tuple, List, Set

import gym
import networkx as nx
import numpy as np
import torch

from optical_rl_gym.utils import Path, Service
from .mod_optical_network_env import ModifiedOpticalNetworkEnv

class ModifiedRMSAEnv(ModifiedOpticalNetworkEnv):
    """
    Modified RMSA Environment incorporating Q* path finding
    """
    metadata = {
        "metrics": [
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
            "path_selection_success_rate",
            "path_length_efficiency",
        ]
    }

    def __init__(
        self,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = None,
        mean_service_holding_time: float = 10.0,
        mean_service_inter_arrival_time: float = None,
        num_spectrum_resources: int = None,
        bit_rate_selection: str = "discrete",
        bit_rates: Sequence = [10, 40, 100],
        bit_rate_probabilities: Optional[np.array] = None,
        node_request_probabilities: Optional[np.array] = None,
        bit_rate_lower_bound: float = 25.0,
        bit_rate_higher_bound: float = 100.0,
        seed: Optional[int] = None,
        allow_rejection: bool = False,
        reset: bool = True,
        channel_width: float = 12.5,
    ):
        

        # Initialize basic attributes first
        self.num_spectrum_resources = num_spectrum_resources
        
        # Create initial topology attributes
        self.topology = copy.deepcopy(topology)
        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            dtype=int
        )
        super().__init__(
            topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            channel_width=channel_width,
        )

        # Path finding specific attributes
        self.path_finding_phase = False
        self.current_path: List[int] = []
        self.visited_nodes: Set[int] = set()
        self.path_hops_limit = self.topology.number_of_nodes() - 1
        
        # Initialize link costs
        self.topology.graph['link_costs'] = {}
        self._initialize_link_costs()

        # Assert modulations are set
        assert "modulations" in self.topology.graph

        # Bit rate selection setup
        assert bit_rate_selection in ["continuous", "discrete"]
        self.bit_rate_selection = bit_rate_selection
        
        if self.bit_rate_selection == "continuous":
            self.bit_rate_lower_bound = bit_rate_lower_bound
            self.bit_rate_higher_bound = bit_rate_higher_bound
            self.bit_rate_function = functools.partial(
                self.rng.randint, self.bit_rate_lower_bound, self.bit_rate_higher_bound
            )
        else:
            if bit_rate_probabilities is None:
                bit_rate_probabilities = [1.0 / len(bit_rates) for _ in range(len(bit_rates))]
            self.bit_rate_probabilities = bit_rate_probabilities
            self.bit_rates = bit_rates
            self.bit_rate_function = functools.partial(
                self.rng.choices, self.bit_rates, self.bit_rate_probabilities, k=1
            )
            
            # Initialize histograms
            self.bit_rate_requested_histogram = defaultdict(int)
            self.bit_rate_provisioned_histogram = defaultdict(int)
            self.episode_bit_rate_requested_histogram = defaultdict(int)
            self.episode_bit_rate_provisioned_histogram = defaultdict(int)
            self.slots_requested_histogram = defaultdict(int)
            self.episode_slots_requested_histogram = defaultdict(int)
            self.slots_provisioned_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)

        # Tracking metrics
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        # Initialize link costs
        self.topology.graph['link_costs'] = {}
        self._initialize_link_costs()

        # # Initialize spectrum resources first
        # self.topology.graph["available_slots"] = np.ones(
        #     (self.topology.number_of_edges(), self.num_spectrum_resources),
        #     dtype=int
        # )

        # Initialize spectrum tracking
        self.spectrum_usage = np.zeros(
            (self.topology.number_of_edges(), self.num_spectrum_resources), 
            dtype=int
        )
        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=int,
        )

        # Define action spaces
        self.action_space = gym.spaces.Dict({
            'path_action': gym.spaces.Discrete(self.topology.number_of_nodes()),
            'spectrum_action': gym.spaces.Discrete(self.num_spectrum_resources)
        })

        # Define observation spaces
        path_obs_size = (
            2 * self.topology.number_of_nodes() +  # Source and destination one-hot
            2 * self.topology.number_of_edges()    # Link distance and fragmentation
        )
        
        self.path_observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(path_obs_size,),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            'phase': gym.spaces.Discrete(2),
            'path_obs': self.path_observation_space,
            'spectrum_obs': gym.spaces.Box(
                low=-1, high=1,
                shape=(2 * self.topology.number_of_nodes() + 3,),
                dtype=np.float32
            )
        })

        # Initialize logging
        self.logger = logging.getLogger("rmsaenv")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of messages. "
                "Set it to INFO if DEBUG is not necessary."
            )

        # Reset environment
        self._new_service = False
        if reset:
            self.reset(only_episode_counters=False)
            
    def _initialize_link_costs(self):
        """Initialize link costs for all edges"""
        for node1, node2 in self.topology.edges():
            self.topology.graph['link_costs'][(node1, node2)] = self._calculate_link_cost(node1, node2)
            self.topology.graph['link_costs'][(node2, node1)] = self.topology.graph['link_costs'][(node1, node2)]

    def reset(self, only_episode_counters=True):
        """Reset environment and initialize path finding"""
        # Reset episode specific counters
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        if self.bit_rate_selection == "discrete":
            self.episode_bit_rate_requested_histogram.clear()
            self.episode_bit_rate_provisioned_histogram.clear()
            self.episode_slots_requested_histogram.clear()
            self.episode_slots_provisioned_histogram.clear()

        if only_episode_counters:
            if self._new_service:
                self.episode_services_processed += 1
                self.episode_bit_rate_requested += self.current_service.bit_rate
                
                if self.bit_rate_selection == "discrete":
                    self.episode_bit_rate_requested_histogram[
                        self.current_service.bit_rate
                    ] += 1
            return self.observation()

        # Full reset
        super().reset()
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        
        # Reset spectrum resources
        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges(), self.num_spectrum_resources), 
            dtype=int
        )
        self.spectrum_slots_allocation.fill(-1)

        # Reset network stats
        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0
        
        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["external_fragmentation"] = 0.0
            self.topology[lnk[0]][lnk[1]]["compactness"] = 0.0

        # Initialize path finding
        self._new_service = False
        self._next_service()
        self.path_finding_phase = True
        self.current_path = [self.current_service.source_id]
        self.visited_nodes = {self.current_service.source_id}
        
        return self.observation()
    

    
    


    
    def _calculate_link_cost(self, node1: int, node2: int) -> float:
        """Calculate combined cost of using a link based on distance and fragmentation"""
        # Get base distance
        distance = self.topology[node1][node2]['weight']
        
        # Get link fragmentation
        link_idx = self.topology[node1][node2]['index']
        slots = self.topology.graph['available_slots'][link_idx]
        fragmentation = self._calculate_fragmentation(slots)
        
        # Combine distance and fragmentation as cost
        # Higher fragmentation means higher cost
        return distance * (1 + fragmentation)
    
    def _calculate_fragmentation(self, slots) -> float:
        """Calculate fragmentation ratio of available slots"""
        if np.sum(slots) == 0:  # If no slots available
            return float('inf')
            
        # Count number of spectrum fragments
        initial_indices, values, lengths = self.rle(slots)
        
        # Get indices of unused blocks
        unused_blocks = [i for i, x in enumerate(values) if x == 1]
        
        if len(unused_blocks) > 0:
            max_block = max(lengths[unused_blocks])
            total_free = np.sum(slots)
            return 1 - (max_block / total_free)
        else:
            return float('inf')
    
    def step(self, action):
        """Handle both path finding and spectrum assignment phases"""
        if self.path_finding_phase:
            return self._handle_path_finding(action['path_action'])
        else:
            return self._handle_spectrum_assignment(action['spectrum_action'])
    
    def _handle_path_finding(self, action: int):
        """Process one step of path finding"""
        # Convert action to 0-based index if needed
        action = int(action)
        
        # Validate action
        if action not in self._get_valid_next_nodes():
            return self.observation(), -1.0, True, {'invalid_move': True}
                
        # Store current state for metrics
        # Convert path nodes to strings for distance calculation
        if len(self.current_path) > 0:
            current_str = str(self.current_path[-1] + 1)
            dest_str = str(self.current_service.destination_id + 1)
            try:
                prev_distance = nx.shortest_path_length(
                    self.topology,
                    current_str,
                    dest_str,
                    weight='weight'
                )
            except nx.NetworkXError:
                prev_distance = float('inf')
        else:
            prev_distance = float('inf')
            
        # Add node to path
        self.current_path.append(action)
        self.visited_nodes.add(action)
        
        # Check if destination reached
        if action == self.current_service.destination_id:
            # Path complete - evaluate quality and transition to spectrum assignment
            path_object = self._create_path_object()
            self.path_finding_phase = False
            path_quality = self._evaluate_path_quality(path_object)
            
            return self.observation(), path_quality, False, {
                'path_complete': True,
                'path_length': len(self.current_path) - 1,
                'path_quality': path_quality
            }
        
        # Calculate reward based on progress
        reward = self._calculate_path_progress_reward(prev_distance)
    
        return self.observation(), reward, False, {}
    
    def _get_valid_next_nodes(self) -> List[int]:
        """Get list of valid next nodes during path finding"""
        if not self.current_path:
            return []
            
        current_str = self.node_ids[self.current_path[-1]]
        valid_nodes = []
        
        for neighbor in self.topology.neighbors(current_str):
            # Convert string node to index using node_map
            neighbor_idx = self.node_map[int(neighbor)]
            if neighbor_idx not in self.visited_nodes:
                valid_nodes.append(neighbor_idx)
                    
        return valid_nodes
    
    def _get_current_path_length(self) -> float:
        """Calculate current partial path length"""
        if len(self.current_path) < 2:
            return 0.0
            
        length = 0.0
        for i in range(len(self.current_path) - 1):
            length += self.topology[self.current_path[i]][self.current_path[i+1]]['weight']
            
        return length
    
    def _calculate_path_progress_reward(self, prev_distance: float) -> float:
        """Calculate reward based on progress towards destination"""
        if len(self.current_path) < 2:
            return 0.0
            
        # Get current node and destination, convert to strings for networkx
        current_str = str(self.current_path[-1] + 1)  # Convert 0-based index to 1-based string
        dest_str = str(self.current_service.destination_id + 1)
        
        try:
            current_distance = nx.shortest_path_length(
                self.topology, 
                current_str,
                dest_str,
                weight='weight'
            )
            
            # Reward based on progress
            distance_progress = prev_distance - current_distance
            
            # Small penalty for each hop to encourage shorter paths
            hop_penalty = -0.1
            
            return distance_progress + hop_penalty
            
        except nx.NetworkXError:
            print(f"Debug - Current node: {current_str}")
            print(f"Debug - Destination node: {dest_str}")
            print(f"Debug - Topology nodes: {list(self.topology.nodes())}")
            return -1.0  # Penalize moving to disconnected component
    
    def _evaluate_path_quality(self, path: Path) -> float:
        """Evaluate quality of found path"""
        if not path or len(path.node_list) < 2:
            return -1.0
            
        total_length = 0.0
        total_fragmentation = 0.0
        
        # Calculate path metrics
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i+1]
            
            # Add distance
            total_length += self.topology[node1][node2]['weight']
            
            # Add fragmentation
            link_idx = self.topology[node1][node2]['index']
            slots = self.topology.graph['available_slots'][link_idx]
            total_fragmentation += self._calculate_fragmentation(slots)
        
        # Normalize metrics
        avg_fragmentation = total_fragmentation / (len(path.node_list) - 1)
        normalized_length = total_length / self.topology.graph['longest_path_length']
        
        # Combine metrics (can be tuned)
        quality = 1.0 / (1.0 + normalized_length + avg_fragmentation)
        
        return quality
    
    def _create_path_object(self) -> Path:
        """Convert current path to Path object"""
        if not self.current_path or len(self.current_path) < 2:
            return None
                
        # Convert path indices to node strings
        path_nodes = [self.node_ids[idx] for idx in self.current_path]
        
        # Calculate path length using string node IDs
        path_length = sum(
            self.topology[path_nodes[i]][path_nodes[i+1]]['weight']
            for i in range(len(path_nodes) - 1)
        )
        
        # Debug print to check available modulations and their attributes
        #print("Available modulations:", self.topology.graph['modulations'])
        #print("First modulation attributes:", vars(self.topology.graph['modulations'][0]))
            
        # Find best modulation for path length
        best_modulation = None
        for modulation in sorted(
            self.topology.graph['modulations'],
            key=lambda x: x.spectral_efficiency,
            reverse=True
        ):
            # You might need to use a different attribute here
            if path_length <= modulation.maximum_length:  # Changed from reach to tr
                best_modulation = modulation
                break
                    
        if best_modulation is None:
            best_modulation = self.topology.graph['modulations'][-1]
                
        return Path(
            node_list=path_nodes,
            weight=path_length,
            best_modulation=best_modulation
        )
        
    def observation(self):
        """Get current observation based on phase"""
        observation = {
            'phase': 0 if self.path_finding_phase else 1,
            'path_obs': self._get_path_observation(),
            'spectrum_obs': self._get_spectrum_observation()
        }
        return observation
        
    def _get_path_observation(self):
        """Get observation for path finding phase"""
        obs = []
        
        # Source and destination one-hot encodings
        source_one_hot = np.zeros(self.topology.number_of_nodes())
        source_one_hot[self.current_service.source_id] = 1
        
        dest_one_hot = np.zeros(self.topology.number_of_nodes())
        dest_one_hot[self.current_service.destination_id] = 1
        
        obs.extend(source_one_hot)
        obs.extend(dest_one_hot)
        
        # Add link states
        for node1, node2 in self.topology.edges():
            # Normalized distance
            distance = self.topology[node1][node2]['weight']
            norm_distance = distance / self.topology.graph['longest_path_length']
            obs.append(norm_distance)
            
            # Link fragmentation
            fragmentation = self._calculate_fragmentation(
                self.topology.graph['available_slots'][
                    self.topology[node1][node2]["index"]
                ]
            )
            obs.append(fragmentation)
        
        return np.array(obs, dtype=np.float32)



    
    def _handle_spectrum_assignment(self, action: int):
        """Handle spectrum assignment phase"""
        path_object = self._create_path_object()
        if path_object is None:
            return self.observation(), -1.0, True, {'invalid_path': True}
            
        # Get required number of slots
        num_slots = self.get_number_slots(path_object)
        
        # Check if spectrum assignment is possible
        if self.is_path_free(path_object, action, num_slots):
            self._provision_path(path_object, action, num_slots)
            self.current_service.accepted = True
            reward = 1.0
        else:
            self.current_service.accepted = False
            reward = -1.0
            
        # Update statistics
        self.topology.graph["services"].append(self.current_service)
        
        # Generate episode info
        info = self._generate_episode_info()
        
        # Prepare for next service
        self._new_service = False
        self._next_service()
        
        return self.observation(), reward, False, info
        
    def _provision_path(self, path: Path, initial_slot: int, number_slots: int):
        """Provision service on selected path and spectrum"""
        if not self.is_path_free(path, initial_slot, number_slots):
            raise ValueError(
                f"Path {path.node_list} does not have enough capacity on slots {initial_slot}-{initial_slot + number_slots}"
            )
            
        self.logger.debug(
            f"{self.current_service.service_id} assigning path {path.node_list} "
            f"on initial slot {initial_slot} for {number_slots} slots"
        )
        
        # Assign spectrum slots
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            link_idx = self.topology[node1][node2]["index"]
            
            # Mark slots as used
            self.topology.graph["available_slots"][
                link_idx,
                initial_slot : initial_slot + number_slots
            ] = 0
            
            # Record service allocation
            self.spectrum_slots_allocation[
                link_idx,
                initial_slot : initial_slot + number_slots
            ] = self.current_service.service_id
            
            # Update service lists
            self.topology[node1][node2]["services"].append(self.current_service)
            self.topology[node1][node2]["running_services"].append(self.current_service)
            
            # Update link stats and costs
            self._update_link_stats(node1, node2)
            self.topology.graph['link_costs'][(node1, node2)] = self._calculate_link_cost(node1, node2)
            
        # Update network state
        self.topology.graph["running_services"].append(self.current_service)
        self.current_service.path = path
        self.current_service.initial_slot = initial_slot
        self.current_service.number_slots = number_slots
        self._update_network_stats()
        
        # Update counters
        self.services_accepted += 1
        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned += self.current_service.bit_rate
        
        # Update histograms for discrete bit rates
        if self.bit_rate_selection == "discrete":
            self.slots_provisioned_histogram[number_slots] += 1
            self.bit_rate_provisioned_histogram[self.current_service.bit_rate] += 1
            self.episode_bit_rate_provisioned_histogram[self.current_service.bit_rate] += 1
            
        # Schedule release
        self._add_release(self.current_service)
        
    def _release_path(self, service: Service):
        """Release resources allocated to a service"""
        for i in range(len(service.path.node_list) - 1):
            node1, node2 = service.path.node_list[i], service.path.node_list[i + 1]
            link_idx = self.topology[node1][node2]["index"]
            
            # Free spectrum slots
            self.topology.graph["available_slots"][
                link_idx,
                service.initial_slot : service.initial_slot + service.number_slots
            ] = 1
            
            # Clear service allocation
            self.spectrum_slots_allocation[
                link_idx,
                service.initial_slot : service.initial_slot + service.number_slots
            ] = -1
            
            # Update service lists
            self.topology[node1][node2]["running_services"].remove(service)
            
            # Update metrics
            self._update_link_stats(node1, node2)
            self.topology.graph['link_costs'][(node1, node2)] = self._calculate_link_cost(node1, node2)
            
        self.topology.graph["running_services"].remove(service)
        
    def _next_service(self):
        """Generate next service request"""
        if self._new_service:
            return
            
        # Generate arrival time
        at = self.current_time + self.rng.expovariate(
            1 / self.mean_service_inter_arrival_time
        )
        self.current_time = at
        
        # Generate holding time
        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        
        # Select source and destination
        src, src_id, dst, dst_id = self._get_node_pair()
        
        # Generate bit rate
        bit_rate = (
            self.bit_rate_function()
            if self.bit_rate_selection == "continuous"
            else self.bit_rate_function()[0]
        )
        
        # Create new service
        self.current_service = Service(
            self.episode_services_processed,
            src, src_id,
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate
        )
        
        # Initialize path finding for new service
        self.path_finding_phase = True
        self.current_path = [src_id]
        self.visited_nodes = {src_id}
        
        # Update counters
        self._new_service = True
        self.services_processed += 1
        self.episode_services_processed += 1
        
        # Update bit rate statistics
        self.bit_rate_requested += bit_rate
        self.episode_bit_rate_requested += bit_rate
        
        if self.bit_rate_selection == "discrete":
            self.bit_rate_requested_histogram[bit_rate] += 1
            self.episode_bit_rate_requested_histogram[bit_rate] += 1
            
            # Track slots for shortest path
            slots = self.get_number_slots(self.k_shortest_paths[src, dst][0])
            self.slots_requested_histogram[slots] += 1
            self.episode_slots_requested_histogram[slots] += 1
            
        # Process pending releases
        self._process_releases()
        
    def _process_releases(self):
        """Process all pending service releases up to current time"""
        while len(self._events) > 0:
            time, service = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service)
            else:
                self._add_release(service)
                break
                
    def _generate_episode_info(self):
        """Generate information dictionary for current episode"""
        info = {
            "service_blocking_rate": (self.services_processed - self.services_accepted)
            / self.services_processed,
            "episode_service_blocking_rate": (
                self.episode_services_processed - self.episode_services_accepted
            )
            / self.episode_services_processed,
            "bit_rate_blocking_rate": (
                self.bit_rate_requested - self.bit_rate_provisioned
            )
            / self.bit_rate_requested,
            "episode_bit_rate_blocking_rate": (
                self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
            )
            / self.episode_bit_rate_requested,
            "network_compactness": self._get_network_compactness(),
            "avg_link_utilization": np.mean([
                self.topology[lnk[0]][lnk[1]]["utilization"]
                for lnk in self.topology.edges()
            ])
        }
        
        # Add bit rate specific metrics for discrete case
        if self.bit_rate_selection == "discrete":
            blocking_per_bit_rate = self._calculate_blocking_per_bit_rate()
            for bit_rate, blocking in blocking_per_bit_rate.items():
                info[f"bit_rate_blocking_{bit_rate}"] = blocking
            info["fairness"] = max(blocking_per_bit_rate.values()) - min(
                blocking_per_bit_rate.values()
            )
            
        return info



    def _get_spectrum_observation(self):
        """Get observation for spectrum assignment phase"""
        if not self.current_path:
            return np.zeros_like(self.observation_space['spectrum_obs'].low)
            
        path_object = self._create_path_object()
        if path_object is None:
            return np.zeros_like(self.observation_space['spectrum_obs'].low)
            
        spectrum_obs = np.full((2 * self.j + 3,), fill_value=-1.0)
        
        # Get available blocks
        available_slots = self.get_available_slots(path_object)
        initial_indices, values, lengths = self.rle(available_slots)
        
        # Get slots needed
        slots_needed = self.get_number_slots(path_object)
        
        # Fill spectrum information
        available_blocks = [i for i, x in enumerate(values) if x == 1 and lengths[i] >= slots_needed]
        
        for idx, block_idx in enumerate(available_blocks[:self.j]):
            # Initial slot index normalized
            spectrum_obs[idx * 2] = (
                2 * (initial_indices[block_idx] - 0.5 * self.num_spectrum_resources)
                / self.num_spectrum_resources
            )
            # Block length normalized
            spectrum_obs[idx * 2 + 1] = (lengths[block_idx] - slots_needed) / slots_needed
            
        # Add path metrics
        spectrum_obs[2 * self.j] = slots_needed / self.num_spectrum_resources
        spectrum_obs[2 * self.j + 1] = np.sum(available_slots) / self.num_spectrum_resources
        spectrum_obs[2 * self.j + 2] = len(available_blocks) / self.j
        
        return spectrum_obs

    def get_number_slots(self, path: Path) -> int:
        """Calculate number of slots needed for current service on given path"""
        if not path or not path.best_modulation:
            return self.num_spectrum_resources + 1  # Invalid path
            
        return math.ceil(
            self.current_service.bit_rate /
            (path.best_modulation.spectral_efficiency * self.channel_width)
        ) + 1  # Adding guard band
        
    def get_available_slots(self, path: Path) -> np.ndarray:
        """Get available slots across the whole path"""
        if not path or len(path.node_list) < 2:
            return np.zeros(self.num_spectrum_resources)
            
        # Get slots available on all links of the path
        available = np.ones(self.num_spectrum_resources)
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            link_idx = self.topology[node1][node2]["index"]
            available &= self.topology.graph["available_slots"][link_idx]
            
        return available
        
    def is_path_free(self, path: Path, initial_slot: int, number_slots: int) -> bool:
        """Check if path has enough contiguous free spectrum slots"""
        if not path or initial_slot + number_slots > self.num_spectrum_resources:
            return False
            
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            link_idx = self.topology[node1][node2]["index"]
            
            if np.any(
                self.topology.graph["available_slots"][
                    link_idx,
                    initial_slot : initial_slot + number_slots
                ] == 0
            ):
                return False
        return True

    def _update_network_stats(self):
        """Update network-wide statistics"""
        last_update = self.topology.graph["last_update"]
        time_diff = self.current_time - last_update
        
        if self.current_time > 0:
            # Update throughput
            last_throughput = self.topology.graph["throughput"]
            cur_throughput = sum(
                service.bit_rate 
                for service in self.topology.graph["running_services"]
            )
            
            self.topology.graph["throughput"] = (
                (last_throughput * last_update) + (cur_throughput * time_diff)
            ) / self.current_time
            
            # Update network compactness
            last_compactness = self.topology.graph["compactness"]
            cur_compactness = self._get_network_compactness()
            
            self.topology.graph["compactness"] = (
                (last_compactness * last_update) + (cur_compactness * time_diff)
            ) / self.current_time
            
        self.topology.graph["last_update"] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        """Update statistics for a specific link"""
        last_update = self.topology[node1][node2]["last_update"]
        time_diff = self.current_time - last_update
        
        if self.current_time > 0:
            # Update utilization
            last_util = self.topology[node1][node2]["utilization"]
            cur_util = self._get_link_utilization(node1, node2)
            
            self.topology[node1][node2]["utilization"] = (
                (last_util * last_update) + (cur_util * time_diff)
            ) / self.current_time
            
            # Update fragmentation metrics
            self._update_link_fragmentation(node1, node2, last_update, time_diff)
            
        self.topology[node1][node2]["last_update"] = self.current_time

    def _get_link_utilization(self, node1: str, node2: str) -> float:
        """Calculate current utilization of a link"""
        link_idx = self.topology[node1][node2]["index"]
        used_slots = np.sum(
            self.topology.graph["available_slots"][link_idx] == 0
        )
        return used_slots / self.num_spectrum_resources

    def _update_link_fragmentation(self, node1: str, node2: str, last_update: float, time_diff: float):
        """Update fragmentation metrics for a link"""
        link_idx = self.topology[node1][node2]["index"]
        slots = self.topology.graph["available_slots"][link_idx]
        
        # Get current metrics
        cur_external_frag = 0.0
        cur_compactness = 0.0
        
        if np.sum(slots) > 0:
            initial_indices, values, lengths = self.rle(slots)
            
            # Calculate external fragmentation
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            if len(unused_blocks) > 1:
                max_empty = max(lengths[unused_blocks])
                cur_external_frag = 1.0 - (max_empty / np.sum(slots))
            
            # Calculate compactness
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                
                unused_slots = np.sum(slots[lambda_min:lambda_max])
                if unused_slots > 0:
                    cur_compactness = (
                        (lambda_max - lambda_min) / np.sum(1 - slots)
                    ) * (1 / unused_slots)
                else:
                    cur_compactness = 1.0
            else:
                cur_compactness = 1.0
        
        # Update time-averaged metrics
        self.topology[node1][node2]["external_fragmentation"] = (
            (self.topology[node1][node2]["external_fragmentation"] * last_update) +
            (cur_external_frag * time_diff)
        ) / self.current_time
        
        self.topology[node1][node2]["compactness"] = (
            (self.topology[node1][node2]["compactness"] * last_update) +
            (cur_compactness * time_diff)
        ) / self.current_time

    def _get_network_compactness(self) -> float:
        """Calculate current network-wide spectrum compactness"""
        sum_slots_paths = 0
        for service in self.topology.graph["running_services"]:
            sum_slots_paths += service.number_slots * len(service.path.node_list)
            
        sum_occupied = 0
        sum_unused_blocks = 0
        
        for node1, node2 in self.topology.edges():
            link_idx = self.topology[node1][node2]["index"]
            slots = self.topology.graph["available_slots"][link_idx]
            
            initial_indices, values, lengths = self.rle(slots)
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += lambda_max - lambda_min
                
                unused_slots = np.sum(slots[lambda_min:lambda_max])
                sum_unused_blocks += unused_slots
        
        if sum_unused_blocks > 0:
            return (sum_occupied / sum_slots_paths) * (
                self.topology.number_of_edges() / sum_unused_blocks
            )
        return 1.0

    @staticmethod
    def rle(inarray):
        """Run length encoding helper"""
        ia = np.asarray(inarray)
        n = len(ia)
        if n == 0:
            return (None, None, None)
            
        y = np.array(ia[1:] != ia[:-1])
        i = np.append(np.where(y), n - 1)
        z = np.diff(np.append(-1, i))
        p = np.cumsum(np.append(0, z))[:-1]
        return p, ia[i], z

    # def reset(self, only_episode_counters=True):
    #     """Reset environment state"""
    #     if only_episode_counters:
    #         self.episode_services_processed = 0
    #         self.episode_services_accepted = 0
    #         self.episode_bit_rate_requested = 0
    #         self.episode_bit_rate_provisioned = 0
            
    #         if self._new_service:
    #             # Account for current service
    #             self.episode_services_processed += 1
    #             self.episode_bit_rate_requested += self.current_service.bit_rate
                
    #         return self.observation()
            
    #     # Full reset
    #     super().reset()
    #     self.bit_rate_requested = 0
    #     self.bit_rate_provisioned = 0
        
    #     # Reset path finding state
    #     self.path_finding_phase = True
    #     self.current_path = []
    #     self.visited_nodes = set()
        
    #     # Initialize network state
    #     self.topology.graph["available_slots"] = np.ones(
    #         (self.topology.number_of_edges(), self.num_spectrum_resources), 
    #         dtype=int
    #     )
        
    #     self._new_service = False
    #     self._next_service()
        
    #     return self.observation()