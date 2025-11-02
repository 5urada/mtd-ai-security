"""
Network Topology Generator for ID-HAM
Based on Waxman model with parameters α=0.2, β=0.15
"""

import networkx as nx
import matplotlib.pyplot as plt
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import random

class HAMTopology:
    """Generate network topology for HAM experiments"""
    
    def __init__(self, num_switches=5, num_hosts=30, alpha=0.2, beta=0.15):
        self.num_switches = num_switches
        self.num_hosts = num_hosts
        self.alpha = alpha
        self.beta = beta
        self.topology = None
        
    def generate_waxman_topology(self):
        """Generate Waxman random graph for switch topology"""
        # Generate Waxman graph for switches
        self.topology = nx.waxman_graph(
            self.num_switches,
            alpha=self.alpha,
            beta=self.beta,
            domain=(0, 0, 1, 1)
        )
        return self.topology
    
    def create_mininet_topology(self):
        """Create Mininet topology from generated graph"""
        info('*** Creating network topology\n')
        
        # Create Mininet network
        net = Mininet(
            controller=RemoteController,
            switch=OVSSwitch,
            autoSetMacs=True,
            autoStaticArp=False
        )
        
        info('*** Adding controller\n')
        # Connect to Ryu controller
        c0 = net.addController(
            'c0',
            controller=RemoteController,
            ip='127.0.0.1',
            port=6633
        )
        
        info('*** Adding switches\n')
        switches = []
        for i in range(self.num_switches):
            switch = net.addSwitch(f's{i+1}')
            switches.append(switch)
        
        # Add switch links based on Waxman topology
        if self.topology is None:
            self.generate_waxman_topology()
            
        info('*** Adding switch links\n')
        for edge in self.topology.edges():
            net.addLink(switches[edge[0]], switches[edge[1]])
        
        info('*** Adding hosts\n')
        hosts = []
        hosts_per_switch = self.num_hosts // self.num_switches
        host_count = 0
        
        for i, switch in enumerate(switches):
            # Distribute hosts across switches
            num_hosts_for_switch = hosts_per_switch
            if i == len(switches) - 1:
                # Last switch gets remaining hosts
                num_hosts_for_switch = self.num_hosts - host_count
            
            for j in range(num_hosts_for_switch):
                host = net.addHost(f'h{host_count+1}')
                net.addLink(host, switch)
                hosts.append(host)
                host_count += 1
        
        return net
    
    def visualize_topology(self, save_path='topology.png'):
        """Visualize the network topology"""
        if self.topology is None:
            self.generate_waxman_topology()
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.topology)
        nx.draw(
            self.topology,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=500,
            font_size=10,
            font_weight='bold'
        )
        plt.title(f'Network Topology (Waxman: α={self.alpha}, β={self.beta})')
        plt.savefig(save_path)
        plt.close()
        info(f'Topology visualization saved to {save_path}\n')


def create_small_network():
    """Create small network scenario: 5 switches, 30 hosts"""
    topo = HAMTopology(num_switches=5, num_hosts=30)
    return topo.create_mininet_topology()


def create_large_network():
    """Create large network scenario: 30 switches, 100 hosts"""
    topo = HAMTopology(num_switches=30, num_hosts=100)
    return topo.create_mininet_topology()


if __name__ == '__main__':
    setLogLevel('info')
    
    # Generate and visualize topology
    info('*** Generating topology\n')
    topo = HAMTopology(num_switches=5, num_hosts=30)
    topo.visualize_topology('network_topology.png')
    
    # Create Mininet network
    net = topo.create_mininet_topology()
    
    info('*** Starting network\n')
    net.start()
    
    info('*** Running CLI\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()
