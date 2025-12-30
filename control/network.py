import socket
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class UDPTransport(ABC):
    """Abstract base class for UDP network transport."""
    
    def __init__(self, host: str, port: int):
        """
        Initialize the transport.
        
        Args:
            host: Target IP address
            port: Target UDP port
        """
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish the connection (create socket, connect to network if needed).
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close the connection and release resources."""
        pass
    
    def send(self, data: bytes) -> int:
        """
        Send data to the target.
        
        Args:
            data: Bytes to send
            
        Returns:
            Number of bytes sent
            
        Raises:
            ConnectionError: If not connected
        """
        if not self._connected or self.sock is None:
            raise ConnectionError("Not connected. Call connect() first.")
        return self.sock.sendto(data, (self.host, self.port))
    
    @property
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected
    
    @property
    def address(self) -> Tuple[str, int]:
        """Get target address as (host, port) tuple."""
        return (self.host, self.port)
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


class DirectUDP(UDPTransport):
    """
    Direct UDP transport using the current network connection.
    
    Use this when ESP32 and host are on the same network.
    """
    
    def __init__(self, host: str, port: int):
        """
        Initialize direct UDP transport.
        
        Args:
            host: Target IP address (e.g., "192.168.1.228")
            port: Target UDP port (e.g., 4210)
        """
        super().__init__(host, port)
    
    def connect(self) -> bool:
        """Create UDP socket."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._connected = True
            return True
        except Exception as e:
            print(f"Failed to create socket: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close UDP socket."""
        if self.sock is not None:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self._connected = False


class EphemeralWiFiUDP(UDPTransport):
    """
    UDP transport that first connects to an ESP32 SoftAP network.
    
    Use this when ESP32 is running as a WiFi Access Point.
    The connection will automatically disconnect and restore
    previous network when done.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        wifi_ssid: str,
        wifi_password: str,
        interface: str = "wlan0",
        hidden: bool = True,
        dhcp_wait: float = 3.0,
    ):
        """
        Initialize ephemeral WiFi UDP transport.
        
        Args:
            host: Target IP address (e.g., "192.168.4.1" for ESP32 SoftAP)
            port: Target UDP port
            wifi_ssid: SSID of the ESP32 SoftAP network
            wifi_password: Password for the network
            interface: Network interface to use (default: "wlan0")
            hidden: Whether the network is hidden (default: True)
            dhcp_wait: Seconds to wait for DHCP after connecting (default: 3.0)
        """
        super().__init__(host, port)
        self.wifi_ssid = wifi_ssid
        self.wifi_password = wifi_password
        self.interface = interface
        self.hidden = hidden
        self.dhcp_wait = dhcp_wait
    
    def _run_cmd(self, cmd: str) -> bool:
        """Run a shell command, suppressing output."""
        try:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def connect(self) -> bool:
        """Connect to WiFi network and create UDP socket."""
        print(f"[*] Connecting to AP '{self.wifi_ssid}'...")
        
        # Build nmcli command
        hidden_flag = "hidden yes" if self.hidden else ""
        cmd = (
            f"nmcli device wifi connect '{self.wifi_ssid}' "
            f"password '{self.wifi_password}' "
            f"{hidden_flag} ifname {self.interface}"
        )
        
        if not self._run_cmd(cmd):
            print(f"[!] Failed to connect to {self.wifi_ssid}")
            return False
        
        print(f"[*] Connected! Waiting {self.dhcp_wait}s for DHCP...")
        time.sleep(self.dhcp_wait)
        
        # Create UDP socket
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._connected = True
            print(f"[*] Ready to send to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[!] Failed to create socket: {e}")
            self._disconnect_wifi()
            return False
    
    def _disconnect_wifi(self):
        """Disconnect from WiFi and delete the connection profile."""
        print(f"\n[*] Disconnecting from '{self.wifi_ssid}'...")
        self._run_cmd(f"nmcli connection delete id '{self.wifi_ssid}'")
        print("[*] Connection deleted. Host should revert to previous WiFi.")
    
    def disconnect(self):
        """Close socket and disconnect from WiFi."""
        if self.sock is not None:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        if self._connected:
            self._disconnect_wifi()
        
        self._connected = False


def create_transport(
    host: str,
    port: int,
    use_ephemeral_wifi: bool = False,
    wifi_ssid: str = "",
    wifi_password: str = "",
    **kwargs
) -> UDPTransport:
    """
    Factory function to create a UDP transport.
    
    Args:
        host: Target IP address
        port: Target UDP port
        use_ephemeral_wifi: If True, use EphemeralWiFiUDP; otherwise use DirectUDP
        wifi_ssid: SSID for ephemeral WiFi (required if use_ephemeral_wifi=True)
        wifi_password: Password for ephemeral WiFi (required if use_ephemeral_wifi=True)
        **kwargs: Additional arguments passed to EphemeralWiFiUDP
        
    Returns:
        UDPTransport instance (not yet connected)
    """
    if use_ephemeral_wifi:
        if not wifi_ssid or not wifi_password:
            raise ValueError("wifi_ssid and wifi_password required for ephemeral WiFi")
        return EphemeralWiFiUDP(
            host=host,
            port=port,
            wifi_ssid=wifi_ssid,
            wifi_password=wifi_password,
            **kwargs
        )
    else:
        return DirectUDP(host=host, port=port)
