# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Test port assignment logic with availability checking."""

import socket
import unittest

from alpasim_wizard.context import create_port_assigner


class TestPortAssignment(unittest.TestCase):
    """Test the port assignment functionality."""

    def test_basic_port_assignment(self) -> None:
        """Test that ports are assigned starting from baseport."""
        baseport = 30000
        port_iter = create_port_assigner(baseport)

        # Get first few ports
        ports = [next(port_iter) for _ in range(5)]

        # Should start from baseport or higher
        self.assertGreaterEqual(ports[0], baseport)

        # Should be sequential (assuming no ports are occupied)
        for i in range(1, len(ports)):
            self.assertGreater(ports[i], ports[i - 1])

    def test_skip_occupied_port(self) -> None:
        """Test that occupied ports are skipped."""
        baseport = 31000

        # Create a server socket to occupy a port
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            # Bind to baseport + 1 to simulate an occupied port
            server_socket.bind(("localhost", baseport + 1))
            server_socket.listen(1)

            # Create port assigner
            port_iter = create_port_assigner(baseport)

            # Get first three ports
            port1 = next(port_iter)
            port2 = next(port_iter)
            port3 = next(port_iter)

            # First port should be baseport (assuming it's free)
            self.assertEqual(port1, baseport)

            # Second port should skip baseport + 1 (occupied)
            self.assertEqual(port2, baseport + 2)

            # Third port should be baseport + 3
            self.assertEqual(port3, baseport + 3)

        finally:
            server_socket.close()

    def test_find_first_available_when_baseport_occupied(self) -> None:
        """Test finding first available port when baseport is occupied."""
        baseport = 32000

        # Occupy the baseport
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server_socket.bind(("localhost", baseport))
            server_socket.listen(1)

            # Create port assigner
            port_iter = create_port_assigner(baseport)

            # First port should be baseport + 1
            first_port = next(port_iter)
            self.assertEqual(first_port, baseport + 1)

        finally:
            server_socket.close()

    def test_max_ports_limit(self) -> None:
        """Test that max 1000 ports are assigned."""
        baseport = 33000
        port_iter = create_port_assigner(baseport)

        # Get 1000 ports successfully
        ports = []
        for _ in range(1000):
            ports.append(next(port_iter))

        self.assertEqual(len(ports), 1000)
        self.assertEqual(len(set(ports)), 1000)  # All unique

        # 1001st port should raise AssertionError
        with self.assertRaises(AssertionError) as ctx:
            next(port_iter)

        self.assertIn("1000 different port numbers", str(ctx.exception))

    def test_multiple_occupied_ports(self) -> None:
        """Test behavior with multiple occupied ports."""
        baseport = 34000
        occupied_offsets = [0, 2, 4]  # Occupy baseport, baseport+2, baseport+4

        server_sockets = []
        try:
            # Create server sockets for occupied ports
            for offset in occupied_offsets:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", baseport + offset))
                sock.listen(1)
                server_sockets.append(sock)

            # Create port assigner
            port_iter = create_port_assigner(baseport)

            # Get first few ports
            ports = [next(port_iter) for _ in range(5)]

            # Should skip occupied ports
            expected_ports = [
                baseport + 1,  # Skip baseport (occupied)
                baseport + 3,  # Skip baseport+2 (occupied)
                baseport + 5,  # Skip baseport+4 (occupied)
                baseport + 6,
                baseport + 7,
            ]

            self.assertEqual(ports, expected_ports)

        finally:
            for sock in server_sockets:
                sock.close()


if __name__ == "__main__":
    unittest.main()
