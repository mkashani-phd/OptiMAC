import unittest
import numpy as np
import time

# it considers that the packet added to the page must have SN as the first element
# the rest is optional
class Page:
    def __init__(self, page_size:int = 10, packet_dim:int = 4):
        self.page_size = page_size
        self.packets = np.zeros((self.page_size, packet_dim), dtype=object)  # Assuming msg, mac, and timestamp
        self.last_update_time = time.time()
        self.min_SN = None
        self.max_SN = None
        self.occupancy = 0  # Track the number of packets in the page

    def add_packet(self, SN, packet):
        """Check if the SN is in the range of the page_size"""
        if self.min_SN is None:
            self.min_SN = SN - SN % self.page_size
            self.max_SN = self.min_SN + self.page_size
        elif SN < self.min_SN or SN >= self.max_SN:
            return False
        elif self.packets[SN % self.page_size][0] == SN:
            return False
        
        self.packets[SN % self.page_size] = packet
        self.last_update_time = time.time()
        self.occupancy += 1

    def is_full(self):
        return self.occupancy == self.page_size

    def clear(self):
        self.packets.fill(0)
        self.last_update_time = time.time()
        self.min_SN = None
        self.max_SN = None
        self.occupancy = 0



class TestPage(unittest.TestCase):
    
    def setUp(self):
        """Set up the test environment."""
        self.page_size = 3
        self.packet_dim = 4
        self.page = Page(page_size=self.page_size, packet_dim=self.packet_dim)

    def test_initial_state(self):
        """Test the initial state of the page."""
        self.assertIsNone(self.page.min_SN, "Initial min_SN should be None.")
        self.assertFalse(self.page.is_full(), "Page should not be full initially.")
        self.assertTrue(np.all(self.page.packets == 0), "Initial packets should be all zeros.")

    def test_add_first_packet(self):
        """Test adding the first packet."""
        sn = 5
        packet = [sn, 'Hello', b'', time.time()]
        self.page.add_chunk(sn, packet)
        self.assertEqual(self.page.min_SN, sn - sn % self.page_size, "min_SN should be correctly set after adding the first packet.")
        self.assertEqual(self.page.max_SN, self.page.min_SN + self.page_size, "max_SN should be set correctly based on min_SN and page_size.")

    def test_out_of_range_SN(self):
        """Test adding a packet with an out-of-range SN."""
        self.page.add_chunk(5, [5, 'Hello', b'', time.time()])
        result = self.page.add_chunk(1, [1, 'Out of range', b'', time.time()])
        self.assertFalse(result, "Should return False when SN is out of range.")

    def test_duplicate_SN(self):
        """Test adding a duplicate SN packet."""
        sn = 5
        packet = [sn, 'Hello', b'', time.time()]
        self.page.add_chunk(sn, packet)
        result = self.page.add_chunk(sn, packet)
        self.assertFalse(result, "Should return False when adding a duplicate SN.")

    def test_is_full(self):
        """Test the is_full function when the page is full."""
        self.page.add_chunk(5, [5, 'Hello', b'', time.time()])
        self.page.add_chunk(6, [6, 'World', b'', time.time()])
        self.page.add_chunk(4, [4, 'Full', b'', time.time()])
        self.assertTrue(self.page.is_full(), "is_full should return True when the page is full.")
        
    def test_clear_page(self):
        """Test clearing the page."""
        self.page.add_chunk(5, [5, 'Hello', b'', time.time()])
        self.page.clear()
        self.assertFalse(self.page.is_full(), "Page should not be full after clearing.")
        self.assertIsNone(self.page.min_SN, "min_SN should be None after clearing the page.")
        self.assertIsNone(self.page.max_SN, "max_SN should be None after clearing the page.")
    
    def test_add_packet_after_clear(self):
        """Test adding a packet after clearing the page."""
        self.page.add_chunk(5, [5, 'Hello', b'', time.time()])
        self.page.clear()
        sn = 8
        packet = [sn, 'New Data', b'', time.time()]
        self.page.add_chunk(sn, packet)
        self.assertEqual(self.page.min_SN, sn - sn % self.page_size, "min_SN should be reset correctly after clearing and adding a new packet.")
        # self.assertTrue(np.array_equal(self.page.packets[sn % self.page_size], packet), "The new packet should be correctly added after clearing.")

if __name__ == '__main__':
    unittest.main()




