
import unittest
import numpy as np
import time
import Page




# class Page:
#     def __init__(self, page_size:int = 10, packet_dim:int = 4):
#         self.page_size = page_size
#         self.packets = np.zeros((self.page_size, packet_dim), dtype=object)  # Assuming msg, mac, and timestamp
#         self.last_update_time = time.time()
#         self.min_SN = None
#         self.max_SN = None
#         self.occupancy = 0  # Track the number of packets in the page

#     def add_packet(self, SN, packet):
#         """Check if the SN is in the range of the page_size"""
#         if self.min_SN is None:
#             self.min_SN = SN - SN % self.page_size
#             self.max_SN = self.min_SN + self.page_size
#         elif SN < self.min_SN or SN >= self.max_SN:
#             return False
#         elif self.packets[SN % self.page_size][0] == SN:
#             return False
        
#         self.packets[SN % self.page_size] = packet
#         self.last_update_time = time.time()
#         self.occupancy += 1
#         return True

#     def is_full(self):
#         return self.occupancy == self.page_size

#     def clear(self):
#         self.packets.fill(0)
#         self.last_update_time = time.time()
#         self.min_SN = None
#         self.max_SN = None
#         self.occupancy = 0


class SlidingBook:
    def __init__(self, num_pages:int = 15, page_size:int = 18, packet_dim:int = 4, timeout:float = 0.001):
        self.pages = {}
        self.num_pages = num_pages
        self.page_size = page_size
        self.packet_dim = packet_dim
        self.global_min_SN = 0
        self.global_max_SN = num_pages * page_size
        self.timeout = timeout


    def get_min_page_index(self):
        return self.global_min_SN // self.page_size
    
    def remove_page(self, page_index):
        if page_index in self.pages:
            page = self.pages.pop(page_index)
            packets = page.packets.copy()  # Detach packets from the page
            # Explicitly delete the page to free up memory
            del page
            if page_index == self.get_min_page_index():
                self.global_min_SN += self.page_size
                self.global_max_SN += self.page_size
            
            return packets  # Return the detached packets
        return None

    def add_packet(self, packet):
        SN = int(packet[0])
        page_index = SN // self.page_size

        if SN < self.global_min_SN or SN >= self.global_max_SN:
            min_page_index = self.get_min_page_index()
            page = self.pages.get(min_page_index)
            if page and page.last_update_time + self.timeout < time.time(): 
                # page = Page(page_size=self.page_size, packet_dim=self.packet_dim)
                # self.pages[page_index] = page
                # page.add_packet(SN, packet) 
                # self.global_max_SN = page.max_SN - self.page_size
                return self.remove_page(min_page_index)
            return None

        page = self.pages.get(page_index)
        
        if page is None:
            page = Page.Page(page_size=self.page_size, packet_dim=self.packet_dim)
            self.pages[page_index] = page


        if page.add_packet(SN, packet):
            if page.is_full():
                return self.remove_page(page_index)
        return None
    
    def get_page_index(self):
        return np.array(list(self.pages.keys()))

    def clear_all(self):
        self.pages = {}
        self.global_min_SN = 0
        self.global_max_SN = self.num_pages * self.page_size
    

class TestSlidingBook(unittest.TestCase):
    
    def setUp(self):
        self.book = SlidingBook(num_pages=3, page_size=10, packet_dim=4, timeout=0.001)
    
    def test_add_packet_within_range(self):
        packet = np.array([1, b"message", b"mac", time.time()])
        result = self.book.add_packet(packet)
        self.assertIsNone(result)
        # print('pages' , self.book.pages, 'adding res', result)
        self.assertEqual(self.book.pages[0].packets[1%self.book.pages[0].page_size][0], b'1')
    
    def test_add_packet_out_of_range(self):
        packet = np.array([31, b"message", b"mac", time.time()])
        result = self.book.add_packet(packet)  # 31 is out of the range for num_pages=3 and page_size=10
        self.assertIsNone(result)
        self.assertNotIn(3, self.book.pages)
    
    def test_remove_page(self):
        packet = np.array([5, b"message", b"mac", time.time()])
        self.book.add_packet(packet)
        removed_packets = self.book.remove_page(0)
        self.assertEqual(removed_packets[5 % 10][0], b'5')
        self.assertNotIn(0, self.book.pages)
    
    def test_add_packet_creates_new_page(self):
        packet = np.array([11, b"message", b"mac", time.time()])
        result = self.book.add_packet(packet)
        self.assertIsNone(result)
        self.assertIn(1, self.book.pages)
        self.assertEqual(self.book.pages[1].packets[11 % 10][0], b'11')
    
    def test_page_full_clears_properly(self):
        for i in range(10):
            packet = np.array([i, b"message", b"mac", time.time()])
            self.book.add_packet(packet)

        self.assertNotIn(0, self.book.pages)
        self.assertEqual(self.book.global_min_SN, 10)
        self.assertEqual(self.book.global_max_SN, 40)
    
    def test_timeout_removes_old_page(self):
        packet = np.array([1, b"message", b"mac", time.time()])
        self.book.add_packet(packet)
        time.sleep(0.002)  # sleep longer than the timeout
        packet_new = np.array([51, b"new_message", b"mac", time.time()])
        self.book.add_packet(packet_new)
        self.assertNotIn(0, self.book.pages)
        # self.assertIn(5, self.book.pages)
        # self.assertEqual(self.book.pages[5].packets[51 % 10][0], b'51')
        # self.assertEqual(self.book.global_min_SN, 10)
        # self.assertEqual(self.book.global_max_SN, 60)

    def test_timeout_when_no_page(self):
        time.sleep(0.002)  # sleep longer than the timeout
        packet_new = np.array([31, b"new_message", b"mac", time.time()])
        self.book.add_packet(packet_new)
        self.assertEqual(self.book.pages, {})




    def test_clear_all(self):
        packet = np.array([5, b"message", b"mac", time.time()])
        self.book.add_packet(packet)
        self.book.clear_all()
        self.assertEqual(self.book.pages, {})
        self.assertEqual(self.book.global_min_SN, 0)
        self.assertEqual(self.book.global_max_SN, self.book.num_pages * self.book.page_size)

if __name__ == '__main__':
    unittest.main()
