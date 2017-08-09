#!/usr/bin/env python

import unittest

from kirke.docstruct import addrutils

    
class TestAddrUtils(unittest.TestCase):

    def test_is_address_line(self):
        "Test is_address_line()"

        self.assertTrue(addrutils.is_address_line('100   Manhattanville Road'))
        self.assertTrue(addrutils.is_address_line('100 Holloway Rd'))
        self.assertTrue(addrutils.is_address_line('100 E Wisconsin Ave'))
        self.assertTrue(addrutils.is_address_line('100 Papercraft Park'))
        self.assertTrue(addrutils.is_address_line('100 Papercraft St.'))
        self.assertTrue(addrutils.is_address_line('100 Papercraft St'))
        self.assertTrue(addrutils.is_address_line('100 Papercraft Park, O’Hara   Township, PA'))
        self.assertTrue(addrutils.is_address_line('100 Executive Drive'))
        self.assertTrue(addrutils.is_address_line('100 Papercraft Park, O’Hara Township, Pennsylvania'))
        self.assertTrue(addrutils.is_address_line('100 Papercraft Wonderplace, Pa'))
        self.assertTrue(addrutils.is_address_line('123 Union Ave, Johnstown'))
        
